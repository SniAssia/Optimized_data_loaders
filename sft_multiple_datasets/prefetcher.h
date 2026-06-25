#pragma once
//  prefetcher.h — Window-based streaming prefetcher
//
//  Architecture per window of N shards
//  ─────────────────────────────────────
//
//  1. Load N shards from disk in parallel (N std::async tasks)
//  2. Shuffle records within each shard independently
//  3. Launch N reader threads, each pushing records from one
//     shard into a shared bounded queue
//  4. Consumer thread pulls batch_size records from shared queue,
//     calls TrimPaddingCollator, pushes Batch to ready_queue
//  5. Training loop calls next() → gets GPU-ready Batch
//
//  RAM at any moment:
//    N shards in memory  ≈ N × shard_size × avg_record_size
//    shared_queue        ≈ 4 × batch_size records
//    ready_queue         ≈ prefetch_depth Batch objects (CPU)
#include <iomanip>
#include <chrono>
#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <future>
#include <mutex>
#include <numeric>
#include <optional>
#include <queue>
#include <random>
#include <thread>
#include <vector>

#include <torch/torch.h>

#include "collator.h"
#include "dataset.h"

namespace dl {

// sentinel pushed by reader threads when their shard is exhausted
static const RawSample* _SHARD_DONE = nullptr;

class CUDAPrefetcher {
public:
    CUDAPrefetcher(
        const InstructionDataset&  dataset,
        const TrimPaddingCollator& collator,
        torch::Device              device,
        int64_t                    epoch          = 0,
        std::size_t                prefetch_depth = 2)
        : dataset_(dataset),
          collator_(collator),
          device_(device),
          cfg_(dataset.config()),
          epoch_(epoch),
          prefetch_depth_(prefetch_depth),
          use_cuda_(device.type() == torch::kCUDA &&
                    torch::cuda::is_available())
    {
        producer_ = std::thread([this] { produce(); });
    }

    ~CUDAPrefetcher() {
        {
            std::lock_guard<std::mutex> lk(ready_mu_);
            stop_ = true;
        }
        ready_cv_producer_.notify_all();
        ready_cv_consumer_.notify_all();
        if (producer_.joinable())
            producer_.join();
    }

    // Returns next GPU-ready batch, or nullopt when epoch is done
    std::optional<Batch> next() {
        std::unique_lock<std::mutex> lk(ready_mu_);
        ready_cv_consumer_.wait(lk, [&] {
            return !ready_queue_.empty() || producer_done_;
        });

        if (ready_queue_.empty())
            return std::nullopt;

        Batch b = std::move(ready_queue_.front());
        ready_queue_.pop();
        ready_cv_producer_.notify_one();
        return b;
    }

private:
    // ── Shared record queue (reader threads → consumer thread) ───
    struct RecordQueue {
        std::queue<const RawSample*> q;
        std::mutex                   mu;
        std::condition_variable      cv_push;
        std::condition_variable      cv_pop;
        std::size_t                  max_size;
        int                          n_done = 0;   // reader threads finished
        int                          n_readers = 0;

        explicit RecordQueue(std::size_t max_sz, int readers)
            : max_size(max_sz), n_readers(readers) {}

        void push(const RawSample* ptr) {
            std::unique_lock<std::mutex> lk(mu);
            cv_push.wait(lk, [&] { return q.size() < max_size; });
            q.push(ptr);
            cv_pop.notify_one();
        }

        // push sentinel — called when a reader thread finishes
        void push_done() {
            std::lock_guard<std::mutex> lk(mu);
            ++n_done;
            cv_pop.notify_all();
        }

        // returns nullptr when all readers are done and queue is empty
        const RawSample* pop() {
            std::unique_lock<std::mutex> lk(mu);
            cv_pop.wait(lk, [&] {
                return !q.empty() || n_done == n_readers;
            });
            if (q.empty()) return nullptr;
            auto* ptr = q.front();
            q.pop();
            cv_push.notify_one();
            return ptr;
        }

        bool exhausted() {
            std::lock_guard<std::mutex> lk(mu);
            return q.empty() && n_done == n_readers;
        }
    };

    // ── Reader thread: pushes all records from one shard into queue ─
    static void reader_fn(
        const std::vector<RawSample>& shard,
        RecordQueue&                  rq)
    {
        for (const auto& sample : shard)
            rq.push(&sample);
        rq.push_done();
    }

    // ── Collate batch_size raw samples → padded Batch on device ────
    Batch collate_and_transfer(
    const std::vector<const RawSample*>& ptrs)
    {
        if (ptrs.empty()) {
            // return empty sentinel batch — will be filtered by consumer
            return { torch::zeros({0,0}, torch::kInt64),
                    torch::zeros({0,0}, torch::kInt64),
                    torch::zeros({0,0}, torch::kInt64),
                    0 };
        }
        using CollatorItem = std::tuple
            torch::Tensor,   // prompt_ids   [prompt_len]
            torch::Tensor,   // response_ids [response_len]
            int64_t,         // prompt_len
            int64_t          // response_len
        >;

        std::vector<CollatorItem> items;
        items.reserve(ptrs.size());

        for (const auto* s : ptrs) {
            auto item = InstructionDataset::to_item(*s);
            items.emplace_back(
                std::move(item.prompt_ids),
                std::move(item.response_ids),
                item.prompt_len,
                item.response_len);
        }

        // ── time collation + H2D transfer ────────────────────────
        auto t0 = std::chrono::high_resolution_clock::now();

        Batch cpu = collator_(items);

        Batch result = cpu;
        if (use_cuda_) {
            result = {
                cpu.input_ids.to(device_,      /*non_blocking=*/true),
                cpu.attention_mask.to(device_, /*non_blocking=*/true),
                cpu.labels.to(device_,         /*non_blocking=*/true),
                cpu.batch_max_len,
            };
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double,std::milli>(t1 - t0).count();
        std::cout << "[producer] collate+transfer: " << ms << " ms  "
                << "batch_max_len=" << result.batch_max_len << "\n";

        return result;
    }
    void produce() {
        std::mt19937 rng(static_cast<uint64_t>(cfg_.seed + epoch_));

        // ── epoch-level accumulators ──────────────────────────────
        double total_shuffle_ms  = 0.0;
        double total_load_ms     = 0.0;
        double total_intra_ms    = 0.0;
        double total_batch_ms    = 0.0;
        int    total_windows     = 0;
        int    total_batches     = 0;
        int    total_records_proc = 0;

        auto epoch_start = std::chrono::high_resolution_clock::now();

        // ── time: shard order shuffle ─────────────────────────────
        auto t0 = std::chrono::high_resolution_clock::now();

        std::vector<int> shard_order(dataset_.num_shards());
        std::iota(shard_order.begin(), shard_order.end(), 0);
        std::shuffle(shard_order.begin(), shard_order.end(), rng);

        total_shuffle_ms = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t0).count();

        std::cout << "[producer] shard order shuffle: "
                << total_shuffle_ms << " ms"
                << "  (" << shard_order.size() << " shards)\n";

        const int W         = cfg_.window_size;
        int       window_idx = 0;

        for (int w = 0; w < static_cast<int>(shard_order.size()); w += W) {

            if (stop_) break;

            int actual_w = std::min(W,
                static_cast<int>(shard_order.size()) - w);

            std::cout << "\n[producer] === window " << window_idx
                    << " (shards " << w << ".." << w + actual_w - 1
                    << " of shuffled order) ===\n";

            // ── Step 1: load W shards from disk in parallel ──────
            auto t_load = std::chrono::high_resolution_clock::now();

            std::vector<std::future<std::vector<RawSample>>> load_futures;
            load_futures.reserve(actual_w);
            for (int k = 0; k < actual_w; ++k) {
                int sidx = shard_order[w + k];
                load_futures.push_back(
                    std::async(std::launch::async,
                        [this, sidx] {
                            return dataset_.load_shard(sidx);
                        }));
            }

            std::vector<std::vector<RawSample>> window_shards;
            window_shards.reserve(actual_w);
            for (auto& fut : load_futures)
                window_shards.push_back(fut.get());

            double load_ms = std::chrono::duration<double, std::milli>(
                std::chrono::high_resolution_clock::now() - t_load).count();
            total_load_ms += load_ms;

            int window_records = 0;
            for (const auto& s : window_shards)
                window_records += static_cast<int>(s.size());
            total_records_proc += window_records;

            std::cout << "[producer] disk load:           "
                    << load_ms << " ms"
                    << "  records=" << window_records
                    << "  (" << actual_w << " shards in parallel)\n";

            // ── Step 2: shuffle within each shard ───────────────
            auto t_intra = std::chrono::high_resolution_clock::now();

            for (auto& shard : window_shards)
                std::shuffle(shard.begin(), shard.end(), rng);

            double intra_ms = std::chrono::duration<double, std::milli>(
                std::chrono::high_resolution_clock::now() - t_intra).count();
            total_intra_ms += intra_ms;

            std::cout << "[producer] intra-shard shuffle: "
                    << intra_ms << " ms"
                    << "  (" << actual_w << " shards)\n";

            // ── Step 3: reader threads + batch construction ──────
            auto t_batch = std::chrono::high_resolution_clock::now();

            std::size_t queue_max =
                static_cast<std::size_t>(cfg_.batch_size) * 4;
            RecordQueue rq(queue_max, actual_w);

            std::vector<std::thread> readers;
            readers.reserve(actual_w);
            for (auto& shard : window_shards)
                readers.emplace_back(reader_fn,
                                    std::ref(shard), std::ref(rq));

            std::vector<const RawSample*> batch_buf;
            batch_buf.reserve(cfg_.batch_size);
            int batches_this_window = 0;

            while (true) {
                const RawSample* ptr = rq.pop();

                if (ptr == nullptr) {
                    if (!batch_buf.empty()) {
                        auto b = collate_and_transfer(batch_buf);
                        if (b.batch_max_len > 0) {
                            push_batch(std::move(b));
                            ++batches_this_window;
                        }
                        batch_buf.clear();
                    }
                    break;
                }

                batch_buf.push_back(ptr);

                if (static_cast<int>(batch_buf.size()) == cfg_.batch_size) {
                    auto b = collate_and_transfer(batch_buf);
                    if (b.batch_max_len > 0) {
                        push_batch(std::move(b));
                        ++batches_this_window;
                    }
                    batch_buf.clear();
                    if (stop_) break;
                }
            }

            for (auto& t : readers)
                t.join();

            double batch_ms = std::chrono::duration<double, std::milli>(
                std::chrono::high_resolution_clock::now() - t_batch).count();
            total_batch_ms  += batch_ms;
            total_batches   += batches_this_window;

            std::cout << "[producer] batch construction:  "
                    << batch_ms << " ms"
                    << "  batches=" << batches_this_window << "\n";

            std::cout << "[producer] window " << window_idx << " total: "
                    << (load_ms + intra_ms + batch_ms) << " ms"
                    << "  breakdown:"
                    << "  load="    << load_ms
                    << "  shuffle=" << intra_ms
                    << "  batches=" << batch_ms << "\n";

            ++window_idx;
            ++total_windows;
        }

        // ── epoch summary ─────────────────────────────────────────
        double epoch_ms = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - epoch_start).count();

        double total_pipeline_ms = total_shuffle_ms
                                + total_load_ms
                                + total_intra_ms
                                + total_batch_ms;

        std::cout << "=========\n";
        std::cout << "      PRODUCER EPOCH TIMING SUMMARY           ║\n";
        std::cout << std::fixed << std::setprecision(2);
        std::cout << " Windows processed   : " << total_windows     << "\n";
        std::cout << "  Total batches       : " << total_batches      << "\n";
        std::cout << "  Total records       : " << total_records_proc << "\n";
        std::cout << "  PHASE BREAKDOWN\n";
        std::cout << " shard order shuffle : " << total_shuffle_ms
                << " ms  ("
                << (100.0 * total_shuffle_ms / total_pipeline_ms)
                << "%)\n";
        std::cout << " disk load (parallel): " << total_load_ms
                << " ms  ("
                << (100.0 * total_load_ms / total_pipeline_ms)
                << "%)\n";
        std::cout << "  intra-shard shuffle : " << total_intra_ms
                << " ms  ("
                << (100.0 * total_intra_ms / total_pipeline_ms)
                << "%)\n";
        std::cout << "  batch construction  : " << total_batch_ms
                << " ms  ("
                << (100.0 * total_batch_ms / total_pipeline_ms)
                << "%)\n";
        std::cout << "\n";
        std::cout << "  Pipeline total      : " << total_pipeline_ms << " ms\n";
        std::cout << "  Epoch wall time     : " << epoch_ms          << " ms\n";
        std::cout << "  Overhead (queue/sync): "
                << (epoch_ms - total_pipeline_ms) << " ms  ("
                << (100.0 * (epoch_ms - total_pipeline_ms) / epoch_ms)
                << "%)\n";
        std::cout << "║\n";
        std::cout << " Throughput\n";
        std::cout << "  records/s : "
                << (int)(total_records_proc / (epoch_ms / 1000.0)) << "\n";
        std::cout << "  batches/s : "
                << (int)(total_batches / (epoch_ms / 1000.0)) << "\n";
        std::cout << "═════════════\n";
        {
            std::lock_guard<std::mutex> lk(ready_mu_);
            producer_done_ = true;
        }
        ready_cv_consumer_.notify_all();
    }
    


    // push one batch into ready_queue (blocks if full)
    void push_batch(Batch b) {
        if (b.batch_max_len == 0) return;   // ← discard empty batches
        std::unique_lock<std::mutex> lk(ready_mu_);
        ready_cv_producer_.wait(lk, [&] {
            return stop_ || ready_queue_.size() < prefetch_depth_;
        });
        if (stop_) return;
        ready_queue_.push(std::move(b));
        ready_cv_consumer_.notify_one();
    }

private:
    const InstructionDataset&   dataset_;
    const TrimPaddingCollator&  collator_;
    torch::Device               device_;
    DataLoaderConfig            cfg_;
    int64_t                     epoch_;
    std::size_t                 prefetch_depth_;
    bool                        use_cuda_;

    // ready queue (producer → training loop)
    std::mutex              ready_mu_;
    std::condition_variable ready_cv_consumer_;
    std::condition_variable ready_cv_producer_;
    std::queue<Batch>       ready_queue_;
    bool                    producer_done_{false};
    bool                    stop_{false};

    std::thread producer_;
};

} // namespace dl