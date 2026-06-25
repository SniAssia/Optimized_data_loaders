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

        Batch cpu = collator_(items);

        if (!use_cuda_) return cpu;

        return {
            cpu.input_ids.to(device_,      /*non_blocking=*/true),
            cpu.attention_mask.to(device_, /*non_blocking=*/true),
            cpu.labels.to(device_,         /*non_blocking=*/true),
            cpu.batch_max_len,
        };
    }

    // ── Main producer thread ────────────────────────────────────────
    void produce() {
        std::mt19937 rng(static_cast<uint64_t>(cfg_.seed + epoch_));

        // shuffle shard order for this epoch
        std::vector<int> shard_order(dataset_.num_shards());
        std::iota(shard_order.begin(), shard_order.end(), 0);
        std::shuffle(shard_order.begin(), shard_order.end(), rng);

        const int W = cfg_.window_size;

        // slide window over shuffled shard order
        for (int w = 0; w < static_cast<int>(shard_order.size()); w += W) {

            if (stop_) break;

            int actual_w = std::min(W,
                static_cast<int>(shard_order.size()) - w);

            // ── Step 1: load W shards from disk in parallel ──────
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

            // collect loaded shards
            std::vector<std::vector<RawSample>> window_shards;
            window_shards.reserve(actual_w);
            for (auto& fut : load_futures)
                window_shards.push_back(fut.get());

            // ── Step 2: shuffle within each shard ───────────────
            for (auto& shard : window_shards)
                std::shuffle(shard.begin(), shard.end(), rng);

            // ── Step 3: launch N reader threads into shared queue ─
            std::size_t queue_max =
                static_cast<std::size_t>(cfg_.batch_size) * 4;
            RecordQueue rq(queue_max, actual_w);

            std::vector<std::thread> readers;
            readers.reserve(actual_w);
            for (auto& shard : window_shards)
                readers.emplace_back(reader_fn,
                                     std::ref(shard), std::ref(rq));

            // ── Step 4: consumer loop — accumulate batch, collate ─
            std::vector<const RawSample*> batch_buf;
            batch_buf.reserve(cfg_.batch_size);

            while (true) {
                const RawSample* ptr = rq.pop();

                if (ptr == nullptr) {
                    // all readers done — flush partial batch
                    if (!batch_buf.empty()) {
                        push_batch(collate_and_transfer(batch_buf));
                        batch_buf.clear();
                    }
                    break;
                }

                batch_buf.push_back(ptr);

                if (static_cast<int>(batch_buf.size()) == cfg_.batch_size) {
                    push_batch(collate_and_transfer(batch_buf));
                    batch_buf.clear();
                    if (stop_) break;
                }
            }

            // join reader threads
            for (auto& t : readers)
                t.join();

            // window_shards goes out of scope here — W shards freed
        }

        // signal end of epoch
        {
            std::lock_guard<std::mutex> lk(ready_mu_);
            producer_done_ = true;
        }
        ready_cv_consumer_.notify_all();
    }

    // push one batch into ready_queue (blocks if full)
    void push_batch(Batch b) {
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