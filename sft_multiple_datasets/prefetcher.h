#pragma once
// ============================================================
//  prefetcher.h — Parallel batch preparation + GPU prefetch
//
//  Architecture
//  ────────────
//  Producer thread:
//    for each batch of indices from the sampler:
//      1. fetch RawSamples from InstructionDataset (CPU RAM)
//      2. call TrimPaddingCollator → Batch (CPU tensors)
//      3. transfer Batch to GPU (non-blocking .to())
//      4. push into ready_queue (bounded by prefetch_depth)
//
//  Consumer (training loop):
//    calls next() → blocks until a GPU batch is ready
//
//  The producer runs concurrently with the training step.
//  While GPU processes batch N, producer prepares batch N+1.
//  This hides both CPU collation time and H2D transfer latency.
//
//  ready_queue depth (prefetch_depth):
//    2 = one batch on GPU being consumed, one being prepared
//    4 = more pipeline depth, more GPU memory used
// ============================================================

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <optional>
#include <queue>
#include <thread>
#include <tuple>
#include <vector>

#include <torch/torch.h>

#include "collator.h"
#include "dataset.h"
#include "sampler.h"

namespace dl {

class CUDAPrefetcher {
public:
    CUDAPrefetcher(
        const InstructionDataset&    dataset,
        SequentialBatchSampler&      sampler,
        const TrimPaddingCollator&   collator,
        torch::Device                device,
        std::size_t                  prefetch_depth = 2)
        : dataset_(dataset),
          collator_(collator),
          device_(device),
          prefetch_depth_(prefetch_depth),
          use_cuda_(device.type() == torch::kCUDA &&
                    torch::cuda::is_available())
    {
        batches_     = sampler.build_batches();
        num_batches_ = batches_.size();

        // launch producer thread
        producer_ = std::thread([this] { produce(); });
    }

    ~CUDAPrefetcher() {
        {
            std::lock_guard<std::mutex> lk(mu_);
            stop_ = true;
        }
        cv_producer_.notify_all();
        cv_consumer_.notify_all();
        if (producer_.joinable())
            producer_.join();
    }

    // Returns the next GPU-ready batch, or nullopt when epoch is done.
    std::optional<Batch> next() {
        std::unique_lock<std::mutex> lk(mu_);
        cv_consumer_.wait(lk, [&] {
            return !ready_queue_.empty() || producer_done_;
        });

        if (ready_queue_.empty())
            return std::nullopt;

        Batch b = std::move(ready_queue_.front());
        ready_queue_.pop();

        cv_producer_.notify_one();  // wake producer — slot freed
        return b;
    }

    std::size_t num_batches() const { return num_batches_; }

private:
    // Build one Batch from a list of sample indices, transfer to device.
    Batch collate_and_transfer(const std::vector<int64_t>& indices) {

        using Item = std::tuple<
            torch::Tensor,   // prompt_ids
            torch::Tensor,   // response_ids
            torch::Tensor,   // attention_mask
            int64_t,         // prompt_len
            int64_t          // response_len
        >;
        std::vector<Item> items;
        items.reserve(indices.size());

        for (auto idx : indices) {
            auto it = dataset_.get(idx);
            items.emplace_back(
                it.prompt_ids,
                it.response_ids,
                it.attention_mask,
                it.prompt_len,
                it.response_len);
        }

        Batch cpu = collator_(items);

        if (!use_cuda_) return cpu;

        // Non-blocking H2D transfer on the default CUDA stream.
        // The training loop synchronizes implicitly when it accesses
        // the tensor data, so this is safe.
        return {
            cpu.input_ids.to(device_,      /*non_blocking=*/true),
            cpu.attention_mask.to(device_, /*non_blocking=*/true),
            cpu.labels.to(device_,         /*non_blocking=*/true),
            cpu.batch_max_len,
        };
    }

    // Producer thread: iterates batches_, collates, transfers, enqueues.
    void produce() {
        for (std::size_t i = 0; i < batches_.size(); ++i) {

            // Wait until there is space in the ready queue
            {
                std::unique_lock<std::mutex> lk(mu_);
                cv_producer_.wait(lk, [&] {
                    return stop_ || ready_queue_.size() < prefetch_depth_;
                });
                if (stop_) return;
            }

            Batch b = collate_and_transfer(batches_[i]);

            {
                std::lock_guard<std::mutex> lk(mu_);
                ready_queue_.push(std::move(b));
            }
            cv_consumer_.notify_one();  // wake consumer — new batch ready
        }

        // Signal end of epoch
        {
            std::lock_guard<std::mutex> lk(mu_);
            producer_done_ = true;
        }
        cv_consumer_.notify_all();
    }

private:
    const InstructionDataset&   dataset_;
    const TrimPaddingCollator&  collator_;
    torch::Device               device_;
    std::size_t                 prefetch_depth_;
    bool                        use_cuda_;

    std::vector<std::vector<int64_t>> batches_;
    std::size_t                       num_batches_{0};

    std::mutex              mu_;
    std::condition_variable cv_consumer_;
    std::condition_variable cv_producer_;

    std::queue<Batch> ready_queue_;
    bool              producer_done_{false};
    bool              stop_{false};

    std::thread producer_;
};

} // namespace dl
