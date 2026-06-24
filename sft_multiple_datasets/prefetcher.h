#pragma once
// ============================================================
//  prefetcher.h — Parallel batch preparation + GPU prefetch
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
        producer_    = std::thread([this] { produce(); });
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

    std::optional<Batch> next() {
        std::unique_lock<std::mutex> lk(mu_);
        cv_consumer_.wait(lk, [&] {
            return !ready_queue_.empty() || producer_done_;
        });

        if (ready_queue_.empty())
            return std::nullopt;

        Batch b = std::move(ready_queue_.front());
        ready_queue_.pop();
        cv_producer_.notify_one();
        return b;
    }

    std::size_t num_batches() const { return num_batches_; }

private:
    Batch collate_and_transfer(const std::vector<int64_t>& indices) {

        // Item: prompt_ids, response_ids, prompt_len, response_len
        // No attention_mask — built by collator at batch time
        using Item = std::tuple
            torch::Tensor,   // prompt_ids   [prompt_len]
            torch::Tensor,   // response_ids [response_len]
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
                it.prompt_len,
                it.response_len);
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

    void produce() {
        for (std::size_t i = 0; i < batches_.size(); ++i) {
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
            cv_consumer_.notify_one();
        }

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