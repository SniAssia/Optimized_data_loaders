#pragma once


#include <atomic>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <thread>
#include <vector>

#include <torch/torch.h>

#include "collator.h"
#include "dataset.h"
#include "sampler.h"

namespace dl {

struct WorkerPool {
    explicit WorkerPool(int /*n*/) {}
};

class CUDAPrefetcher {
public:
    CUDAPrefetcher(
        const InstructionDataset& dataset,
        BucketBatchSampler& sampler,
        const DynamicPaddingCollator& collator,
        torch::Device device,
        int num_workers = 4,
        std::size_t prefetch_depth = 2)
        : dataset_(dataset),
          collator_(collator),
          device_(device),
          prefetch_depth_(prefetch_depth),
          use_cuda_(device.type() == torch::kCUDA && torch::cuda::is_available()),
          worker_pool_(std::max(1, num_workers))
    {
        batches_     = sampler.build_batches();
        num_batches_ = batches_.size();
        producer_ = std::thread([this] { this->produce(); });
    }

    ~CUDAPrefetcher() {
        {
            std::lock_guard<std::mutex> lk(mu_);
            stop_ = true;
        }
        cv_consumer_.notify_all();
        cv_producer_.notify_all();

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

        Batch batch = std::move(ready_queue_.front());
        ready_queue_.pop();

        cv_producer_.notify_one();
        lk.unlock();

        return batch;
    }

    std::size_t num_batches() const { return num_batches_; }

private:
    Batch collate_and_transfer(const std::vector<int64_t>& indices) {

        // FIX 2+3: build vector<pair<Tensor,Tensor>> using the correct
        // Item fields (.instruction_ids / .response_ids), not .input_ids
        std::vector<std::pair<torch::Tensor, torch::Tensor>> pairs;
        pairs.reserve(indices.size());

        for (auto idx : indices) {
            auto item = dataset_.get(idx);
            pairs.emplace_back(item.instruction_ids, item.response_ids);
        }

        // FIX 2: collator now receives the correct type
        Batch cpu = collator_(pairs);

        if (!use_cuda_) return cpu;

        // FIX 1: plain .to() on the default stream — no CUDAStreamGuard needed
        return {
            cpu.input_ids.to(device_, /*non_blocking=*/true),
            cpu.attention_mask.to(device_, /*non_blocking=*/true),
            cpu.labels.to(device_, /*non_blocking=*/true),
        };
    }

    void produce() {
        for (size_t i = 0; i < batches_.size(); ++i) {

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
    const InstructionDataset& dataset_;
    const DynamicPaddingCollator& collator_;
    torch::Device device_;

    std::size_t prefetch_depth_;
    bool use_cuda_;

    WorkerPool worker_pool_;

    std::vector<std::vector<int64_t>> batches_;
    std::size_t num_batches_{0};

    // FIX 1: transfer_stream_ removed entirely

    std::mutex mu_;
    std::condition_variable cv_consumer_;
    std::condition_variable cv_producer_;

    std::queue<Batch> ready_queue_;
    bool producer_done_{false};
    bool stop_{false};

    std::thread producer_;
};

} // namespace dl














//  FIX 1: Removed c10::cuda::CUDAStream / c10::cuda::getStreamFromPool()
//          and at::cuda::CUDAStreamGuard — these APIs are not available
//          in this build of LibTorch.  H2D transfer is done on the
//          default CUDA stream, which is correct and safe here because
//          each batch is fully assembled on CPU before the transfer
//          begins and the training loop calls next() sequentially.
//
//  FIX 2: collate_and_transfer now passes vector<pair<Tensor,Tensor>>
//          to DynamicPaddingCollator, matching its operator() signature.
//          Previously it passed vector<Tensor> (single tensor per item),
//          which caused a type mismatch.
//
//  FIX 3: dataset_.get(idx) returns Item{instruction_ids, response_ids, …};
//          we now correctly access .instruction_ids / .response_ids instead
//          of the non-existent .input_ids field.
// ============================================================
