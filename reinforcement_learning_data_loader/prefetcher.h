#pragma once
#include "dataset.h"
#include "sampler.h"
#include "collator.h"
#include<atomic>
#include<condition_variable>
#include<cstddef>
#include<functional>
#include<memory>
#include<mutex>
#include <optional>
#include<queue>
#include<thread>
#include <vector>
#include<torch/torch.h>

namespace dl {

//moving a batch's tensors to a device (pin + async copy)
namespace detail {
inline torch::Tensor to_device(const torch::Tensor& t, torch::Device dev) {
    if (dev.is_cuda())
        return t.pin_memory().to(dev, /*non_blocking=*/true);
    return t.to(dev);
}
// SFTBatch
inline SFTBatch to_device(const SFTBatch& b, torch::Device dev) {
    return {
        to_device(b.input_ids, dev),
        to_device(b.attention_mask,dev),
        to_device(b.labels, dev)
    };
}
// RewardModelBatch
inline RewardModelBatch to_device(const RewardModelBatch& b, torch::Device dev) {
    return {
        to_device(b.chosen_input_ids, dev),
        to_device(b.chosen_attention_mask,dev),
        to_device(b.rejected_input_ids, dev),
        to_device(b.rejected_attention_mask,dev)
    };
}

// DPOBatch
inline DPOBatch to_device(const DPOBatch& b, torch::Device dev) {
    return {
        to_device(b.chosen_input_ids, dev),
        to_device(b.chosen_attention_mask, dev),
        to_device(b.chosen_labels, dev),
        to_device(b.rejected_input_ids,dev),
        to_device(b.rejected_attention_mask,dev),
        to_device(b.rejected_labels,dev)
    };
}

// RolloutBatch
inline RolloutBatch to_device(const RolloutBatch& b, torch::Device dev) {
    return {
        to_device(b.prompt_ids,dev),
        to_device(b.prompt_mask, dev)
    };
}

} 

template<typename BatchT>
class TypedPrefetcher {
public:
    // CollatorFn: a callable (vector<items>) → BatchT  on CPU
    using CollatorFn = std::function<BatchT(const std::vector<int64_t>&)>;

    TypedPrefetcher(
        CollatorFn      collate_fn,
        BucketBatchSampler& sampler,
        torch::Device   device,
        int             num_workers    = 4,
        std::size_t     prefetch_factor = 2)
        :collate_fn_(std::move(collate_fn))
        ,device_(device)
        ,prefetch_factor_(prefetch_factor)
        ,done_(false)
        ,stop_(false){
        batch_list_ = sampler.batches();
        cursor_      = 0;
        for (int i = 0; i < num_workers; ++i)
            workers_.emplace_back([this]{ worker_loop(); });
    }

    ~TypedPrefetcher() {
        {
            std::unique_lock<std::mutex> lk(mu_);
            stop_ = true;
        }
        cv_produce_.notify_all();
        cv_consume_.notify_all();
        for (auto& w : workers_) w.join();
    }

    std::optional<BatchT> next() {
        std::unique_lock<std::mutex> lk(mu_);
        cv_consume_.wait(lk, [this]{
            return !queue_.empty() || done_;
        });
        if (queue_.empty()) return std::nullopt;
        auto batch = std::move(queue_.front());
        queue_.pop();
        lk.unlock();
        cv_produce_.notify_one();
        return batch;
    }
    std::size_t num_batches() const { return batch_list_.size(); }
private:
    CollatorFn collate_fn_;
    torch::Device device_;
    std::size_t prefetch_factor_;
    std::vector<std::vector<int64_t>> batch_list_;
    std::size_t  cursor_;
    std::mutex mu_;
    std::condition_variable cv_produce_;
    std::condition_variable cv_consume_;
    std::queue<BatchT> queue_;
    std::atomic<bool> done_;
    std::atomic<bool> stop_;
    std::vector<std::thread> workers_;
    void worker_loop() {
        while (true) {
            std::vector<int64_t> indices;
            {
                std::unique_lock<std::mutex> lk(mu_);
                cv_produce_.wait(lk, [this]{
                    return stop_ || cursor_ >= batch_list_.size() ||
                           queue_.size() < prefetch_factor_;
                });
                if (stop_) return;
                if (cursor_ >= batch_list_.size()) {
                    done_ = true;
                    cv_consume_.notify_all();
                    return;
                }
                indices = batch_list_[cursor_++];
            }

            // Build batch on CPU (outside lock)
            BatchT cpu_batch = collate_fn_(indices);
            // Move to device
            BatchT gpu_batch = detail::to_device(cpu_batch, device_);

            {
                std::unique_lock<std::mutex> lk(mu_);
                queue_.push(std::move(gpu_batch));
            }
            cv_consume_.notify_one();
        }
    }
};

// Concrete type aliases for the four modes
using SFTPrefetcher = TypedPrefetcher<SFTBatch>;
using RewardModelPrefetcher = TypedPrefetcher<RewardModelBatch>;
using DPOPrefetcher = TypedPrefetcher<DPOBatch>;
using RolloutPrefetcher = TypedPrefetcher<RolloutBatch>;
// Backward-compat alias — legacy code used CUDAPrefetcher<SFTBatch>
using CUDAPrefetcher = SFTPrefetcher;

} 
















// prefetcher.h — Multi-threaded CUDA Prefetcher
// CUDAPrefetcher runs a background thread pool that stays
// prefetch_factor batches ahead of the training loop.
// It is templated on the Batch type so it works with all four
// batch structs (SFTBatch, RewardModelBatch, DPOBatch,
// RolloutBatch) without modification to its core logic.
// The training loop calls .next() which blocks only when the
// prefetch queue is empty.  Returns std::nullopt at epoch end.
