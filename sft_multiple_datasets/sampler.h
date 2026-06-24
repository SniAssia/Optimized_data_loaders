#pragma once
// ============================================================
//  sampler.h — Distributed index sharding + batch sampler
// ============================================================

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <vector>
#include <map>

#include <torch/torch.h>

namespace dl {

// ─────────────────────────────────────────────────────────────
// shard_indices: split dataset indices across ranks
// ─────────────────────────────────────────────────────────────
inline std::vector<int64_t> shard_indices(
    int64_t total,
    int rank,
    int world_size,
    bool shuffle,
    int64_t seed,
    int64_t epoch)
{
    std::vector<int64_t> indices(total);
    std::iota(indices.begin(), indices.end(), 0);

    if (shuffle) {
        torch::manual_seed(static_cast<uint64_t>(seed + epoch));
        auto perm = torch::randperm(total, torch::kInt64);
        auto acc  = perm.accessor<int64_t, 1>();
        for (int64_t i = 0; i < total; ++i)
            indices[i] = acc[i];
    }

    // pad so total is divisible by world_size
    int64_t remainder =
        (world_size - (static_cast<int64_t>(indices.size()) % world_size))
        % world_size;
    for (int64_t i = 0; i < remainder; ++i)
        indices.push_back(indices[i]);

    int64_t per_rank = static_cast<int64_t>(indices.size()) / world_size;
    int64_t start    = rank * per_rank;

    return std::vector<int64_t>(
        indices.begin() + start,
        indices.begin() + start + per_rank);
}

// ─────────────────────────────────────────────────────────────
// SequentialBatchSampler
//
// Groups samples into batches of `batch_size` in the order
// given by `indices` (already shuffled by shard_indices if
// cfg.shuffle=true).  No bucket logic — batches are formed
// sequentially.  The collator will trim padding per batch.
// ─────────────────────────────────────────────────────────────
class SequentialBatchSampler {
public:
    SequentialBatchSampler(
        std::vector<int64_t> indices,
        int32_t batch_size)
        : indices_(std::move(indices)),
          batch_size_(batch_size)
    {}

    std::vector<std::vector<int64_t>> build_batches() const {
        std::vector<std::vector<int64_t>> batches;
        batches.reserve(
            (indices_.size() + batch_size_ - 1) / batch_size_);

        for (std::size_t start = 0;
             start < indices_.size();
             start += static_cast<std::size_t>(batch_size_))
        {
            std::size_t end = std::min(
                start + static_cast<std::size_t>(batch_size_),
                indices_.size());
            batches.emplace_back(
                indices_.begin() + start,
                indices_.begin() + end);
        }
        return batches;
    }

    std::size_t num_batches() const {
        return (indices_.size() + batch_size_ - 1) / batch_size_;
    }

private:
    std::vector<int64_t> indices_;
    int32_t              batch_size_;
};

} // namespace dl
