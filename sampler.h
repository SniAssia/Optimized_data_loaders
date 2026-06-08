#pragma once
#include <algorithm>
#include <cstdint>
#include <map>
#include <numeric>
#include <stdexcept>
#include <vector>

#include <torch/torch.h>

namespace dl {

inline std::vector<int64_t> shard_indices(
    int64_t total,
    int rank, // index of the current GPU 
    int world_size, // total number of gpus 
    bool shuffle,
    int64_t seed,
    int64_t epoch)
{
    std::vector<int64_t> indices(total);
    std::iota(indices.begin(), indices.end(), 0);

    if (shuffle) {
        // FIX: use torch::manual_seed + torch::randperm with no generator argument
        torch::manual_seed(static_cast<uint64_t>(seed + epoch));

        auto perm = torch::randperm(total, torch::kInt64);

        auto acc = perm.accessor<int64_t, 1>();
        for (int64_t i = 0; i < total; ++i)
            indices[i] = acc[i];
    }

    int64_t remainder =
        (world_size - (static_cast<int64_t>(indices.size()) % world_size)) % world_size;

    for (int64_t i = 0; i < remainder; ++i)
        indices.push_back(indices[i]);

    int64_t per_rank = static_cast<int64_t>(indices.size()) / world_size;
    int64_t start    = rank * per_rank;

    return std::vector<int64_t>(
        indices.begin() + start,
        indices.begin() + start + per_rank
    );
}

class BucketBatchSampler {
public:
    BucketBatchSampler(
        const std::vector<int32_t>& lengths,
        int32_t batch_size,
        std::vector<int32_t> boundaries,
        std::vector<int64_t> indices,
        bool shuffle = true,
        int64_t seed = 42,
        int64_t epoch = 0)
        : lengths_(lengths),
          batch_size_(batch_size),
          boundaries_(std::move(boundaries)),
          indices_(std::move(indices)),
          shuffle_(shuffle),
          seed_(seed),
          epoch_(epoch)
    {
        std::sort(boundaries_.begin(), boundaries_.end());
    }

    void set_epoch(int64_t epoch) { epoch_ = epoch; }

    std::vector<std::vector<int64_t>> build_batches() const
    {
        // FIX: set global seed once at the top, no generator object
        torch::manual_seed(static_cast<uint64_t>(seed_ + epoch_));

        // ── 1. Bucket assignment ───────────────────────────────
        std::map<int32_t, std::vector<int64_t>> buckets;
        for (int32_t b : boundaries_)
            buckets[b] = {};

        for (int64_t idx : indices_) {
            int32_t len = lengths_[static_cast<size_t>(idx)];
            int32_t b   = assign_bucket(len);
            buckets[b].push_back(idx);
        }

        std::vector<std::vector<int64_t>> batches;

        // ── 2. Shuffle inside buckets + batch ─────────────────
        for (auto& [b, b_indices] : buckets) {
            if (b_indices.empty()) continue;

            if (shuffle_) {
                auto perm = torch::randperm(
                    static_cast<int64_t>(b_indices.size()),
                    torch::kInt64
                );

                auto acc = perm.accessor<int64_t, 1>();
                std::vector<int64_t> shuffled;
                shuffled.reserve(b_indices.size());
                for (int64_t i = 0; i < (int64_t)b_indices.size(); ++i)
                    shuffled.push_back(b_indices[acc[i]]);
                b_indices = std::move(shuffled);
            }

            for (size_t start = 0; start < b_indices.size(); start += batch_size_) {
                size_t end = std::min(start + (size_t)batch_size_, b_indices.size());
                batches.emplace_back(
                    b_indices.begin() + start,
                    b_indices.begin() + end
                );
            }
        }

        // ── 3. Shuffle batch order ────────────────────────────
        if (shuffle_ && !batches.empty()) {
            auto perm = torch::randperm(
                static_cast<int64_t>(batches.size()),
                torch::kInt64
            );

            auto acc = perm.accessor<int64_t, 1>();
            std::vector<std::vector<int64_t>> shuffled;
            shuffled.reserve(batches.size());
            for (int64_t i = 0; i < (int64_t)batches.size(); ++i)
                shuffled.push_back(std::move(batches[acc[i]]));
            batches = std::move(shuffled);
        }

        return batches;
    }

    std::size_t num_batches() const {
        return (indices_.size() + batch_size_ - 1) / batch_size_;
    }

private:
    int32_t assign_bucket(int32_t length) const {
        for (int32_t b : boundaries_)
            if (length <= b) return b;
        return boundaries_.back();
    }

private:
    const std::vector<int32_t>& lengths_;
    int32_t batch_size_;
    std::vector<int32_t> boundaries_;
    std::vector<int64_t> indices_;
    bool shuffle_;
    int64_t seed_;
    int64_t epoch_;
};

} // namespace dl