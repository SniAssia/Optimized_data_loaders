#pragma once

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <numeric>
#include <random>
#include <vector>

namespace dl {

inline std::vector<int64_t> shard_indices(
    int64_t total,
    int      rank,
    int      world_size,
    bool     shuffle  = true,
    int64_t  seed     = 42,
    int64_t  epoch    = 0)
{
    std::vector<int64_t> idx(total);
    std::iota(idx.begin(), idx.end(), 0);

    if (shuffle) {
        std::mt19937_64 rng(static_cast<uint64_t>(seed + epoch));
        std::shuffle(idx.begin(), idx.end(), rng);
    }

    // Pad to a multiple of world_size so every rank gets equal work
    while (static_cast<int64_t>(idx.size()) % world_size != 0)
        idx.push_back(idx[0]);  // repeat first sample as dummy

    int64_t per_rank = static_cast<int64_t>(idx.size()) / world_size;
    int64_t start    = rank * per_rank;

    return std::vector<int64_t>(idx.begin() + start,
                                idx.begin() + start + per_rank);
}

class BucketBatchSampler {
public:
    // lengths     dataset.lengths() for the current mode
    // batch_size    number of samples per batch
    // boundaries        bucket upper-bounds (sorted ascending)
    // shard    indices for this rank (from shard_indices)
    // shuffle      shuffle within buckets and batch order
    // seed / epoch    RNG seed
    BucketBatchSampler(
        const std::vector<int32_t>& lengths,
        int32_t                     batch_size,
        const std::vector<int32_t>& boundaries,
        std::vector<int64_t>        shard,
        bool                        shuffle = true,
        int64_t                     seed    = 42,
        int64_t                     epoch   = 0)
        : batch_size_(batch_size)
        , shuffle_(shuffle)
        , seed_(seed)
        , epoch_(epoch)
    {
        assert(batch_size_ > 0);
        int nb = static_cast<int>(boundaries.size()) + 1;
        // Assign each shard index to a bucket
        std::vector<std::vector<int64_t>> buckets(nb);
        for (int64_t idx : shard) {
            int32_t len = lengths[static_cast<std::size_t>(idx)];
            int b = nb - 1;
            for (int k = 0; k < static_cast<int>(boundaries.size()); ++k) {
                if (len <= boundaries[k]) { b = k; break; }
            }
            buckets[b].push_back(idx);
        }
        build_batches(buckets);
    }

    const std::vector<std::vector<int64_t>>& batches() const {
        return batches_;
    }
    std::size_t num_batches() const { return batches_.size(); }
private:
    int32_t batch_size_;
    bool    shuffle_;
    int64_t seed_;
    int64_t epoch_;
    std::vector<std::vector<int64_t>> batches_;
    void build_batches(std::vector<std::vector<int64_t>>& buckets) {
        std::mt19937_64 rng(static_cast<uint64_t>(seed_ + epoch_));
        for (auto& bucket : buckets) {
            if (bucket.empty()) continue;
            if (shuffle_)
                std::shuffle(bucket.begin(), bucket.end(), rng);

            for (std::size_t i = 0; i < bucket.size(); i += batch_size_) {
                std::size_t end = std::min(i + static_cast<std::size_t>(batch_size_),
                                           bucket.size());
                batches_.emplace_back(bucket.begin() + i, bucket.begin() + end);
            }
        }
        if (shuffle_)
            std::shuffle(batches_.begin(), batches_.end(), rng);
    }
};

} 






















// BucketBatchSampler bins samples by sequence length so that
// samples in the same batch have similar lengths, minimising
// padding waste.  It is completely mode-agnostic: it only
// operates on a lengths array and index counts
// Compatible with all TrainingMode values — the UnifiedDataset
// already returns the right length definition per mode
