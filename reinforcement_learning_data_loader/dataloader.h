#pragma once
//  dataloader.h — Mode-aware factory
//Wires together: UnifiedDatasetReader -> UnifiedDataset
//   -> shard_indices -> BucketBatchSampler
//          -> UnifiedCollator -> TypedPrefetcher<BatchT>
// .  One build_*_dataloader() function per mode, plus a single
//entry point build_dataloader() that dispatches on cfg.mode
//  and returns the SFT prefetcher (legacy path) for backward
//   compatibility. For RL / preference modes, call the mode
//specific factory directly so the return type is concrete.

#include "dataset.h"
#include "sampler.h"
#include "collator.h"
#include "prefetcher.h"
#include "distributed.h"

#include <memory>

namespace dl {
// build dataset + sampler for any mode
namespace detail {
inline std::shared_ptr<BucketBatchSampler> make_sampler(
    const UnifiedDataset& dataset,
    const DataLoaderConfig& cfg,
    int rank, int world_size, int64_t epoch)
{
    auto shard = shard_indices(
        static_cast<int64_t>(dataset.size()),
        rank, world_size,
        cfg.shuffle, cfg.seed, epoch);

    const std::vector<int32_t> boundaries = cfg.bucket_boundaries;

    return std::make_shared<BucketBatchSampler>(
        dataset.lengths(),
        cfg.batch_size,
        boundaries,
        std::move(shard),
        cfg.shuffle,
        cfg.seed,
        epoch);
}

} 
// SFT dataloader
inline std::unique_ptr<SFTPrefetcher> build_sft_dataloader(
      std::shared_ptr<UnifiedDataset> dataset,
    const DataLoaderConfig& cfg,
      int rank = 0, int world_size = 1, int64_t epoch = 0,
    torch::Device device = torch::kCPU)
{
    auto sampler = detail::make_sampler(*dataset, cfg, rank, world_size, epoch);
    auto collator = std::make_shared<UnifiedCollator>(cfg);

    SFTPrefetcher::CollatorFn fn =
        [dataset, collator](const std::vector<int64_t>& indices) -> SFTBatch {
            std::vector<UnifiedDataset::SFTItem> items;
            items.reserve(indices.size());
            for (auto idx : indices)
                items.push_back(dataset->get_sft(idx));
            return collator->collate_sft(items);
        };

    return std::make_unique<SFTPrefetcher>(
        std::move(fn), *sampler, device,
        cfg.num_workers, static_cast<std::size_t>(cfg.prefetch_factor));
}
// Reward model dataloader (prompt, chosen, rejected), no labels
inline std::unique_ptr<RewardModelPrefetcher> build_reward_model_dataloader(
    std::shared_ptr<UnifiedDataset> dataset,
    const DataLoaderConfig& cfg,
    int rank = 0, int world_size = 1, int64_t epoch = 0,
    torch::Device device = torch::kCPU)
{
    auto sampler = detail::make_sampler(*dataset, cfg, rank, world_size, epoch);
    auto collator = std::make_shared<UnifiedCollator>(cfg);

    RewardModelPrefetcher::CollatorFn fn =
        [dataset, collator](const std::vector<int64_t>& indices) -> RewardModelBatch {
            std::vector<UnifiedDataset::PreferenceItem> items;
            items.reserve(indices.size());
            for (auto idx : indices)
                items.push_back(dataset->get_preference(idx));
            return collator->collate_reward_model(items);
        };

    return std::make_unique<RewardModelPrefetcher>(
        std::move(fn), *sampler, device,
        cfg.num_workers, static_cast<std::size_t>(cfg.prefetch_factor));
}
// DPO dataloader — (prompt, chosen, rejected), with labels
inline std::unique_ptr<DPOPrefetcher> build_dpo_dataloader(
    std::shared_ptr<UnifiedDataset> dataset,
    const DataLoaderConfig& cfg,
    int rank = 0, int world_size = 1, int64_t epoch = 0,
    torch::Device device = torch::kCPU)
{
    auto sampler = detail::make_sampler(*dataset, cfg, rank, world_size, epoch);
    auto collator = std::make_shared<UnifiedCollator>(cfg);

    DPOPrefetcher::CollatorFn fn =
        [dataset, collator](const std::vector<int64_t>& indices) -> DPOBatch {
            std::vector<UnifiedDataset::PreferenceItem> items;
            items.reserve(indices.size());
            for (auto idx : indices)
                items.push_back(dataset->get_preference(idx));
            return collator->collate_dpo(items);
        };

    return std::make_unique<DPOPrefetcher>(
        std::move(fn), *sampler, device,
        cfg.num_workers, static_cast<std::size_t>(cfg.prefetch_factor));
}
// Rollout dataloader — prompt-only, left-padded + gen space
// Used by both PPO and GRPO. cfg.grpo_group_size is carried in
// the config purely for the training loop's benefit; the data
// loader itself never reads it.
inline std::unique_ptr<RolloutPrefetcher> build_rollout_dataloader(
    std::shared_ptr<UnifiedDataset> dataset,
    const DataLoaderConfig& cfg,
    int rank = 0, int world_size = 1, int64_t epoch = 0,
    torch::Device device = torch::kCPU)
{
    auto sampler = detail::make_sampler(*dataset, cfg, rank, world_size, epoch);
    auto collator = std::make_shared<UnifiedCollator>(cfg);

    RolloutPrefetcher::CollatorFn fn =
        [dataset, collator](const std::vector<int64_t>& indices) -> RolloutBatch {
            std::vector<UnifiedDataset::RolloutItem> items;
            items.reserve(indices.size());
            for (auto idx : indices)
                items.push_back(dataset->get_rollout(idx));
            return collator->collate_rollout(items);
        };

    return std::make_unique<RolloutPrefetcher>(
        std::move(fn), *sampler, device,
        cfg.num_workers, static_cast<std::size_t>(cfg.prefetch_factor));
}
// Single entry point for callers that load data and then pick
// the right factory based on cfg.mode at runtime.
// Usage:
//   auto samples = UnifiedDatasetReader::load(cfg);
//   auto dataset = std::make_shared<UnifiedDataset>(std::move(samples), cfg);
//
//   switch (cfg.mode) {
//     case TrainingMode::SFT:
//       { auto pf = build_sft_dataloader(dataset, cfg, ...); ... }
//     case TrainingMode::REWARD_MODEL:
//       { auto pf = build_reward_model_dataloader(dataset, cfg, ...); ... }
//     case TrainingMode::DPO:
//       { auto pf = build_dpo_dataloader(dataset, cfg, ...); ... }
//     case TrainingMode::ROLLOUT:
//       { auto pf = build_rollout_dataloader(dataset, cfg, ...); ... }
//   }
//
// A templated single-return-type dispatcher isn't provided
// because each mode legitimately returns a different Batch type
// — exactly the boundary described in the architecture: the
// data loader is mode-aware only at the collator/factory layer.
// ── Backward-compat: original SFT-only entry point ───────────
inline std::unique_ptr<CUDAPrefetcher> build_dataloader(
    const InstructionDataset& dataset_legacy,
    const DataLoaderConfig& cfg,
    int rank = 0, int world_size = 1, int64_t epoch = 0,
    torch::Device device = torch::kCPU)
{
    // Re-load via the unified path so the modern collator/sampler
    // stack is used even for legacy callers.
    DataLoaderConfig sft_cfg = cfg;
    sft_cfg.mode = TrainingMode::SFT;

    auto samples = UnifiedDatasetReader::load(sft_cfg);
    auto dataset = std::make_shared<UnifiedDataset>(std::move(samples), sft_cfg);

    (void)dataset_legacy; // legacy parameter kept for source compatibility
    return build_sft_dataloader(dataset, sft_cfg, rank, world_size, epoch, device);
}

}
// Pipeline overview
// [Python] tokenize_dataset.py --mode {sft|reward_model|dpo|rollout}
//      │
//      │ prompts.bin (+ responses.bin | chosen.bin/rejected.bin)
//      
// UnifiedDatasetReader::load(cfg)   (reads only what mode needs)
//      │  std::vector<RawSample>
//      
// UnifiedDataset(samples, cfg)      (mode-specific truncation + lengths)
//      │  dataset.lengths()
//      
// shard_indices(total, rank, world_size, ...)
//      │  shard (indices for this rank)
//      
// BucketBatchSampler(lengths, batch_size, buckets, shard)
//      │  batches() -> vector<vector<int64_t>>
//      
// UnifiedCollator::collate_*(items)  (mode-specific batch struct)
//      
// TypedPrefetcher<BatchT>(collate_fn, sampler, device)
//      │  .next() -> optional<BatchT>  (GPU)
//      
// Training loop
//   SFT / REWARD_MODEL / DPO -> complete batch, run loss directly
//   ROLLOUT (PPO/GRPO)  -> prompt batch only; rollout buffer
//                         handles everything generated after










