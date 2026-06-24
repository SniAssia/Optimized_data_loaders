#pragma once
// ============================================================
//  dataloader.h — Convenience factory
//
//  Pipeline overview
//  ─────────────────
//
//  [Python] tokenize_dataset.py
//      │
//      │  shard_XX/samples.bin + lengths.bin  +  manifest.json
//      ▼
//  InstructionDataset(manifest_path, cfg)
//      │  reads all shards sequentially into RAM
//      │  std::vector<RawSample>  (padded to max_seq_len)
//      ▼
//  shard_indices(total, rank, world_size, shuffle, seed, epoch)
//      │  std::vector<int64_t>  (this rank's indices, shuffled)
//      ▼
//  SequentialBatchSampler(indices, batch_size)
//      │  vector<vector<int64_t>>  (batches)
//      ▼
//  TrimPaddingCollator(pad_token_id, max_seq_len)
//      │  trims each batch to its max real token length
//      │  Batch { input_ids, attention_mask, labels, batch_max_len }
//      ▼
//  CUDAPrefetcher(dataset, sampler, collator, device, prefetch_depth)
//      │  producer thread prepares next batch while GPU trains on current
//      │  .next() → optional<Batch>   (on device)
//      ▼
//  Training loop
// ============================================================

#include "collator.h"
#include "dataset.h"
#include "distributed.h"
#include "prefetcher.h"
#include "sampler.h"

namespace dl {

inline std::unique_ptr<CUDAPrefetcher> build_dataloader(
    const InstructionDataset& dataset,
    const DataLoaderConfig&   cfg,
    int                       rank       = 0,
    int                       world_size = 1,
    int64_t                   epoch      = 0,
    torch::Device             device     = torch::kCPU)
{
    // 1. Shard + shuffle indices for this rank
    auto indices = shard_indices(
        static_cast<int64_t>(dataset.size()),
        rank, world_size,
        cfg.shuffle, cfg.seed, epoch);

    // 2. Sequential batch sampler
    auto sampler = std::make_shared<SequentialBatchSampler>(
        std::move(indices), cfg.batch_size);

    // 3. Trim-padding collator
    auto collator = std::make_shared<TrimPaddingCollator>(
        cfg.pad_token_id, cfg.max_seq_len);

    // 4. Prefetcher (owns producer thread)
    return std::make_unique<CUDAPrefetcher>(
        dataset,
        *sampler,
        *collator,
        device,
        static_cast<std::size_t>(cfg.prefetch_factor));
}

} // namespace dl
