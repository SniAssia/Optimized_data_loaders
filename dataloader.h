#pragma once

#include "dataset.h"
#include "sampler.h"
#include "collator.h"
#include "prefetcher.h"
#include "distributed.h"




namespace dl {

// ── build_dataloader ────────────────────────────────────────────────────────
//  Convenience factory that wires together all components.
//  Returns a CUDAPrefetcher ready to iterate.

inline std::unique_ptr<CUDAPrefetcher> build_dataloader(
    const InstructionDataset& dataset,
    const DataLoaderConfig&   cfg,
    int                       rank       = 0,
    int                       world_size = 1,
    int64_t                   epoch      = 0,
    torch::Device             device     = torch::kCPU)
{
    // 1. Shard indices for this rank
    auto shard = shard_indices(
        static_cast<int64_t>(dataset.size()),
        rank, world_size,
        cfg.shuffle, cfg.seed, epoch);

    // 2. Build sampler (heap-allocated so it outlives the lambda)
    auto sampler = std::make_shared<BucketBatchSampler>(
        dataset.lengths(),
        cfg.batch_size,
        cfg.bucket_boundaries,
        std::move(shard),
        cfg.shuffle,
        cfg.seed,
        epoch);

    // 3. Collator
    auto collator = std::make_shared<DynamicPaddingCollator>(cfg.pad_token_id);

    // 4. Prefetcher owns the sampler + collator
    return std::make_unique<CUDAPrefetcher>(
        dataset,
        *sampler,
        *collator,
        device,
        cfg.num_workers,
        static_cast<std::size_t>(cfg.prefetch_factor));  // FIX: use cfg.prefetch_factor
}

} // namespace dl



































//  Include this single header to access the full DataLoader stack:
//
//    #include "dataloader.h"
//
//  Pipeline overview
//  ─────────────────
//
//   [Python]  tokenize_dataset.py
//       │
//       │  instructions.bin + responses.bin
//       ▼
//   BinaryDatasetReader::load(instructions_path, responses_path)
//       │
//       │  std::vector<RawSample>
//       ▼
//   InstructionDataset(samples, cfg)
//       │
//       │  dataset.lengths()
//       ▼
//   shard_indices(total, rank, world_size, ...)
//       │
//       │  shard  (indices for this rank)
//       ▼
//   BucketBatchSampler(lengths, batch_size, buckets, shard)
//       │
//       │  build_batches()  →  vector<vector<int64_t>>
//       ▼
//   DynamicPaddingCollator(pad_token_id)
//       │
//       │  Batch { input_ids, attention_mask, labels }  (CPU)
//       ▼
//   CUDAPrefetcher(dataset, sampler, collator, device)
//       │
//       │  .next()  →  optional<Batch>               (GPU)
//       ▼
//   Training loop
//
