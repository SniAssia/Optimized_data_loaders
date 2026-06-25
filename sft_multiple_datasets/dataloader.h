#pragma once
//  dataloader.h — Convenience factory
//
//  Pipeline
//  ────────
//  InstructionDataset  (manifest only, no data in RAM)
//       │
//       ▼
//  CUDAPrefetcher
//    producer thread:
//      per window of W shards:
//        load W shards in parallel (std::async)
//        shuffle each shard
//        N reader threads → shared bounded queue
//        consumer accumulates batch_size records
//        TrimPaddingCollator → Batch
//        .to(device, non_blocking)
//        push to ready_queue
//       │
//       ▼
//  training loop: loader->next() → optional<Batch>

#include "collator.h"
#include "dataset.h"
#include "distributed.h"
#include "prefetcher.h"

namespace dl {

inline std::unique_ptr<CUDAPrefetcher> build_dataloader(
    const InstructionDataset& dataset,
    int64_t                   epoch  = 0,
    torch::Device             device = torch::kCPU)
{
    const auto& cfg = dataset.config();

    auto collator = std::make_shared<TrimPaddingCollator>(
        cfg.pad_token_id, cfg.max_seq_len);

    return std::make_unique<CUDAPrefetcher>(
        dataset,
        *collator,
        device,
        epoch,
        static_cast<std::size_t>(cfg.prefetch_factor));
}

} // namespace dl