// ============================================================
// main.cpp — Example usage of the unified post-training data loader
// ============================================================
//
// Demonstrates how to build a dataloader for each TrainingMode
// and shows where the rollout buffer fits into a PPO-style loop.
//
// Build with the provided CMakeLists.txt:
//   mkdir build && cd build
//   cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
//   cmake --build . -j
//
// Run:
//   ./dataloader_demo sft
//   ./dataloader_demo reward_model
//   ./dataloader_demo dpo
//   ./dataloader_demo rollout
// ============================================================

#include "dataset.h"
#include "sampler.h"
#include "collator.h"
#include "prefetcher.h"
#include "dataloader.h"
#include "rollout_buffer.h"
#include "distributed.h"

#include <iostream>
#include <string>

using namespace dl;

static void run_sft(const DataLoaderConfig& base_cfg, DistributedContext& dctx)
{
    DataLoaderConfig cfg = base_cfg;
    cfg.mode = TrainingMode::SFT;

    auto samples = UnifiedDatasetReader::load(cfg);
    auto dataset = std::make_shared<UnifiedDataset>(std::move(samples), cfg);

    auto prefetcher = build_sft_dataloader(
        dataset, cfg, dctx.rank, dctx.world_size, /*epoch=*/0, dctx.device);

    std::cout << "[SFT] " << prefetcher->num_batches() << " batches\n";

    int n = 0;
    while (auto batch = prefetcher->next()) {
        std::cout << "[SFT] batch " << n
                  << " input_ids=" << batch->input_ids.sizes()
                  << " labels=" << batch->labels.sizes() << "\n";
        if (++n >= 3) break; // just preview a few batches
    }
}

static void run_reward_model(const DataLoaderConfig& base_cfg, DistributedContext& dctx)
{
    DataLoaderConfig cfg = base_cfg;
    cfg.mode = TrainingMode::REWARD_MODEL;

    auto samples = UnifiedDatasetReader::load(cfg);
    auto dataset = std::make_shared<UnifiedDataset>(std::move(samples), cfg);

    auto prefetcher = build_reward_model_dataloader(
        dataset, cfg, dctx.rank, dctx.world_size, /*epoch=*/0, dctx.device);

    std::cout << "[REWARD_MODEL] " << prefetcher->num_batches() << " batches\n";

    int n = 0;
    while (auto batch = prefetcher->next()) {
        std::cout << "[REWARD_MODEL] batch " << n
                  << " chosen=" << batch->chosen_input_ids.sizes()
                  << " rejected=" << batch->rejected_input_ids.sizes() << "\n";
        if (++n >= 3) break;
    }
}

static void run_dpo(const DataLoaderConfig& base_cfg, DistributedContext& dctx)
{
    DataLoaderConfig cfg = base_cfg;
    cfg.mode = TrainingMode::DPO;

    auto samples = UnifiedDatasetReader::load(cfg);
    auto dataset = std::make_shared<UnifiedDataset>(std::move(samples), cfg);

    auto prefetcher = build_dpo_dataloader(
        dataset, cfg, dctx.rank, dctx.world_size, /*epoch=*/0, dctx.device);

    std::cout << "[DPO] " << prefetcher->num_batches() << " batches\n";

    int n = 0;
    while (auto batch = prefetcher->next()) {
        std::cout << "[DPO] batch " << n
                  << " chosen=" << batch->chosen_input_ids.sizes()
                  << " chosen_labels=" << batch->chosen_labels.sizes()
                  << " rejected=" << batch->rejected_input_ids.sizes() << "\n";
        if (++n >= 3) break;
    }
}

// PPO/GRPO rollout phase + a sketch of how the rollout buffer
// is used afterwards by the training loop.
static void run_rollout(const DataLoaderConfig& base_cfg, DistributedContext& dctx)
{
    DataLoaderConfig cfg = base_cfg;
    cfg.mode = TrainingMode::ROLLOUT;

    auto samples = UnifiedDatasetReader::load(cfg);
    auto dataset = std::make_shared<UnifiedDataset>(std::move(samples), cfg);

    auto prefetcher = build_rollout_dataloader(
        dataset, cfg, dctx.rank, dctx.world_size, /*epoch=*/0, dctx.device);

    std::cout << "[ROLLOUT] " << prefetcher->num_batches()
              << " prompt batches (group_size=" << cfg.grpo_group_size << ")\n";

    RolloutBuffer buffer(/*minibatch_size=*/cfg.batch_size, /*normalize_adv=*/true);

    int n = 0;
    while (auto batch = prefetcher->next()) {
        std::cout << "[ROLLOUT] prompt batch " << n
                  << " prompt_ids=" << batch->prompt_ids.sizes() << "\n";

        // ---- Everything below is TRAINING-LOOP responsibility ----
        // The data loader's job ends at batch->prompt_ids.
        // In a real PPO/GRPO loop you would now:
        //   1. run the actor to generate responses into the
        //      reserved generation-space columns
        //   2. run the reference model for KL
        //   3. score with the reward model (or rule-based fn)
        //   4. compute advantages (GAE for PPO, group stats for GRPO)
        //   5. push everything into the RolloutBuffer

        const int64_t B = batch->prompt_ids.size(0);
        const int64_t gen_len = cfg.max_gen_len;

        RolloutEntry entry;
        entry.prompt_ids    = batch->prompt_ids;
        entry.response_ids  = torch::zeros({B, gen_len}, torch::kInt64);   // placeholder
        entry.old_log_probs = torch::zeros({B, gen_len});                  // placeholder
        entry.ref_log_probs = torch::zeros({B, gen_len});                  // placeholder
        entry.rewards       = torch::zeros({B});                           // placeholder
        entry.advantages    = torch::zeros({B, gen_len});                  // placeholder

        buffer.push(std::move(entry));

        if (++n >= 2) break;
    }

    buffer.seal();
    std::cout << "[ROLLOUT] buffer sealed: " << buffer.total_samples()
              << " samples, " << buffer.num_minibatches() << " minibatches\n";

    // PPO update epochs would iterate here:
    // for (int epoch = 0; epoch < K; ++epoch) {
    //     buffer.reset_cursor();
    //     while (auto mb = buffer.next_minibatch()) { ... gradient step ... }
    // }
    buffer.clear();
}

int main(int argc, char** argv)
{
    std::string mode = argc > 1 ? argv[1] : "sft";

    auto dctx = init_distributed();

    DataLoaderConfig cfg;
    cfg.batch_size      = 4;
    cfg.num_workers     = 2;
    cfg.prefetch_factor = 2;
    cfg.shuffle         = true;
    cfg.max_seq_len     = 512;
    cfg.max_prompt_len  = 128;
    cfg.max_gen_len     = 256;
    cfg.grpo_group_size = 8;

    // Default file names — adjust to wherever tokenize_dataset.py wrote them.
    cfg.prompt_path   = "prompts.bin";
    cfg.response_path = "responses.bin";
    cfg.chosen_path   = "chosen.bin";
    cfg.rejected_path = "rejected.bin";

    try {
        if (mode == "sft")              run_sft(cfg, dctx);
        else if (mode == "reward_model") run_reward_model(cfg, dctx);
        else if (mode == "dpo")          run_dpo(cfg, dctx);
        else if (mode == "rollout")      run_rollout(cfg, dctx);
        else {
            std::cerr << "Unknown mode: " << mode
                      << " (expected sft|reward_model|dpo|rollout)\n";
            return 1;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        std::cerr << "Did you run tokenize_dataset.py to produce the .bin files?\n";
        return 1;
    }

    cleanup_distributed();
    return 0;
}
