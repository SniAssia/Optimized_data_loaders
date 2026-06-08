
//  Reads instructions.bin + responses.bin (produced by tokenize_dataset.py)
//    Builds the full pipeline for rank 0 / world_size 1
//    Iterates one epoch, prints per-batch stats
//        Reports total tokens, batches, avg padding ratio
//
//  Build with:
//    cmake -B build -DCMAKE_PREFIX_PATH=/path/to/libtorch
//         cmake --build build -j$(nproc)
//
//  Run:
//    ./build/dataloader_test instructions.bin responses.bin

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <string>
#include "dataloader.h"

int main(int argc, char* argv[]) {
    
    std::string instructions_path = "instructions.bin";
    std::string responses_path    = "responses.bin";
    if (argc >= 2) instructions_path = argv[1];
    if (argc >= 3) responses_path    = argv[2];

    std::cout << "=== C++ Instruction DataLoader test ===\n\n";

    // ── Distributed init ──────────────────────────────────────────────
    auto ctx = dl::init_distributed();
    // device is already correct, no need to construct it manually
    std::cout << "rank=" << ctx.rank
              << "  world_size=" << ctx.world_size << "\n";

    // ── Device ────────────────────────────────────────────────────────
    torch::Device device = ctx.device;

    std::cout << "device: " << device << "\n\n";

    // ── Load binary dataset ───────────────────────────────────────────
    std::cout << "Loading: " << instructions_path
              << " + " << responses_path << " ...\n";
    // FIX: pass both paths
    auto samples = dl::BinaryDatasetReader::load(
        instructions_path, responses_path);
    std::cout << "  " << samples.size() << " samples loaded\n\n";

    // ── Config ────────────────────────────────────────────────────────
    dl::DataLoaderConfig cfg;
    cfg.max_seq_len     = 512;
    cfg.pad_token_id    = 50256; // GPT-2 EOS re-used as pad
    cfg.eos_token_id    = 50256;
    cfg.batch_size      = 8;
    cfg.num_workers     = 2;
    cfg.prefetch_factor = 2;    // FIX: field now exists in DataLoaderConfig
    cfg.shuffle         = true;
    cfg.seed            = 42;
    cfg.bucket_boundaries = {64, 128, 256, 512};

    // ── Dataset ───────────────────────────────────────────────────────
    dl::InstructionDataset dataset(std::move(samples), cfg);

    // ── DataLoader ────────────────────────────────────────────────────
    auto loader = dl::build_dataloader(dataset, cfg,
                                       ctx.rank, ctx.world_size,
                                       /*epoch=*/0, device);

    std::cout << "Batches (approx): " << loader->num_batches() << "\n";
    std::cout << "Bucket boundaries: [";
    for (std::size_t i = 0; i < cfg.bucket_boundaries.size(); ++i) {
        std::cout << cfg.bucket_boundaries[i];
        if (i + 1 < cfg.bucket_boundaries.size()) std::cout << ", ";
    }
    std::cout << "]\n\n";

    // ── Iterate one epoch ─────────────────────────────────────────────
    auto t0 = std::chrono::high_resolution_clock::now();

    int64_t     total_tokens   = 0;
    double      padding_waste  = 0.0;
    std::size_t batch_count    = 0;

    while (auto maybe_batch = loader->next()) {
        const auto& batch = *maybe_batch;
        int64_t B = batch.input_ids.size(0);
        int64_t L = batch.input_ids.size(1);

        total_tokens  += B * L;
        double pad_pct = 1.0 - batch.attention_mask.to(torch::kFloat32).mean().item<float>();
        padding_waste += pad_pct;

        if (batch_count < 5) {
            std::cout << "  batch " << batch_count
                      << " | shape [" << B << ", " << L << "]"
                      << " | device " << batch.input_ids.device()
                      << " | padding " << static_cast<int>(pad_pct * 100) << "%\n";
        }
        ++batch_count;
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();
    double avg_pad = batch_count ? padding_waste / static_cast<double>(batch_count) : 0.0;

    std::cout << "\n  " << batch_count  << " batches in "
              << elapsed << " s\n";
    std::cout << "  " << total_tokens  << " total token slots\n";
    std::cout << "  " << static_cast<int>(avg_pad * 100)
              << "% average padding ratio across all batches\n";
    std::cout << "\n✓ test passed.\n";

    dl::cleanup_distributed();
    return 0;
}
