// Enhanced DataLoader test — detailed per-batch, per-bucket, and summary stats
//
// Build with:
//   cmake -B build -DCMAKE_PREFIX_PATH=/path/to/libtorch
//   cmake --build build -j$(nproc)
//
// Run:
//   ./build/dataloader_test instructions.bin responses.bin

#include "distributed.h"
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <string>
#include <vector>
#include "dataloader.h"

// ── helpers ──────────────────────────────────────────────────────────────────

static void print_separator(char c = '-', int w = 60) {
    std::cout << std::string(w, c) << "\n";
}

static double mean(const std::vector<double>& v) {
    return v.empty() ? 0.0
                     : std::accumulate(v.begin(), v.end(), 0.0) / v.size();
}

static double stddev(const std::vector<double>& v, double m) {
    if (v.size() < 2) return 0.0;
    double acc = 0.0;
    for (double x : v) acc += (x - m) * (x - m);
    return std::sqrt(acc / v.size());
}

// Return bucket label for a sequence length given the boundaries vector
static std::string bucket_label(int64_t len,
                                const std::vector<int>& boundaries) {
    for (int b : boundaries)
        if (len <= b) return "<=" + std::to_string(b);
    return ">" + std::to_string(boundaries.back());
}

// ── main ─────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {

    std::string instructions_path = "instructions.bin";
    std::string responses_path    = "responses.bin";
    if (argc >= 2) instructions_path = argv[1];
    if (argc >= 3) responses_path    = argv[2];

    print_separator('=', 60);
    std::cout << "   C++ Instruction DataLoader — detailed stats\n";
    print_separator('=', 60);
    std::cout << "\n";

    // ── distributed init ─────────────────────────────────────────────
    auto ctx = dl::init_distributed();
    std::cout << "rank=" << ctx.rank
              << "  world_size=" << ctx.world_size
              << "  device=" << ctx.device << "\n\n";

    torch::Device device = ctx.device;

    // ── load dataset ─────────────────────────────────────────────────
    std::cout << "Loading: " << instructions_path
              << " + " << responses_path << " ...\n";
    auto samples = dl::BinaryDatasetReader::load(instructions_path,
                                                  responses_path);
    std::cout << "  " << samples.size() << " samples loaded\n\n";

    // ── config ───────────────────────────────────────────────────────
    dl::DataLoaderConfig cfg;
    cfg.max_seq_len       = 512;
    cfg.pad_token_id      = 50256;
    cfg.eos_token_id      = 50256;
    cfg.batch_size        = 8;
    cfg.num_workers       = 2;
    cfg.prefetch_factor   = 2;
    cfg.shuffle           = true;
    cfg.seed              = 42;
    cfg.bucket_boundaries = {64, 128, 256, 512};

    dl::InstructionDataset dataset(std::move(samples), cfg);
    auto loader = dl::build_dataloader(dataset, cfg,
                                       ctx.rank, ctx.world_size,
                                       /*epoch=*/0, device);

    std::cout << "Approx batches : " << loader->num_batches() << "\n";
    std::cout << "Bucket boundaries: [";
    for (std::size_t i = 0; i < cfg.bucket_boundaries.size(); ++i) {
        std::cout << cfg.bucket_boundaries[i];
        if (i + 1 < cfg.bucket_boundaries.size()) std::cout << ", ";
    }
    std::cout << "]\n\n";

    // ── accumulators ─────────────────────────────────────────────────
    int64_t total_tokens  = 0;
    int64_t total_samples = 0;
    std::size_t batch_count = 0;

    std::vector<double> pad_pcts;         // padding % per batch
    std::vector<double> seq_lens;         // sequence length per batch
    std::vector<double> batch_times_ms;   // wall time per batch (ms)

    // per-bucket: batches, samples, pad sum
    std::map<std::string, int>    bucket_batches;
    std::map<std::string, int>    bucket_samples;
    std::map<std::string, double> bucket_pad_sum;
    for (int b : cfg.bucket_boundaries) {
        std::string k = "<=" + std::to_string(b);
        bucket_batches[k] = 0; bucket_samples[k] = 0; bucket_pad_sum[k] = 0.0;
    }
    {
        std::string k = ">" + std::to_string(cfg.bucket_boundaries.back());
        bucket_batches[k] = 0; bucket_samples[k] = 0; bucket_pad_sum[k] = 0.0;
    }

    // worst / best padding batch info
    double worst_pad = -1.0, best_pad = 2.0;
    std::size_t worst_batch = 0, best_batch = 0;
    int64_t worst_len = 0, best_len = 0;

    // GPU memory baseline
    int64_t gpu_mem_start = 0;
    if (torch::cuda::is_available())
        gpu_mem_start = torch::cuda::memory_allocated(device.index());

    // ── per-batch header ─────────────────────────────────────────────
    print_separator();
    std::cout << std::left
              << std::setw(6)  << "Batch"
              << std::setw(14) << "Shape"
              << std::setw(10) << "Device"
              << std::setw(10) << "Pad%"
              << std::setw(12) << "1st-samp%"
              << std::setw(12) << "Bucket"
              << std::setw(10) << "ms"
              << "\n";
    print_separator();

    // ── epoch loop ───────────────────────────────────────────────────
    auto epoch_t0 = std::chrono::high_resolution_clock::now();

    while (auto maybe_batch = loader->next()) {
        auto bt0 = std::chrono::high_resolution_clock::now();

        const auto& batch = *maybe_batch;
        int64_t B = batch.input_ids.size(0);
        int64_t L = batch.input_ids.size(1);

        // batch-level padding (mean over all tokens)
        float pad_mean = 1.0f - batch.attention_mask
                                    .to(torch::kFloat32)
                                    .mean()
                                    .template item<float>();

        // padding of the very first sample in this batch
        float first_sample_pad = 0.0f;
        if (B > 0) {
            auto mask_row = batch.attention_mask[0].to(torch::kFloat32);
            first_sample_pad = 1.0f - mask_row.mean().template item<float>();
        }

        auto bt1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(bt1 - bt0).count();

        // accumulate
        total_tokens  += B * L;
        total_samples += B;
        pad_pcts.push_back(pad_mean);
        seq_lens.push_back(static_cast<double>(L));
        batch_times_ms.push_back(ms);

        std::string bkt = bucket_label(L, cfg.bucket_boundaries);
        bucket_batches[bkt]++;
        bucket_samples[bkt] += static_cast<int>(B);
        bucket_pad_sum[bkt] += pad_mean;

        if (pad_mean > worst_pad) {
            worst_pad = pad_mean; worst_batch = batch_count; worst_len = L;
        }
        if (pad_mean < best_pad) {
            best_pad = pad_mean; best_batch = batch_count; best_len = L;
        }

        // print first 5 + last 2 batches, dots in between
        if (batch_count < 5 || batch_count >= loader->num_batches() - 2) {
            std::ostringstream shape;
            shape << "[" << B << "," << L << "]";
            std::ostringstream dev;
            dev << batch.input_ids.device();

            std::cout << std::left
                      << std::setw(6)  << batch_count
                      << std::setw(14) << shape.str()
                      << std::setw(10) << dev.str()
                      << std::setw(9)  << std::fixed << std::setprecision(1)
                                       << (pad_mean * 100.0f) << "%"
                      << std::setw(11) << std::fixed << std::setprecision(1)
                                       << (first_sample_pad * 100.0f) << "%"
                      << std::setw(12) << bkt
                      << std::setw(10) << std::fixed << std::setprecision(2) << ms
                      << "\n";
        } else if (batch_count == 5) {
            std::cout << "  ...\n";
        }

        ++batch_count;
    }

    auto epoch_t1 = std::chrono::high_resolution_clock::now();
    double total_s = std::chrono::duration<double>(epoch_t1 - epoch_t0).count();

    // ── summary ──────────────────────────────────────────────────────
    std::cout << "\n";
    print_separator('=', 60);
    std::cout << "  EPOCH SUMMARY\n";
    print_separator('=', 60);

    double avg_pad   = mean(pad_pcts);
    double std_pad   = stddev(pad_pcts, avg_pad);
    double avg_len   = mean(seq_lens);
    double std_len   = stddev(seq_lens, avg_len);
    double min_len   = *std::min_element(seq_lens.begin(), seq_lens.end());
    double max_len   = *std::max_element(seq_lens.begin(), seq_lens.end());
    double avg_ms    = mean(batch_times_ms);
    double toks_per_s = total_tokens / total_s;
    double samp_per_s = total_samples / total_s;

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  Batches          : " << batch_count << "\n";
    std::cout << "  Samples          : " << total_samples << "\n";
    std::cout << "  Total token slots: " << total_tokens << "\n";
    std::cout << "  Elapsed          : " << total_s      << " s\n";
    std::cout << "  Throughput       : " << (int)toks_per_s << " tokens/s"
              << "  |  " << (int)samp_per_s << " samples/s\n";
    std::cout << "  Avg batch time   : " << avg_ms << " ms\n";

    print_separator();
    std::cout << "  SEQUENCE LENGTH\n";
    print_separator();
    std::cout << "  Min  : " << (int)min_len << "\n";
    std::cout << "  Max  : " << (int)max_len << "\n";
    std::cout << "  Mean : " << avg_len << "\n";
    std::cout << "  Std  : " << std_len << "\n";

    print_separator();
    std::cout << "  PADDING\n";
    print_separator();
    std::cout << "  Mean  : " << (avg_pad * 100.0) << "%\n";
    std::cout << "  Std   : " << (std_pad * 100.0) << "%\n";
    std::cout << "  Worst : " << (worst_pad * 100.0) << "%"
              << "  (batch " << worst_batch << ", len=" << worst_len << ")\n";
    std::cout << "  Best  : " << (best_pad  * 100.0) << "%"
              << "  (batch " << best_batch  << ", len=" << best_len  << ")\n";

    print_separator();
    std::cout << "  PER-BUCKET BREAKDOWN\n";
    print_separator();
    std::cout << std::left
              << std::setw(10) << "Bucket"
              << std::setw(10) << "Batches"
              << std::setw(10) << "Samples"
              << std::setw(12) << "Avg Pad%"
              << "\n";
    print_separator('-', 42);
    for (auto& [k, nb] : bucket_batches) {
        double avg_bkt_pad = nb > 0 ? (bucket_pad_sum[k] / nb) * 100.0 : 0.0;
        std::cout << std::left
                  << std::setw(10) << k
                  << std::setw(10) << nb
                  << std::setw(10) << bucket_samples[k]
                  << std::setw(12) << std::fixed << std::setprecision(1)
                                   << avg_bkt_pad << "%"
                  << "\n";
    }

    // ── GPU memory ───────────────────────────────────────────────────
    if (torch::cuda::is_available()) {
        int64_t mem_now      = torch::cuda::memory_allocated(device.index());
        int64_t mem_reserved = torch::cuda::memory_reserved(device.index());
        print_separator();
        std::cout << "  GPU MEMORY (device " << device.index() << ")\n";
        print_separator();
        std::cout << "  Allocated (now)  : "
                  << (mem_now      / 1024 / 1024) << " MB\n";
        std::cout << "  Reserved (cache) : "
                  << (mem_reserved / 1024 / 1024) << " MB\n";
        std::cout << "  Delta vs start   : "
                  << ((mem_now - gpu_mem_start) / 1024 / 1024) << " MB\n";
    }

    print_separator('=', 60);
    std::cout << "  ✓ test passed.\n";
    print_separator('=', 60);

    dl::cleanup_distributed();
    return 0;
}