
#include "dataloader.h"
#include "distributed.h"

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

static void sep(char c = '-', int w = 64) {
    std::cout << std::string(w, c) << "\n";
}

static double mean_v(const std::vector<double>& v) {
    return v.empty() ? 0.0
        : std::accumulate(v.begin(), v.end(), 0.0) / v.size();
}

static double stddev_v(const std::vector<double>& v, double m) {
    if (v.size() < 2) return 0.0;
    double acc = 0.0;
    for (double x : v) acc += (x - m) * (x - m);
    return std::sqrt(acc / v.size());
}

int main(int argc, char* argv[]) {

    std::string manifest_path = "manifest.json";
    if (argc >= 2) manifest_path = argv[1];

    sep('=', 64);
    std::cout << "   Sharded Instruction DataLoader — test\n";
    sep('=', 64);
    std::cout << "\nmanifest: " << manifest_path << "\n\n";

    // ── distributed init ─────────────────────────────────────
    auto ctx = dl::init_distributed();
    std::cout << "rank=" << ctx.rank
              << "  world_size=" << ctx.world_size
              << "  device=" << ctx.device << "\n\n";

    torch::Device device = ctx.device;

    // ── config ───────────────────────────────────────────────
    dl::DataLoaderConfig cfg;
    cfg.batch_size      = 64;
    cfg.num_workers     = 2;
    cfg.prefetch_factor = 2;
    cfg.window_size     = 4;
    cfg.shuffle         = true;
    cfg.seed            = 42;

    // ── load dataset from manifest ───────────────────────────
    std::cout << "Loading manifest...\n";
    dl::InstructionDataset dataset(manifest_path, cfg);
    std::cout << "  total_samples = " << dataset.total_samples() << "\n";
    std::cout << "  num_shards    = " << dataset.num_shards()    << "\n";
    std::cout << "  max_seq_len   = " << dataset.config().max_seq_len  << "\n";
    std::cout << "  pad_token_id  = " << dataset.config().pad_token_id << "\n";
    std::cout << "  window_size   = " << dataset.config().window_size  << "\n\n";

    // ── build dataloader ─────────────────────────────────────
    auto loader = dl::build_dataloader(dataset, /*epoch=*/0, device);

    int approx_batches = dataset.total_samples() / cfg.batch_size;
    std::cout << "Batches (approx): " << approx_batches << "\n";
    std::cout << "Batch size       : " << cfg.batch_size << "\n\n";

    // ── per-batch stats ───────────────────────────────────────
    std::vector<double> pad_pcts;
    std::vector<double> seq_lens;
    std::vector<double> batch_times_ms;

    int64_t total_tokens  = 0;
    int64_t total_samples = 0;
    std::size_t batch_count = 0;

    double worst_pad = -1.0, best_pad = 2.0;
    std::size_t worst_batch = 0, best_batch = 0;
    int64_t worst_len = 0, best_len = 0;

    sep();
    std::cout << std::left
              << std::setw(6)  << "Batch"
              << std::setw(12) << "Shape"
              << std::setw(10) << "Device"
              << std::setw(10) << "Pad%"
              << std::setw(10) << "TrimLen"
              << std::setw(10) << "ms"
              << "\n";
    sep();

    auto epoch_t0 = std::chrono::high_resolution_clock::now();

    while (auto maybe = loader->next()) {
        auto bt0 = std::chrono::high_resolution_clock::now();

        const auto& batch = *maybe;
        if (batch.batch_max_len == 0) continue;  
        int64_t B = batch.input_ids.size(0);
        int64_t L = batch.input_ids.size(1);

        float pad_mean = 1.0f -
            batch.attention_mask.to(torch::kFloat32)
                                .mean().template item<float>();

        auto bt1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double,std::milli>(bt1-bt0).count();

        total_tokens  += B * L;
        total_samples += B;
        pad_pcts.push_back(pad_mean);
        seq_lens.push_back(static_cast<double>(L));
        batch_times_ms.push_back(ms);

        if (pad_mean > worst_pad) {
            worst_pad = pad_mean; worst_batch = batch_count; worst_len = L;
        }
        if (pad_mean < best_pad) {
            best_pad = pad_mean; best_batch = batch_count; best_len = L;
        }

        if (batch_count < 5) {
            std::ostringstream shape;
            shape << "[" << B << "," << L << "]";
            std::ostringstream dev;
            dev << batch.input_ids.device();

            std::cout << std::left
                      << std::setw(6)  << batch_count
                      << std::setw(12) << shape.str()
                      << std::setw(10) << dev.str()
                      << std::setw(9)  << std::fixed << std::setprecision(1)
                                       << (pad_mean * 100.0f) << "%"
                      << std::setw(10) << L
                      << std::setw(10) << std::fixed << std::setprecision(2)
                                       << ms
                      << "\n";
        } else if (batch_count == 5) {
            std::cout << "  ...\n";
        }

        ++batch_count;
    }

    auto epoch_t1 = std::chrono::high_resolution_clock::now();
    double total_s = std::chrono::duration<double>(epoch_t1-epoch_t0).count();

    // ── summary ──────────────────────────────────────────────
    std::cout << "\n";
    sep('=', 64);
    std::cout << "  EPOCH SUMMARY\n";
    sep('=', 64);

    double avg_pad  = mean_v(pad_pcts);
    double std_pad  = stddev_v(pad_pcts, avg_pad);
    double avg_len  = mean_v(seq_lens);
    double std_len  = stddev_v(seq_lens, avg_len);
    double avg_ms   = mean_v(batch_times_ms);
    double toks_s   = total_tokens  / total_s;
    double samps_s  = total_samples / total_s;

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  Batches         : " << batch_count    << "\n";
    std::cout << "  Samples         : " << total_samples  << "\n";
    std::cout << "  Token slots     : " << total_tokens   << "\n";
    std::cout << "  Elapsed         : " << total_s        << " s\n";
    std::cout << "  Throughput      : " << (int)toks_s    << " tokens/s"
              << "  |  "               << (int)samps_s   << " samples/s\n";
    std::cout << "  Avg batch time  : " << avg_ms         << " ms\n";

    sep();
    std::cout << "  SEQUENCE LENGTH (trimmed per batch)\n";
    sep();
    std::cout << "  Mean : " << avg_len << "\n";
    std::cout << "  Std  : " << std_len << "\n";
    std::cout << "  Min  : "
              << *std::min_element(seq_lens.begin(), seq_lens.end()) << "\n";
    std::cout << "  Max  : "
              << *std::max_element(seq_lens.begin(), seq_lens.end()) << "\n";

    sep();
    std::cout << "  PADDING (after per-batch trim)\n";
    sep();
    std::cout << "  Mean  : " << (avg_pad * 100.0) << "%\n";
    std::cout << "  Std   : " << (std_pad * 100.0) << "%\n";
    std::cout << "  Worst : " << (worst_pad * 100.0) << "%"
              << "  (batch " << worst_batch
              << ", trimmed_len=" << worst_len << ")\n";
    std::cout << "  Best  : " << (best_pad  * 100.0) << "%"
              << "  (batch " << best_batch
              << ", trimmed_len=" << best_len  << ")\n";

    sep('=', 64);
    std::cout << "  test passed.\n";
    sep('=', 64);

    dl::cleanup_distributed();
    return 0;
}