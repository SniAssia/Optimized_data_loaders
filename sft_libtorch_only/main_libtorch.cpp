// ============================================================
//  main_libtorch.cpp — LibTorch DataLoader full benchmark
// ============================================================

#include "libtorch_dataloader.h"
#include "distributed.h"

#include <chrono>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

static void sep(char c = '-', int w = 66) {
    std::cout << std::string(w, c) << "\n";
}

int main(int argc, char* argv[]) {

    std::string manifest_path = "manifest.json";
    if (argc >= 2) manifest_path = argv[1];

    int batch_size_arg = 0;
    if (argc >= 3) batch_size_arg = std::stoi(argv[2]);

    sep('=', 66);
    std::cout << "   LibTorch Sharded DataLoader — Full Benchmark\n";
    sep('=', 66);
    std::cout << "\nmanifest: " << manifest_path << "\n\n";

    // ── distributed init ─────────────────────────────────────
    auto ctx = dl::init_distributed();
    std::cout << "rank=" << ctx.rank
              << "  world_size=" << ctx.world_size
              << "  device=" << ctx.device << "\n\n";
    torch::Device device = ctx.device;

    // ── config ───────────────────────────────────────────────
    dl::DataLoaderConfig cfg;
    cfg.batch_size      = batch_size_arg > 0 ? batch_size_arg : 8;
    cfg.num_workers     = 4;
    cfg.prefetch_factor = 2;
    cfg.shuffle         = true;
    cfg.seed            = 42;

    std::cout << "Config:\n";
    std::cout << "  batch_size  = " << cfg.batch_size  << "\n";
    std::cout << "  num_workers = " << cfg.num_workers
              << "  (= shard cache slots)\n";
    std::cout << "  shuffle     = " << cfg.shuffle     << "\n\n";

    // ── reset global benchmark stats ─────────────────────────
    dl::g_stats.reset();

    // ── build dataset ─────────────────────────────────────────
    dl::ShardedDataset dataset(manifest_path, cfg);
    std::cout << "  total_samples = " << dataset.total_samples() << "\n";
    std::cout << "  num_shards    = " << dataset.num_shards()    << "\n";
    std::cout << "  max_seq_len   = " << dataset.config().max_seq_len  << "\n";
    std::cout << "  pad_token_id  = " << dataset.config().pad_token_id << "\n\n";

    // ── disk storage report ───────────────────────────────────
    {
        int64_t total_disk = 0;
        for (const auto& s : dataset.shards()) {
            std::string p = s.dir + "/samples.bin";
            std::ifstream f(p, std::ios::binary | std::ios::ate);
            if (f) total_disk += static_cast<int64_t>(f.tellg());
        }
        std::cout << "Storage:\n";
        std::cout << "  Total disk usage : "
                  << std::fixed << std::setprecision(2)
                  << total_disk / 1e6 << " MB  ("
                  << total_disk / 1e9 << " GB)\n";
        std::cout << "  Avg per shard    : "
                  << total_disk / dataset.num_shards() / 1e6
                  << " MB\n\n";
    }

    // ── build loader ─────────────────────────────────────────
    auto loader = dl::build_libtorch_dataloader(
        dataset, /*epoch=*/0, device);

    int approx = dataset.total_samples() / cfg.batch_size;
    std::cout << "Batches (approx): " << approx << "\n";
    std::cout << "Batch size       : " << cfg.batch_size << "\n\n";

    // ── per-batch tracking ────────────────────────────────────
    std::vector<double> pad_pcts;
    std::vector<double> seq_lens;
    std::vector<double> h2d_times;
    pad_pcts.reserve(approx);
    seq_lens.reserve(approx);
    h2d_times.reserve(approx);

    std::size_t batch_count   = 0;
    int64_t     total_tokens  = 0;
    int64_t     total_samples = 0;

    double worst_pad = -1.0, best_pad = 2.0;
    std::size_t worst_batch = 0, best_batch = 0;
    int64_t worst_len = 0, best_len = 0;

    sep();
    std::cout << std::left
              << std::setw(7)  << "Batch"
              << std::setw(12) << "Shape"
              << std::setw(10) << "Device"
              << std::setw(9)  << "Pad%"
              << std::setw(9)  << "TrimLen"
              << std::setw(9)  << "H2D ms"
              << "\n";
    sep();

    // ── epoch loop ────────────────────────────────────────────
    auto epoch_t0 = std::chrono::high_resolution_clock::now();

    for (auto& batch : *loader) {
        if (batch.batch_max_len == 0) continue;

        int64_t B = batch.input_ids.size(0);
        int64_t L = batch.input_ids.size(1);

        // ── H2D transfer + timing ─────────────────────────────
        auto t_h2d0 = std::chrono::high_resolution_clock::now();
        auto input_ids      = batch.input_ids.to(
            device, /*non_blocking=*/true);
        auto attention_mask = batch.attention_mask.to(
            device, /*non_blocking=*/true);
        auto labels         = batch.labels.to(
            device, /*non_blocking=*/true);
        // synchronize to get accurate H2D time
        if (device.type() == torch::kCUDA)
            c10::cuda::device_synchronize();
        auto t_h2d1 = std::chrono::high_resolution_clock::now();
        double h2d_ms = std::chrono::duration<double,std::milli>(
            t_h2d1 - t_h2d0).count();

        // record H2D time in stats
        BenchmarkStats::atomic_add(dl::g_stats.total_h2d_ms, h2d_ms);

        float pad_mean = 1.0f -
            attention_mask.to(torch::kFloat32)
                          .mean().template item<float>();

        total_tokens  += B * L;
        total_samples += B;
        pad_pcts.push_back(static_cast<double>(pad_mean));
        seq_lens.push_back(static_cast<double>(L));
        h2d_times.push_back(h2d_ms);

        if (static_cast<double>(pad_mean) > worst_pad) {
            worst_pad = pad_mean; worst_batch = batch_count; worst_len = L;
        }
        if (static_cast<double>(pad_mean) < best_pad) {
            best_pad = pad_mean; best_batch = batch_count; best_len = L;
        }

        if (batch_count < 5) {
            std::ostringstream shape;
            shape << "[" << B << "," << L << "]";
            std::ostringstream dev;
            dev << input_ids.device();
            std::cout << std::left
                      << std::setw(7)  << batch_count
                      << std::setw(12) << shape.str()
                      << std::setw(10) << dev.str()
                      << std::setw(8)  << std::fixed
                                       << std::setprecision(1)
                                       << (pad_mean*100.0f) << "%"
                      << std::setw(9)  << L
                      << std::setw(9)  << std::setprecision(2)
                                       << h2d_ms
                      << "\n";
        } else if (batch_count == 5) {
            std::cout << "  ...\n";
        }

        ++batch_count;
    }

    auto epoch_t1 = std::chrono::high_resolution_clock::now();
    dl::g_stats.epoch_wall_ms =
        std::chrono::duration<double,std::milli>(
            epoch_t1 - epoch_t0).count();

    // ── per-batch summary ─────────────────────────────────────
    auto mean_v = [](const std::vector<double>& v) {
        return v.empty() ? 0.0
            : std::accumulate(v.begin(),v.end(),0.0) / v.size();
    };
    auto stddev_v = [&](const std::vector<double>& v, double m) {
        if (v.size() < 2) return 0.0;
        double acc = 0.0;
        for (double x : v) acc += (x-m)*(x-m);
        return std::sqrt(acc/v.size());
    };

    double avg_pad = mean_v(pad_pcts);
    double avg_len = mean_v(seq_lens);
    double avg_h2d = mean_v(h2d_times);

    std::cout << "\n";
    sep('=', 66);
    std::cout << "  BATCH STATISTICS\n";
    sep('=', 66);
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  Total batches    : " << batch_count   << "\n";
    std::cout << "  Total samples    : " << total_samples << "\n";
    std::cout << "  Total tok slots  : " << total_tokens  << "\n";
    sep();
    std::cout << "  SEQUENCE LENGTH (trimmed)\n";
    sep();
    std::cout << "  Mean : " << avg_len << "\n";
    std::cout << "  Std  : " << stddev_v(seq_lens, avg_len) << "\n";
    std::cout << "  Min  : "
              << *std::min_element(seq_lens.begin(),seq_lens.end()) << "\n";
    std::cout << "  Max  : "
              << *std::max_element(seq_lens.begin(),seq_lens.end()) << "\n";
    sep();
    std::cout << "  PADDING\n";
    sep();
    std::cout << "  Mean  : " << avg_pad*100.0 << "%\n";
    std::cout << "  Std   : "
              << stddev_v(pad_pcts,avg_pad)*100.0 << "%\n";
    std::cout << "  Worst : " << worst_pad*100.0 << "%"
              << "  (batch " << worst_batch
              << " len=" << worst_len << ")\n";
    std::cout << "  Best  : " << best_pad*100.0  << "%"
              << "  (batch " << best_batch
              << " len=" << best_len  << ")\n";
    sep();
    std::cout << "  H2D TRANSFER\n";
    sep();
    std::cout << "  Mean/batch : " << avg_h2d << " ms\n";
    std::cout << "  Min        : "
              << *std::min_element(h2d_times.begin(),h2d_times.end())
              << " ms\n";
    std::cout << "  Max        : "
              << *std::max_element(h2d_times.begin(),h2d_times.end())
              << " ms\n";

    // ── full benchmark report ─────────────────────────────────
    dl::g_stats.print_report();

    dl::cleanup_distributed();
    return 0;
}
