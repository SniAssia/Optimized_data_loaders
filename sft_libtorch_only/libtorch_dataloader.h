#pragma once
//  libtorch_dataloader.h — LibTorch DataLoader with shard cache
//  + full benchmark instrumentation
//
//  Metrics tracked
//  Per shard load:
//    disk_read_ms  — time to read samples.bin from disk
//    parse_ms     — time to parse binary format into RawSample
//    cache_hits      — shard already in RAM
//    cache_misses          — shard loaded from disk
//
//  Per batch:
//    collate_ms  — time to pad + build tensors on CPU
//    h2d_ms  — time to transfer batch to GPU
//    total_batch_ms   — collate + h2d
//    batch_max_len — trimmed sequence length
//    pad_pct  — padding percentage
//
//  Per epoch:
//    total_disk_ms    — total disk read time across all shard loads
//    total_parse_ms   — total binary parse time
//    total_collate_ms — total collation time
//    total_h2d_ms     — total H2D transfer time
//    total_epoch_ms   — wall time of full epoch
//    cache_hit_rate   — % of get() calls served from RAM
//    throughput       — samples/s, tokens/s
//
//  Storage sizes:
//    disk_bytes_per_shard   — size of samples.bin on disk
//    ram_bytes_per_shard    — size of loaded shard in RAM
//    batch_vram_bytes       — size of one batch in VRAM
// ============================================================

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <list>
#include <mutex>
#include <numeric>
#include <cstdlib>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <torch/torch.h>

#include "dataset.h"

namespace dl {
// BenchmarkStats — accumulated across one epoch
struct BenchmarkStats {
    // shard loading
    std::atomic<int64_t> cache_hits{0};
    std::atomic<int64_t> cache_misses{0};
    std::atomic<double>  total_disk_read_ms{0};
    std::atomic<double>  total_parse_ms{0};
    std::atomic<int64_t> total_disk_bytes{0};
    std::atomic<int64_t> total_ram_bytes{0};

    // batch construction
    std::atomic<double>  total_collate_ms{0};
    std::atomic<double>  total_h2d_ms{0};
    std::atomic<int64_t> total_batches{0};
    std::atomic<int64_t> total_samples{0};
    std::atomic<int64_t> total_real_tokens{0}; 
    std::atomic<int64_t> total_token_slots{0};
    std::atomic<double>  total_pad_pct{0};
    std::atomic<int64_t> total_vram_bytes{0};

    // epoch timing
    double epoch_wall_ms = 0.0;

    void reset() {
        cache_hits        = 0;
        cache_misses      = 0;
        total_disk_read_ms = 0;
        total_parse_ms    = 0;
        total_disk_bytes  = 0;
        total_ram_bytes   = 0;
        total_collate_ms  = 0;
        total_h2d_ms      = 0;
        total_batches     = 0;
        total_samples     = 0;
        total_real_tokens = 0;  
        total_token_slots = 0;
        total_pad_pct     = 0;
        total_vram_bytes  = 0;
        epoch_wall_ms     = 0.0;
    }

    // atomic double add helper
    static void atomic_add(std::atomic<double>& a, double v) {
        double old = a.load(std::memory_order_relaxed);
        while (!a.compare_exchange_weak(old, old + v,
               std::memory_order_relaxed)) {}
    }

    void add_disk_read(double disk_ms, double parse_ms,
                       int64_t disk_bytes, int64_t ram_bytes) {
        ++cache_misses;
        atomic_add(total_disk_read_ms, disk_ms);
        atomic_add(total_parse_ms,     parse_ms);
        total_disk_bytes += disk_bytes;
        total_ram_bytes  += ram_bytes;
    }

    void add_cache_hit() { ++cache_hits; }

    void add_batch(double collate_ms, double h2d_ms,
                   int64_t B, int64_t L, double pad_pct,
                   int64_t vram_bytes, int64_t real_tokens) {   
        atomic_add(total_collate_ms, collate_ms);
        atomic_add(total_h2d_ms,     h2d_ms);
        ++total_batches;
        total_samples     += B;
        total_token_slots += B * L;
        total_real_tokens += real_tokens;                        
        atomic_add(total_pad_pct, pad_pct);
        total_vram_bytes  += vram_bytes;
    }

    void print_report() const {
        double hit_rate = (cache_hits + cache_misses > 0)
            ? 100.0 * cache_hits / (cache_hits + cache_misses)
            : 0.0;
        double avg_pad = (total_batches > 0)
            ? total_pad_pct.load() / total_batches
            : 0.0;
        double toks_s = (epoch_wall_ms > 0)
            ? total_token_slots / (epoch_wall_ms / 1000.0)
            : 0.0;
        double samps_s = (epoch_wall_ms > 0)
            ? total_samples / (epoch_wall_ms / 1000.0)
            : 0.0;

        auto sep = [](char c = '-', int w = 66) {
            std::cout << std::string(w, c) << "\n";
        };

        std::cout << std::fixed << std::setprecision(2);

        sep('=', 66);
        std::cout << "  FULL BENCHMARK REPORT — LibTorch DataLoader\n";
        sep('=', 66);

        std::cout << "\n  ── SHARD CACHE ────────────────────────────────────\n";
        std::cout << "  Cache hits           : " << cache_hits   << "\n";
        std::cout << "  Cache misses         : " << cache_misses << "\n";
        std::cout << "  Cache hit rate       : " << hit_rate     << "%\n";

        std::cout << "\n  ── DISK I/O ───────────────────────────────────────\n";
        std::cout << "  Total disk read time : "
                  << total_disk_read_ms << " ms\n";
        std::cout << "  Total parse time     : "
                  << total_parse_ms     << " ms\n";
        std::cout << "  Total bytes read     : "
                  << total_disk_bytes / 1e6 << " MB\n";
        if (cache_misses > 0) {
            std::cout << "  Avg per shard load   : "
                      << total_disk_read_ms / cache_misses
                      << " ms  (disk)\n";
            std::cout << "  Avg per shard parse  : "
                      << total_parse_ms / cache_misses
                      << " ms  (parse)\n";
            std::cout << "  Avg shard size disk  : "
                      << total_disk_bytes / cache_misses / 1e6
                      << " MB\n";
            std::cout << "  Avg shard size RAM   : "
                      << total_ram_bytes / cache_misses / 1e6
                      << " MB\n";
        }

        std::cout << "\n  ── BATCH CONSTRUCTION ─────────────────────────────\n";
        std::cout << "  Total batches        : " << total_batches    << "\n";
        std::cout << "  Total samples        : " << total_samples    << "\n";
        std::cout << "  Total token slots    : " << total_token_slots<< "\n";
        std::cout << "  Total collate time   : "
                  << total_collate_ms << " ms\n";
        std::cout << "  Avg collate/batch    : "
                  << (total_batches > 0
                      ? total_collate_ms.load() / total_batches : 0.0)
                  << " ms\n";
        std::cout << "  Avg padding          : " << avg_pad << "%\n";
        // ── PROXY TRAINING TIME (no real training) ─────────────
        // Estimates one-epoch train time assuming step time scales with padded
        // tokens: proxy = alpha*padded + beta*batches. alpha=1,beta=0 => padded
        // tokens (comparable unit). Set PROXY_ALPHA (sec/token) to get seconds.
        {
            double alpha = 1.0, beta = 0.0;
            if (const char* a = std::getenv("PROXY_ALPHA")) alpha = std::atof(a);
            if (const char* b = std::getenv("PROXY_BETA"))  beta  = std::atof(b);
            int64_t padded = total_token_slots.load();
            int64_t real   = total_real_tokens.load();
            double  ppct   = padded > 0 ? 100.0*(padded-real)/padded : 0.0;
            double  proxy  = alpha*(double)padded + beta*(double)total_batches.load();
            std::cout << "\n  ── PROXY TRAINING TIME (no real training) ─────────\n";
            std::cout << "  Padded tokens        : " << padded << "\n";
            std::cout << "  Real (useful) tokens : " << real   << "\n";
            std::cout << "  Padding %            : " << ppct   << "%\n";
            std::cout << "  Proxy train time     : " << std::fixed << proxy
                      << "  (alpha=" << alpha << ", beta=" << beta << ")\n";
            std::cout << "  Proxy train time (h) : " << proxy/3600.0 << "\n";
        }

        std::cout << "\n  ── GPU TRANSFER (H2D) ─────────────────────────────\n";
        std::cout << "  Total H2D time       : "
                  << total_h2d_ms << " ms\n";
        std::cout << "  Avg H2D/batch        : "
                  << (total_batches > 0
                      ? total_h2d_ms.load() / total_batches : 0.0)
                  << " ms\n";
        std::cout << "  Total VRAM bytes     : "
                  << total_vram_bytes / 1e6 << " MB (cumulative)\n";
        if (total_batches > 0)
            std::cout << "  Avg VRAM/batch       : "
                      << total_vram_bytes / total_batches / 1e3
                      << " KB\n";

        std::cout << "\n  ── STORAGE SIZES ──────────────────────────────────\n";
        std::cout << "  Total disk usage     : "
                  << total_disk_bytes / 1e6 << " MB\n";
        std::cout << "  Peak RAM (cache)     : "
                  << total_ram_bytes / std::max((int64_t)1, cache_misses.load())
                     * 4   // cache capacity
                     / 1e6
                  << " MB  (est. 4-shard cache)\n";

        std::cout << "\n  ── EPOCH TIMING ───────────────────────────────────\n";
        std::cout << "  Epoch wall time      : "
                  << epoch_wall_ms << " ms  ("
                  << epoch_wall_ms / 1000.0 << " s)\n";
        double pipeline_ms = total_disk_read_ms.load()
                           + total_parse_ms.load()
                           + total_collate_ms.load()
                           + total_h2d_ms.load();
        std::cout << "  Pipeline total       : "
                  << pipeline_ms << " ms\n";
        std::cout << "  Overhead (sync/wait) : "
                  << epoch_wall_ms - pipeline_ms << " ms  ("
                  << (epoch_wall_ms > 0
                      ? 100.0*(epoch_wall_ms-pipeline_ms)/epoch_wall_ms
                      : 0.0) << "%)\n";

        std::cout << "\n  ── PHASE BREAKDOWN ────────────────────────────────\n";
        if (pipeline_ms > 0) {
            auto pct = [&](double v) {
                return 100.0 * v / pipeline_ms;
            };
            std::cout << "  Disk read   : "
                      << std::setw(8) << total_disk_read_ms
                      << " ms  (" << pct(total_disk_read_ms) << "%)\n";
            std::cout << "  Parse       : "
                      << std::setw(8) << total_parse_ms
                      << " ms  (" << pct(total_parse_ms) << "%)\n";
            std::cout << "  Collate     : "
                      << std::setw(8) << total_collate_ms
                      << " ms  (" << pct(total_collate_ms) << "%)\n";
            std::cout << "  H2D transfer: "
                      << std::setw(8) << total_h2d_ms
                      << " ms  (" << pct(total_h2d_ms) << "%)\n";
        }

        std::cout << "\n  ── THROUGHPUT ─────────────────────────────────────\n";
        std::cout << "  Samples/s    : " << (int)samps_s << "\n";
        std::cout << "  Tokens/s     : " << (int)toks_s  << "\n";
        sep('=', 66);
    }
};

// global stats object — shared between ShardCache, CollateFunction, main loop
static BenchmarkStats g_stats;

// ShardCache — LRU cache with benchmark instrumentation
class ShardCache {
public:
    explicit ShardCache(int capacity, const std::vector<ShardEntry>& shards)
        : capacity_(capacity), shards_(shards) {}

    const std::vector<RawSample>& get(int shard_idx) {
        std::lock_guard<std::mutex> lk(mu_);

        auto it = cache_.find(shard_idx);
        if (it != cache_.end()) {
            lru_.remove(shard_idx);
            lru_.push_front(shard_idx);
            g_stats.add_cache_hit();
            return it->second;
        }

        // evict LRU if at capacity
        if (static_cast<int>(cache_.size()) >= capacity_) {
            int evict = lru_.back();
            lru_.pop_back();
            cache_.erase(evict);
        }

        //measure disk read 
        const std::string path = shards_[shard_idx].dir + "/samples.bin";

        int64_t disk_bytes = 0;
        {
            std::ifstream f(path, std::ios::binary | std::ios::ate);
            if (f) disk_bytes = static_cast<int64_t>(f.tellg());
        }

        auto t_disk0 = std::chrono::high_resolution_clock::now();

        // read raw bytes into buffer (pure disk I/O)
        std::vector<char> buf(disk_bytes);
        {
            std::ifstream f(path, std::ios::binary);
            f.read(buf.data(), disk_bytes);
        }

        auto t_disk1 = std::chrono::high_resolution_clock::now();

        // ── measure parse ────────────────────────────────────
        auto t_parse0 = std::chrono::high_resolution_clock::now();

        cache_[shard_idx] = parse_from_buffer(buf);

        auto t_parse1 = std::chrono::high_resolution_clock::now();

        double disk_ms  = std::chrono::duration<double,std::milli>(
            t_disk1 - t_disk0).count();
        double parse_ms = std::chrono::duration<double,std::milli>(
            t_parse1 - t_parse0).count();

        // RAM size estimate: sum of all token ID vectors
        int64_t ram_bytes = 0;
        for (const auto& s : cache_[shard_idx])
            ram_bytes += static_cast<int64_t>(
                (s.prompt_ids.size() + s.response_ids.size()) * 4
                + sizeof(RawSample));

        g_stats.add_disk_read(disk_ms, parse_ms, disk_bytes, ram_bytes);

        std::cout << "[ShardCache] shard_" << shard_idx
                  << "  disk=" << std::fixed << std::setprecision(1)
                  << disk_ms  << "ms"
                  << "  parse=" << parse_ms << "ms"
                  << "  disk=" << disk_bytes/1e6 << "MB"
                  << "  ram="  << ram_bytes/1e6  << "MB"
                  << "  records=" << cache_[shard_idx].size()
                  << "\n";

        lru_.push_front(shard_idx);
        return cache_[shard_idx];
    }

    void clear() {
        std::lock_guard<std::mutex> lk(mu_);
        cache_.clear();
        lru_.clear();
    }

private:
    // parse binary buffer into RawSample vector
    static std::vector<RawSample> parse_from_buffer(
        const std::vector<char>& buf)
    {
        const char* ptr = buf.data();
        const char* end = buf.data() + buf.size();

        auto read_u32 = [&]() -> uint32_t {
            uint32_t v; std::memcpy(&v, ptr, 4); ptr += 4; return v;
        };
        auto read_u16 = [&]() -> uint16_t {
            uint16_t v; std::memcpy(&v, ptr, 2); ptr += 2; return v;
        };

        uint32_t n = read_u32();
        std::vector<RawSample> out;
        out.reserve(n);

        for (uint32_t i = 0; i < n && ptr < end; ++i) {
            RawSample s;
            uint16_t pl = read_u16();
            s.prompt_len = pl;
            s.prompt_ids.resize(pl);
            for (uint16_t j = 0; j < pl; ++j) {
                uint32_t tok = read_u32();
                s.prompt_ids[j] = static_cast<int32_t>(tok);
            }
            uint16_t rl = read_u16();
            s.response_len = rl;
            s.response_ids.resize(rl);
            for (uint16_t j = 0; j < rl; ++j) {
                uint32_t tok = read_u32();
                s.response_ids[j] = static_cast<int32_t>(tok);
            }
            out.push_back(std::move(s));
        }
        return out;
    }

    int                                             capacity_;
    const std::vector<ShardEntry>&                  shards_;
    std::mutex                                      mu_;
    std::unordered_map<int, std::vector<RawSample>> cache_;
    std::list<int>                                  lru_;
};

// ─────────────────────────────────────────────────────────────
// ShardAwareSampler
// ─────────────────────────────────────────────────────────────
class ShardAwareSampler
    : public torch::data::samplers::Sampler<> {
public:
    ShardAwareSampler(const std::vector<ShardEntry>& shards,
                      bool shuffle, int64_t seed)
        : shards_(shards), shuffle_(shuffle), seed_(seed), epoch_(0)
    {
        cumulative_.resize(shards_.size() + 1, 0);
        for (std::size_t i = 0; i < shards_.size(); ++i)
            cumulative_[i+1] = cumulative_[i] + shards_[i].n_samples;
        total_ = cumulative_.back();
        build();
    }

    void set_epoch(int64_t e) { epoch_ = e; build(); }

    void reset(torch::optional<std::size_t> = torch::nullopt) override {
        pos_ = 0;
    }

    torch::optional<std::vector<std::size_t>> next(
        std::size_t batch_size) override
    {
        if (pos_ >= index_.size()) return torch::nullopt;
        std::size_t end = std::min(pos_ + batch_size, index_.size());
        std::vector<std::size_t> b(
            index_.begin() + pos_,
            index_.begin() + end);
        pos_ = end;
        return b;
    }

    std::size_t size() const override { return total_; }

private:
    void build() {
        std::mt19937 rng(static_cast<uint64_t>(seed_ + epoch_));
        std::vector<int> order(shards_.size());
        std::iota(order.begin(), order.end(), 0);
        if (shuffle_) std::shuffle(order.begin(), order.end(), rng);

        index_.clear();
        index_.reserve(total_);
        for (int si : order) {
            int base = cumulative_[si];
            int n    = shards_[si].n_samples;
            std::vector<std::size_t> idx(n);
            std::iota(idx.begin(), idx.end(),
                      static_cast<std::size_t>(base));
            if (shuffle_) std::shuffle(idx.begin(), idx.end(), rng);
            for (auto v : idx) index_.push_back(v);
        }
        pos_ = 0;
    }

    const std::vector<ShardEntry>& shards_;
    bool    shuffle_;
    int64_t seed_, epoch_;
    std::vector<int>          cumulative_;
    std::size_t               total_ = 0;
    std::vector<std::size_t>  index_;
    std::size_t               pos_   = 0;
};

// ─────────────────────────────────────────────────────────────
// ShardedDataset
// ─────────────────────────────────────────────────────────────
class ShardedDataset
    : public torch::data::Dataset<ShardedDataset,
                                   InstructionDataset::Item>
{
public:
    explicit ShardedDataset(const std::string& manifest_path,
                             const DataLoaderConfig& cfg)
        : cfg_(cfg)
    {
        auto manifest    = manifest_parser::parse(manifest_path);
        cfg_.max_seq_len  = manifest.max_seq_len;
        cfg_.pad_token_id = manifest.pad_token_id;
        shards_           = manifest.shards;

        cumulative_.resize(shards_.size() + 1, 0);
        for (std::size_t i = 0; i < shards_.size(); ++i)
            cumulative_[i+1] = cumulative_[i] + shards_[i].n_samples;
        total_samples_ = cumulative_.back();

        cache_ = std::make_shared<ShardCache>(cfg_.num_workers, shards_);

        std::cout << "[ShardedDataset] "
                  << shards_.size() << " shards  "
                  << total_samples_ << " samples  "
                  << "cache=" << cfg_.num_workers << " slots\n";
    }

    InstructionDataset::Item get(std::size_t global_index) override {
        auto [si, ri] = locate(global_index);
        const auto& shard = cache_->get(si);
        return InstructionDataset::to_item(shard[ri]);
    }

    torch::optional<std::size_t> size() const override {
        return total_samples_;
    }

    const DataLoaderConfig&        config()        const { return cfg_; }
    int                            num_shards()    const {
        return static_cast<int>(shards_.size()); }
    int                            total_samples() const {
        return static_cast<int>(total_samples_); }
    const std::vector<ShardEntry>& shards()        const { return shards_; }
    void clear_cache() { cache_->clear(); }

private:
    std::pair<int,int> locate(std::size_t idx) const {
        auto it = std::upper_bound(cumulative_.begin(),
                                    cumulative_.end(),
                                    static_cast<int>(idx));
        int si = static_cast<int>(
            std::distance(cumulative_.begin(), it)) - 1;
        int ri = static_cast<int>(idx) - cumulative_[si];
        return {si, ri};
    }

    DataLoaderConfig            cfg_;
    std::vector<ShardEntry>     shards_;
    std::vector<int>            cumulative_;
    std::size_t                 total_samples_ = 0;
    std::shared_ptr<ShardCache> cache_;
};

// ─────────────────────────────────────────────────────────────
// CollateFunction — with collation timing
// ─────────────────────────────────────────────────────────────
struct CollateFunction {
    int64_t pad_id;
    int32_t max_seq_len;

    Batch operator()(
        std::vector<InstructionDataset::Item> items) const
    {
        const int64_t B = static_cast<int64_t>(items.size());
        if (B == 0)
            return {torch::zeros({0,0},torch::kInt64),
                    torch::zeros({0,0},torch::kInt64),
                    torch::zeros({0,0},torch::kInt64), 0};

        auto t0 = std::chrono::high_resolution_clock::now();

        int64_t batch_max_len = 0;
        for (const auto& it : items) {
            int64_t real = it.prompt_len + it.response_len;
            batch_max_len = std::max(batch_max_len, real);
        }
        batch_max_len = std::min(batch_max_len,
                                  static_cast<int64_t>(max_seq_len));

        auto input_ids      = torch::full(
            {B, batch_max_len}, pad_id, torch::kInt64);
        auto attention_mask = torch::zeros(
            {B, batch_max_len}, torch::kInt64);
        auto labels         = torch::full(
            {B, batch_max_len}, static_cast<int64_t>(-100), torch::kInt64);

        int64_t total_real = 0;
        for (int64_t i = 0; i < B; ++i) {
            const auto& it   = items[i];
            int64_t p_actual = std::min(it.prompt_len,   batch_max_len);
            int64_t r_actual = std::min(it.response_len,
                                         batch_max_len - p_actual);
            int64_t real     = p_actual + r_actual;
            total_real      += real;

            if (p_actual > 0)
                input_ids[i].slice(0,0,p_actual)
                    .copy_(it.prompt_ids.slice(0,0,p_actual));
            if (r_actual > 0)
                input_ids[i].slice(0,p_actual,p_actual+r_actual)
                    .copy_(it.response_ids.slice(0,0,r_actual));
            attention_mask[i].slice(0,0,real).fill_(1);
            if (r_actual > 0)
                labels[i].slice(0,p_actual,p_actual+r_actual)
                    .copy_(it.response_ids.slice(0,0,r_actual));
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        double collate_ms = std::chrono::duration<double,std::milli>(
            t1 - t0).count();

        double pad_pct = 100.0 * (1.0 -
            static_cast<double>(total_real) /
            static_cast<double>(B * batch_max_len));

        // record collate stats (H2D measured in main loop)
        // batch VRAM: 3 tensors × B × batch_max_len × 8 bytes
        int64_t vram_bytes = 3 * B * batch_max_len * 8;
        g_stats.add_batch(collate_ms, 0.0,
                          B, batch_max_len, pad_pct, vram_bytes,
                          total_real);                           
        return {input_ids, attention_mask, labels, batch_max_len};
    }
};

// ─────────────────────────────────────────────────────────────
// build_libtorch_dataloader
// ─────────────────────────────────────────────────────────────
auto build_libtorch_dataloader(
    ShardedDataset& dataset,
    int64_t         epoch  = 0,
    torch::Device   device = torch::kCPU)
{
    const auto& cfg = dataset.config();

    auto sampler = std::make_shared<ShardAwareSampler>(
        dataset.shards(), cfg.shuffle, cfg.seed);
    sampler->set_epoch(epoch);

    auto mapped = dataset.map(CollateFunction{
        cfg.pad_token_id, cfg.max_seq_len});

    auto options = torch::data::DataLoaderOptions()
        .batch_size(cfg.batch_size)
        .workers(cfg.num_workers)
        .enforce_ordering(true);

    return torch::data::make_data_loader(
        std::move(mapped), *sampler, options);
}

} // namespace dl
