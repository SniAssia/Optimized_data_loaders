#pragma once
// ============================================================
//  dataset.h — Sharded Instruction/Response Binary Dataset (SFT)
//
//  Reads shards produced by tokenize_dataset.py.
//
//  Each shard contains samples.bin and lengths.bin.
//
//  Binary sample format (samples.bin per shard):
//    uint32  n_samples
//    per sample:
//      uint16  prompt_len          real token count
//      uint16  response_len        real token count
//      uint32  prompt_ids   [max_seq_len]   padded
//      uint32  response_ids [max_seq_len]   padded
//      uint8   attention_mask[max_seq_len]  1=real, 0=pad
//
//  Binary lengths format (lengths.bin per shard):
//    uint32  n_samples
//    per sample:
//      uint16  prompt_len
//      uint16  response_len
// ============================================================

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <numeric>

#include <torch/torch.h>

namespace dl {
struct RawSample {
    std::vector<int32_t> prompt_ids;    // exact real tokens
    std::vector<int32_t> response_ids;  // exact real tokens
    // prompt_ids.size() == prompt_len, no padding stored
};
struct ShardEntry {
    int         shard_idx  = 0;
    int         n_samples  = 0;
    std::string dir;
};

// ─────────────────────────────────────────────────────────────
// Config
// ─────────────────────────────────────────────────────────────
struct DataLoaderConfig {
    int32_t max_seq_len   = 512;
    int32_t pad_token_id  = 0;
    int32_t eos_token_id  = 2;
    int32_t batch_size    = 8;
    int32_t num_workers   = 4;
    int32_t prefetch_factor = 2;
    bool    shuffle       = true;
    int64_t seed          = 42;
};

// ─────────────────────────────────────────────────────────────
// Low-level binary reader helpers
// ─────────────────────────────────────────────────────────────
namespace detail {

template<typename T>
static void read_pod(std::ifstream& f, T& out) {
    f.read(reinterpret_cast<char*>(&out), sizeof(T));
    if (!f) throw std::runtime_error("Unexpected EOF in binary file");
}

// Read one shard's samples.bin into a vector<RawSample>
static std::vector<RawSample> read_samples_bin(
    const std::string& path, int32_t max_seq_len)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open: " + path);

    uint32_t n_samples;
    read_pod(f, n_samples);

    std::vector<RawSample> out;
    out.reserve(n_samples);

    for (uint32_t i = 0; i < n_samples; ++i) {
        RawSample s;
        read_pod(f, s.prompt_len);
        read_pod(f, s.response_len);

        uint16_t prompt_len;
        read_pod(f, prompt_len);
        s.prompt_ids.resize(prompt_len);
        for (int j = 0; j < prompt_len; ++j) {
            uint32_t tok; read_pod(f, tok);
            s.prompt_ids[j] = static_cast<int32_t>(tok);
        }
        uint16_t response_len;
        read_pod(f, response_len);
        s.response_ids.resize(response_len);
        for (int j = 0; j < response_len; ++j) {
            uint32_t tok; read_pod(f, tok);
            s.response_ids[j] = static_cast<int32_t>(tok);
        }
        s.prompt_len   = prompt_len;
        s.response_len = response_len;

        s.response_ids.resize(max_seq_len);
        for (int j = 0; j < max_seq_len; ++j) {
            uint32_t tok; read_pod(f, tok);
            s.response_ids[j] = static_cast<int32_t>(tok);
        }

        s.attention_mask.resize(max_seq_len);
        for (int j = 0; j < max_seq_len; ++j) {
            uint8_t m; read_pod(f, m);
            s.attention_mask[j] = m;
        }

        out.push_back(std::move(s));
    }
    return out;
}

// Read one shard's lengths.bin — returns (prompt_len, response_len) pairs
static std::vector<std::pair<uint16_t,uint16_t>>
read_lengths_bin(const std::string& path)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open: " + path);

    uint32_t n; read_pod(f, n);
    std::vector<std::pair<uint16_t,uint16_t>> out;
    out.reserve(n);

    for (uint32_t i = 0; i < n; ++i) {
        uint16_t pl, rl;
        read_pod(f, pl);
        read_pod(f, rl);
        out.emplace_back(pl, rl);
    }
    return out;
}

} // namespace detail

// ─────────────────────────────────────────────────────────────
// Minimal JSON manifest parser
// ─────────────────────────────────────────────────────────────
namespace manifest_parser {

inline std::string extract_string(const std::string& json,
                                   const std::string& key) {
    std::string search = "\"" + key + "\"";
    auto pos = json.find(search);
    if (pos == std::string::npos) return "";
    pos = json.find('"', pos + search.size() + 1);
    if (pos == std::string::npos) return "";
    auto end = json.find('"', pos + 1);
    return json.substr(pos + 1, end - pos - 1);
}

inline int extract_int(const std::string& json, const std::string& key) {
    std::string search = "\"" + key + "\"";
    auto pos = json.find(search);
    if (pos == std::string::npos) return 0;
    pos = json.find_first_of("-0123456789", pos + search.size());
    if (pos == std::string::npos) return 0;
    return std::stoi(json.substr(pos));
}

struct Manifest {
    int32_t              max_seq_len  = 512;
    int32_t              pad_token_id = 0;
    std::vector<ShardEntry> shards;
};

inline Manifest parse(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("Cannot open manifest: " + path);
    std::string json((std::istreambuf_iterator<char>(f)),
                      std::istreambuf_iterator<char>());

    Manifest m;
    m.max_seq_len  = extract_int(json, "max_seq_len");
    m.pad_token_id = extract_int(json, "pad_token_id");

    auto shards_pos = json.find("\"shards\"");
    if (shards_pos == std::string::npos)
        throw std::runtime_error("manifest.json: missing 'shards' key");

    auto arr_open  = json.find('[', shards_pos);
    auto arr_close = json.rfind(']');
    std::string arr = json.substr(arr_open + 1, arr_close - arr_open - 1);

    std::size_t cursor = 0;
    while (true) {
        auto obj_open  = arr.find('{', cursor);
        if (obj_open == std::string::npos) break;
        auto obj_close = arr.find('}', obj_open);
        std::string obj = arr.substr(obj_open, obj_close - obj_open + 1);

        ShardEntry e;
        e.shard_idx = extract_int(obj, "shard");
        e.n_samples = extract_int(obj, "n_samples");
        e.dir       = extract_string(obj, "dir");
        m.shards.push_back(e);
        cursor = obj_close + 1;
    }

    std::sort(m.shards.begin(), m.shards.end(),
              [](const ShardEntry& a, const ShardEntry& b){
                  return a.shard_idx < b.shard_idx;
              });
    return m;
}

} // namespace manifest_parser

// ─────────────────────────────────────────────────────────────
// InstructionDataset
// Loads all shards sequentially and exposes samples.
// ─────────────────────────────────────────────────────────────
class InstructionDataset {
public:
    // Construct from a manifest.json path
    explicit InstructionDataset(const std::string& manifest_path,
                                 const DataLoaderConfig& cfg)
        : cfg_(cfg)
    {
        auto manifest = manifest_parser::parse(manifest_path);

        // override max_seq_len from manifest so it matches what Python wrote
        cfg_.max_seq_len  = manifest.max_seq_len;
        cfg_.pad_token_id = manifest.pad_token_id;

        // load all shards sequentially
        for (const auto& shard : manifest.shards) {
            std::string samples_path = shard.dir + "/samples.bin";
            auto shard_samples = detail::read_samples_bin(
                samples_path, cfg_.max_seq_len);

            for (auto& s : shard_samples)
                samples_.push_back(std::move(s));
        }

        // build lengths_ (total real token count per sample, capped at max_seq_len)
        lengths_.reserve(samples_.size());
        for (const auto& s : samples_)
            lengths_.push_back(
                std::min(s.total_real_len(), cfg_.max_seq_len));
    }

    // Construct directly from a vector<RawSample> (legacy / testing)
    InstructionDataset(std::vector<RawSample> samples,
                        const DataLoaderConfig& cfg)
        : cfg_(cfg), samples_(std::move(samples))
    {
        lengths_.reserve(samples_.size());
        for (const auto& s : samples_)
            lengths_.push_back(
                std::min(s.total_real_len(), cfg_.max_seq_len));
    }

    std::size_t size() const { return samples_.size(); }
    const std::vector<int32_t>& lengths() const { return lengths_; }
    const DataLoaderConfig& config() const { return cfg_; }

    // ── Item returned per sample ─────────────────────────────
    struct Item {
        torch::Tensor prompt_ids;      // [max_seq_len]
        torch::Tensor response_ids;    // [max_seq_len]
        torch::Tensor attention_mask;  // [max_seq_len]
        int64_t       prompt_len;
        int64_t       response_len;
    };

    Item get(int64_t idx) const {
        const auto& s = samples_[static_cast<std::size_t>(idx)];

        auto t_prompt = torch::from_blob(
            const_cast<int32_t*>(s.prompt_ids.data()),
            {cfg_.max_seq_len}, torch::kInt32
        ).to(torch::kInt64).clone();

        auto t_response = torch::from_blob(
            const_cast<int32_t*>(s.response_ids.data()),
            {cfg_.max_seq_len}, torch::kInt32
        ).to(torch::kInt64).clone();

        auto t_mask = torch::from_blob(
            const_cast<uint8_t*>(s.attention_mask.data()),
            {cfg_.max_seq_len}, torch::kByte
        ).to(torch::kInt64).clone();

        return { t_prompt, t_response, t_mask,
                 static_cast<int64_t>(s.prompt_len),
                 static_cast<int64_t>(s.response_len) };
    }

private:
    DataLoaderConfig         cfg_;
    std::vector<RawSample>   samples_;
    std::vector<int32_t>     lengths_;
};

} // namespace dl
