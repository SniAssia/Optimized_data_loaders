#pragma once
//  dataset.h — Sharded Instruction/Response Binary Dataset (SFT)
//
//  Binary sample format (samples.bin per shard):
//    uint32  n_samples
//    per sample:
//      uint16  prompt_len
//      uint32  prompt_ids[prompt_len]
//      uint16  response_len
//      uint32  response_ids[response_len]
//
//  No padding on disk. No attention mask on disk.
//  Padding and attention mask built at batch time in collator.h

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include <torch/torch.h>

namespace dl {

struct RawSample {
    std::vector<int32_t> prompt_ids;
    std::vector<int32_t> response_ids;
    uint16_t prompt_len   = 0;
    uint16_t response_len = 0;

    int32_t total_real_len() const {
        return static_cast<int32_t>(prompt_len) +
               static_cast<int32_t>(response_len);
    }
};

// Shard entry (from manifest.json)
struct ShardEntry {
    int         shard_idx  = 0;
    int         n_samples  = 0;
    std::string dir;
};

struct DataLoaderConfig {
    int32_t max_seq_len     = 512;
    int32_t pad_token_id    = 0;
    int32_t eos_token_id    = 2;
    int32_t batch_size      = 8;
    int32_t num_workers     = 4;
    int32_t prefetch_factor = 2;
    int32_t window_size     = 4;    // shards loaded simultaneously
    bool    shuffle         = true;
    int64_t seed            = 42;
};

// Low-level binary reader
namespace detail {

template<typename T>
static void read_pod(std::ifstream& f, T& out) {
    f.read(reinterpret_cast<char*>(&out), sizeof(T));
    if (!f) throw std::runtime_error("Unexpected EOF in binary file");
}

static std::vector<RawSample> read_samples_bin(const std::string& path)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open: " + path);

    uint32_t n_samples;
    read_pod(f, n_samples);

    std::vector<RawSample> out;
    out.reserve(n_samples);

    for (uint32_t i = 0; i < n_samples; ++i) {
        RawSample s;

        uint16_t prompt_len;
        read_pod(f, prompt_len);
        s.prompt_len = prompt_len;
        s.prompt_ids.resize(prompt_len);
        for (uint16_t j = 0; j < prompt_len; ++j) {
            uint32_t tok; read_pod(f, tok);
            s.prompt_ids[j] = static_cast<int32_t>(tok);
        }

        uint16_t response_len;
        read_pod(f, response_len);
        s.response_len = response_len;
        s.response_ids.resize(response_len);
        for (uint16_t j = 0; j < response_len; ++j) {
            uint32_t tok; read_pod(f, tok);
            s.response_ids[j] = static_cast<int32_t>(tok);
        }

        out.push_back(std::move(s));
    }
    return out;
}

} // namespace detail

// Minimal JSON manifest parser
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
    int32_t                 max_seq_len  = 512;
    int32_t                 pad_token_id = 0;
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
        auto obj_open = arr.find('{', cursor);
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
// Lightweight — stores only manifest metadata.
// Does NOT load any samples at construction time.
// load_shard(idx) loads one shard on demand.
// ─────────────────────────────────────────────────────────────
class InstructionDataset {
public:
    explicit InstructionDataset(const std::string& manifest_path,
                                 const DataLoaderConfig& cfg)
        : cfg_(cfg)
    {
        auto manifest    = manifest_parser::parse(manifest_path);
        cfg_.max_seq_len  = manifest.max_seq_len;
        cfg_.pad_token_id = manifest.pad_token_id;
        shards_           = manifest.shards;

        total_samples_ = 0;
        for (const auto& s : shards_)
            total_samples_ += s.n_samples;

        std::cout << "[InstructionDataset] "
                  << shards_.size() << " shards, "
                  << total_samples_ << " total samples "
                  << "(nothing loaded into RAM yet)\n";
    }

    // Load one shard from disk — called by prefetcher per window
    std::vector<RawSample> load_shard(int shard_idx) const {
        const auto& entry = shards_[static_cast<std::size_t>(shard_idx)];
        std::string path  = entry.dir + "/samples.bin";
        return detail::read_samples_bin(path);
    }

    int  num_shards()     const { return static_cast<int>(shards_.size()); }
    int  total_samples()  const { return total_samples_; }
    const DataLoaderConfig& config() const { return cfg_; }

    // Item type used by collator
    struct Item {
        torch::Tensor prompt_ids;    // [prompt_len]
        torch::Tensor response_ids;  // [response_len]
        int64_t       prompt_len;
        int64_t       response_len;
    };

    // Convert one RawSample to tensors (called inside collator)
    static Item to_item(const RawSample& s) {
        auto t_prompt = torch::from_blob(
            const_cast<int32_t*>(s.prompt_ids.data()),
            {static_cast<int64_t>(s.prompt_len)},
            torch::kInt32
        ).to(torch::kInt64).clone();

        auto t_response = torch::from_blob(
            const_cast<int32_t*>(s.response_ids.data()),
            {static_cast<int64_t>(s.response_len)},
            torch::kInt32
        ).to(torch::kInt64).clone();

        Item item;
        item.prompt_ids   = std::move(t_prompt);
        item.response_ids = std::move(t_response);
        item.prompt_len   = static_cast<int64_t>(s.prompt_len);
        item.response_len = static_cast<int64_t>(s.response_len);
        return item;
    }

private:
    DataLoaderConfig         cfg_;
    std::vector<ShardEntry>  shards_;
    int                      total_samples_ = 0;
};

} // namespace dl