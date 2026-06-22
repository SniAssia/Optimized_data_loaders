#pragma once
// shard_manager.h — Parallel multi-shard loader
// Parses manifest.json produced by tokenize_dataset.py,
// loads all shards concurrently, merges into flat vector<RawSample>.
#include <fstream>
#include <future>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
namespace dl {
//Minimal JSON manifest parser
//Only understands the exact structure written by tokenize_dataset.py:
// { "mode": "...", "shards": [ {"shard":N, "n_samples":N, "dir":"..."}, ... ] }
struct ShardEntry {
    int  shard_idx=0;
    int n_samples=0;
    std::string dir;
};
struct Manifest{
    std::string mode;
    std::vector<ShardEntry> shards;
};
inline std::string extract_string(const std::string& json,
                                   const std::string& key){
    //finds "key": "value" and returns value
    std::string search = "\"" + key + "\"";
    auto pos = json.find(search);
    if (pos == std::string::npos) return "";
    pos = json.find('"', pos + search.size() + 1);//opening quote
    if (pos == std::string::npos) return "";
    auto end = json.find('"', pos + 1);
    return json.substr(pos + 1, end - pos - 1);
}
inline int extract_int(const std::string& json, const std::string& key){
    std::string search = "\"" + key + "\"";
    auto pos = json.find(search);
    if (pos == std::string::npos) return 0;
    pos = json.find_first_of("-0123456789", pos + search.size());
    if (pos == std::string::npos) return 0;
    return std::stoi(json.substr(pos));
}
inline Manifest parse_manifest(const std::string& path){
    std::ifstream f(path);
    if (!f) throw std::runtime_error("Cannot open manifest: " + path);
    std::ostringstream ss;
    ss << f.rdbuf();
    std::string json = ss.str();
    Manifest manifest;
    manifest.mode = extract_string(json, "mode");
    // parse each { ... } object inside "shards": [ ... ]
    auto shards_pos = json.find("\"shards\"");
    if (shards_pos == std::string::npos)
        throw std::runtime_error("manifest.json: missing 'shards' key");
    auto arr_open = json.find('[', shards_pos);
    auto arr_close = json.rfind(']');
    std::string arr = json.substr(arr_open + 1, arr_close - arr_open - 1);
    std::size_t cursor = 0;
    while (true) {
        auto obj_open  = arr.find('{', cursor);
        auto obj_close = arr.find('}', obj_open);
        if (obj_open == std::string::npos) break;
        std::string obj = arr.substr(obj_open, obj_close - obj_open + 1);
        ShardEntry e;
        e.shard_idx = extract_int(obj, "shard");
        e.n_samples = extract_int(obj, "n_samples");
        e.dir  = extract_string(obj, "dir");
        manifest.shards.push_back(e);
        cursor = obj_close + 1;
    }
    // sort by shard index so merge order is deterministic
    std::sort(manifest.shards.begin(), manifest.shards.end(),
              [](const ShardEntry& a, const ShardEntry& b){
                  return a.shard_idx < b.shard_idx;
              });
    return manifest;
}
// Path helpers
inline std::string shard_file(const std::string& dir,
                               const std::string& filename){
    return dir + "/" + filename;
}

// Parallel shard loader 
class ShardManager {
public:
    // Loads all shards listed in manifest_path in parallel.
    // Returns a flat vector<RawSample> in shard order.
    static std::vector<RawSample> load(const std::string& manifest_path,
                                        const DataLoaderConfig& cfg){
        Manifest manifest = parse_manifest(manifest_path);
        if (manifest.shards.empty())
            throw std::runtime_error("manifest.json: no shards found");
        const std::size_t n = manifest.shards.size();
        // Launch one async task per shard
        std::vector<std::future<std::vector<RawSample>>> futures;
        futures.reserve(n);
        for (const auto& entry : manifest.shards) {
            futures.push_back(
                std::async(std::launch::async,
                           load_shard, entry, cfg)
            );
        }
        // Merge results in shard order
        std::vector<RawSample> all;
        std::size_t total_hint = 0;
        for (const auto& e : manifest.shards)
            total_hint += static_cast<std::size_t>(e.n_samples);
        all.reserve(total_hint);

        for (auto& fut : futures) {
            auto shard_samples = fut.get();
            for (auto& s : shard_samples)
                all.push_back(std::move(s));
        }

        return all;
    }

private:
    static std::vector<RawSample> load_shard(const ShardEntry& entry,
                                              const DataLoaderConfig& cfg)
    {
        switch (cfg.mode) {
            case TrainingMode::SFT:
                return load_sft(entry);
            case TrainingMode::REWARD_MODEL:
            case TrainingMode::DPO:
                return load_preference(entry);
            case TrainingMode::ROLLOUT:
                return load_rollout(entry);
        }
        throw std::runtime_error("ShardManager: unknown training mode");
    }

    static std::vector<RawSample> load_sft(const ShardEntry& entry)
    {
        auto prompts   = BinarySequenceReader::load(shard_file(entry.dir, "prompts.bin"));
        auto responses = BinarySequenceReader::load(shard_file(entry.dir, "responses.bin"));

        if (prompts.size() != responses.size())
            throw std::runtime_error(
                "ShardManager SFT: prompt/response count mismatch in " + entry.dir);

        std::vector<RawSample> out;
        out.reserve(prompts.size());
        for (std::size_t i = 0; i < prompts.size(); ++i) {
            RawSample s;
            s.prompt_ids   = std::move(prompts[i]);
            s.response_ids = std::move(responses[i]);
            out.push_back(std::move(s));
        }
        return out;
    }

    static std::vector<RawSample> load_preference(const ShardEntry& entry)
    {
        auto prompts  = BinarySequenceReader::load(shard_file(entry.dir, "prompts.bin"));
        auto chosen   = BinarySequenceReader::load(shard_file(entry.dir, "chosen.bin"));
        auto rejected = BinarySequenceReader::load(shard_file(entry.dir, "rejected.bin"));

        if (prompts.size() != chosen.size() || prompts.size() != rejected.size())
            throw std::runtime_error(
                "ShardManager preference: count mismatch in " + entry.dir);

        std::vector<RawSample> out;
        out.reserve(prompts.size());
        for (std::size_t i = 0; i < prompts.size(); ++i) {
            RawSample s;
            s.prompt_ids   = std::move(prompts[i]);
            s.chosen_ids   = std::move(chosen[i]);
            s.rejected_ids = std::move(rejected[i]);
            out.push_back(std::move(s));
        }
        return out;
    }

    static std::vector<RawSample> load_rollout(const ShardEntry& entry)
    {
        auto prompts = BinarySequenceReader::load(shard_file(entry.dir, "prompts.bin"));

        std::vector<RawSample> out;
        out.reserve(prompts.size());
        for (auto& p : prompts) {
            RawSample s;
            s.prompt_ids = std::move(p);
            out.push_back(std::move(s));
        }
        return out;
    }
};

} 