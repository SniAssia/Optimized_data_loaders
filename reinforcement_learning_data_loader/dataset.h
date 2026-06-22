#pragma once
// Supports four training modes:
//   SFT          — (prompt, response) pairs
//   REWARD_MODEL — (prompt, chosen, rejected) triplets (no labels)
//   DPO          — (prompt, chosen, rejected) triplets (with labels)
//   ROLLOUT      — prompt-only, left-padded, with generation space
//
// Binary file format (same for all files):
//   uint32  n_samples
//   [per sample]
//     uint16  length
//     uint32  token_ids[length]
// ============================================================

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <torch/torch.h>

namespace dl {

// Training mode
enum class TrainingMode {
    SFT,           // supervised fine-tuning
    REWARD_MODEL,  // preference model training (no token labels)
    DPO,           // direct preference optimisation (with token labels)
    ROLLOUT        // RL prompt delivery (PPO / GRPO rollout phase)
};

// Unified raw sample
struct RawSample {
    std::vector<int32_t> prompt_ids;    // always present
    std::vector<int32_t> response_ids;  // SFT only
    std::vector<int32_t> chosen_ids;    // REWARD_MODEL / DPO
    std::vector<int32_t> rejected_ids;  // REWARD_MODEL / DPO
};

struct DataLoaderConfig {
    TrainingMode mode = TrainingMode::SFT;
    // universal
    int32_t batch_size      = 8;
    int32_t num_workers     = 4;
    int32_t prefetch_factor = 2;
    bool    shuffle         = true;
    int64_t seed            = 42;
    // sequence budget
    int32_t max_seq_len   = 2048;  // SFT / REWARD_MODEL / DPO
    int32_t max_prompt_len = 512;  // ROLLOUT
    int32_t max_gen_len    = 1024; // ROLLOUT (generation space)
    int32_t pad_token_id = 0;
    int32_t eos_token_id = 2;
    int32_t bos_token_id = 1;
    std::vector<int32_t> bucket_boundaries = {64, 128, 256, 512, 1024, 2048};
    //  number of responses generated per prompt (read by training loop)
    int32_t grpo_group_size = 8;
    std::string prompt_path   = "prompts.bin";
    std::string response_path = "responses.bin";
    std::string chosen_path   = "chosen.bin";
    std::string rejected_path = "rejected.bin";
    std::string manifest_path  = ""; 
};

// Lowest-level binary reader: reads one .bin file into a vector
// of token-id vectors. Format: uint32 n_samples, then per sample
// uint16 length + uint32 token_ids[length].
class BinarySequenceReader {
public:
    static std::vector<std::vector<int32_t>> load(const std::string& path) {
        std::ifstream f(path, std::ios::binary);
        if (!f) throw std::runtime_error("Cannot open: " + path);
        uint32_t n_samples;
        read_pod(f, n_samples);
        std::vector<std::vector<int32_t>> data;
        data.reserve(n_samples);
        for (uint32_t i = 0; i < n_samples; ++i) {
            uint16_t length;
            read_pod(f, length);
            std::vector<int32_t> seq(length);
            for (uint16_t j = 0; j < length; ++j) {
                uint32_t tok;
                read_pod(f, tok);
                seq[j] = static_cast<int32_t>(tok);
            }
            data.push_back(std::move(seq));
        }
        return data;
    }

private:
    template<typename T>
    static void read_pod(std::ifstream& f, T& out) {
        f.read(reinterpret_cast<char*>(&out), sizeof(T));
        if (!f) throw std::runtime_error("Unexpected EOF in binary file");
    }
};

class UnifiedDatasetReader {
public:
    static std::vector<RawSample> load(const DataLoaderConfig& cfg) {
        if (!cfg.manifest_path.empty()) {
            // sharded layout — delegate to ShardManager
            return ShardManager::load(cfg.manifest_path, cfg);
        }
        // flat layout (legacy / --shard-size 0)
        switch (cfg.mode) {
            case TrainingMode::SFT:
                return load_sft(cfg);
            case TrainingMode::REWARD_MODEL:
            case TrainingMode::DPO:
                return load_preference(cfg);
            case TrainingMode::ROLLOUT:
                return load_rollout(cfg);
        }
        throw std::runtime_error("Unknown training mode");
        }

private:
    static std::vector<RawSample> load_sft(const DataLoaderConfig& cfg) {
        auto prompts   = BinarySequenceReader::load(cfg.prompt_path);
        auto responses = BinarySequenceReader::load(cfg.response_path);
        if (prompts.size() != responses.size())
            throw std::runtime_error("SFT: prompt/response count mismatch");

        std::vector<RawSample> out;
        out.reserve(prompts.size());
        for (size_t i = 0; i < prompts.size(); ++i) {
            RawSample s;
            s.prompt_ids   = std::move(prompts[i]);
            s.response_ids = std::move(responses[i]);
            out.push_back(std::move(s));
        }
        return out;
    }

    static std::vector<RawSample> load_preference(const DataLoaderConfig& cfg) {
        auto prompts   = BinarySequenceReader::load(cfg.prompt_path);
        auto chosen    = BinarySequenceReader::load(cfg.chosen_path);
        auto rejected  = BinarySequenceReader::load(cfg.rejected_path);
        if (prompts.size() != chosen.size() || prompts.size() != rejected.size())
            throw std::runtime_error("Preference: prompt/chosen/rejected count mismatch");

        std::vector<RawSample> out;
        out.reserve(prompts.size());
        for (size_t i = 0; i < prompts.size(); ++i) {
            RawSample s;
            s.prompt_ids   = std::move(prompts[i]);
            s.chosen_ids   = std::move(chosen[i]);
            s.rejected_ids = std::move(rejected[i]);
            out.push_back(std::move(s));
        }
        return out;
    }

    static std::vector<RawSample> load_rollout(const DataLoaderConfig& cfg) {
        auto prompts = BinarySequenceReader::load(cfg.prompt_path);
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

class UnifiedDataset {
public:
    UnifiedDataset(std::vector<RawSample> samples, const DataLoaderConfig& cfg)
        : cfg_(cfg), samples_(std::move(samples))
    {
        lengths_.reserve(samples_.size());
        for (const auto& s : samples_)
            lengths_.push_back(compute_length(s));
    }

    std::size_t size() const { return samples_.size(); }
    const std::vector<int32_t>& lengths() const { return lengths_; }

    struct SFTItem {
        torch::Tensor prompt_ids;
        torch::Tensor response_ids;
    };

    struct PreferenceItem {
        torch::Tensor prompt_ids;
        torch::Tensor chosen_ids;
        torch::Tensor rejected_ids;
    };

    struct RolloutItem {
        torch::Tensor prompt_ids; // left-padded, shape (max_prompt_len + max_gen_len,)
        torch::Tensor prompt_mask;
    };

    SFTItem get_sft(int64_t idx) const {
        const auto& s = samples_[idx];
        auto [prompt, response] = truncate_sft(s.prompt_ids, s.response_ids);
        return { to_tensor(prompt), to_tensor(response) };
    }

    PreferenceItem get_preference(int64_t idx) const {
        const auto& s = samples_[idx];
        auto chosen   = truncate_one_side(s.chosen_ids,   cfg_.max_seq_len - (int32_t)s.prompt_ids.size());
        auto rejected = truncate_one_side(s.rejected_ids, cfg_.max_seq_len - (int32_t)s.prompt_ids.size());
        auto prompt   = s.prompt_ids;
        if ((int32_t)prompt.size() > cfg_.max_seq_len / 4)
            prompt.resize(cfg_.max_seq_len / 4);
        return { to_tensor(prompt), to_tensor(chosen), to_tensor(rejected) };
    }

    RolloutItem get_rollout(int64_t idx) const {
        const auto& s = samples_[idx];
        auto prompt = s.prompt_ids;
        // right-truncate prompt to max_prompt_len
        if ((int32_t)prompt.size() > cfg_.max_prompt_len)
            prompt.resize(cfg_.max_prompt_len);

        int32_t total_len = cfg_.max_prompt_len + cfg_.max_gen_len;
        int32_t prompt_len = (int32_t)prompt.size();
        int32_t pad_len = cfg_.max_prompt_len - prompt_len;

        // left-pad: [pad ... pad | prompt tokens | 0 ... 0 (gen space)]
        std::vector<int64_t> ids(total_len, cfg_.pad_token_id);
        std::vector<int64_t> mask(total_len, 0);

        for (int32_t i = 0; i < prompt_len; ++i) {
            ids[pad_len + i]  = prompt[i];
            mask[pad_len + i] = 1;
        }

        auto t_ids  = torch::from_blob(ids.data(),  {total_len}, torch::kInt64).clone();
        auto t_mask = torch::from_blob(mask.data(), {total_len}, torch::kInt64).clone();
        return { t_ids, t_mask };
    }

private:
    DataLoaderConfig         cfg_;
    std::vector<RawSample>   samples_;
    std::vector<int32_t>     lengths_;

    int32_t compute_length(const RawSample& s) const {
        switch (cfg_.mode) {
            case TrainingMode::SFT:
                return std::min(
                    (int32_t)(s.prompt_ids.size() + s.response_ids.size()),
                    cfg_.max_seq_len);
            case TrainingMode::REWARD_MODEL:
            case TrainingMode::DPO: {
                int32_t prompt_len = std::min((int32_t)s.prompt_ids.size(), cfg_.max_seq_len / 4);
                int32_t side = cfg_.max_seq_len - prompt_len;
                int32_t longest = std::max((int32_t)s.chosen_ids.size(),
                                           (int32_t)s.rejected_ids.size());
                return std::min(prompt_len + longest, cfg_.max_seq_len - side + longest);
            }
            case TrainingMode::ROLLOUT:
                return std::min((int32_t)s.prompt_ids.size(), cfg_.max_prompt_len);
        }
        return 0;
    }

    std::pair<std::vector<int32_t>, std::vector<int32_t>>
    truncate_sft(const std::vector<int32_t>& prompt,
                 const std::vector<int32_t>& response) const
    {
        std::vector<int32_t> p = prompt;
        std::vector<int32_t> r = response;
        int32_t total = (int32_t)(p.size() + r.size());
        if (total <= cfg_.max_seq_len) return {p, r};

        // Give response up to half the budget; give prompt the rest
        int32_t r_len = std::min((int32_t)r.size(), cfg_.max_seq_len / 2);
        int32_t p_len = std::min((int32_t)p.size(), cfg_.max_seq_len - r_len);
        r_len = cfg_.max_seq_len - p_len;
        r_len = std::min(r_len, (int32_t)r.size());
        p.resize(p_len);
        r.resize(r_len);
        if (!p.empty()) p.back() = cfg_.eos_token_id;
        if (!r.empty()) r.back() = cfg_.eos_token_id;
        return {p, r};
    }

    std::vector<int32_t> truncate_one_side(const std::vector<int32_t>& seq,
                                            int32_t budget) const {
        if (budget <= 0) return {};
        std::vector<int32_t> out = seq;
        if ((int32_t)out.size() > budget) {
            out.resize(budget);
            out.back() = cfg_.eos_token_id;
        }
        return out;
    }

    static torch::Tensor to_tensor(const std::vector<int32_t>& v) {
        auto t = torch::from_blob(
            const_cast<int32_t*>(v.data()),
            {(int64_t)v.size()}, torch::kInt32
        ).to(torch::kInt64).clone();
        return t;
    }
};

// ── Backward-compat alias (SFT legacy API) ───────────────────
using RawSampleLegacy = RawSample;

class BinaryDatasetReader {
public:
    static std::vector<RawSample> load(
        const std::string& instructions_path,
        const std::string& responses_path)
    {
        DataLoaderConfig cfg;
        cfg.mode          = TrainingMode::SFT;
        cfg.prompt_path   = instructions_path;
        cfg.response_path = responses_path;
        return UnifiedDatasetReader::load(cfg);
    }
};

// ── Legacy InstructionDataset wrapper ────────────────────────
class InstructionDataset {
public:
    InstructionDataset(std::vector<RawSample> samples, const DataLoaderConfig& cfg)
        : inner_(std::move(samples), cfg) {}

    std::size_t size() const { return inner_.size(); }
    const std::vector<int32_t>& lengths() const { return inner_.lengths(); }

    struct Item {
        torch::Tensor instruction_ids;
        torch::Tensor response_ids;
        int64_t instruction_length;
        int64_t response_length;
    };

    Item get(int64_t idx) const {
        auto sft = inner_.get_sft(idx);
        return {
            sft.prompt_ids,
            sft.response_ids,
            sft.prompt_ids.size(0),
            sft.response_ids.size(0)
        };
    }

private:
    UnifiedDataset inner_;
};

} 
#include "shard_manager.h"
