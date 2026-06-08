#pragma once
// ============================================================
//  dataset.h — Instruction/Response Binary Dataset (SFT)
// ============================================================
//
//  Reads:
//    instructions.bin
//    responses.bin
//
//  Format of each file:
//    uint32 n_samples
//    uint16 length
//    uint32 token_ids[length]
//
//  Output sample:
//    instruction_ids
//    response_ids
//    total_length = instruction + response
// ============================================================

#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <algorithm>

#include <torch/torch.h>

namespace dl {

// ─────────────────────────────────────────────────────────────
// constants
// ─────────────────────────────────────────────────────────────
static constexpr uint32_t kMagic   = 0xD47A1D07u;
static constexpr uint32_t kVersion = 1u;

// ─────────────────────────────────────────────────────────────
// Raw sample (SFT pair)
// ─────────────────────────────────────────────────────────────
struct RawSample {
    std::vector<int32_t> instruction_ids;
    std::vector<int32_t> response_ids;

    int32_t instruction_len() const {
        return static_cast<int32_t>(instruction_ids.size());
    }

    int32_t response_len() const {
        return static_cast<int32_t>(response_ids.size());
    }

    int32_t total_len() const {
        return instruction_len() + response_len();
    }
};

// ─────────────────────────────────────────────────────────────
// config
// ─────────────────────────────────────────────────────────────
struct DataLoaderConfig {
    int32_t max_seq_len  = 2048;
    int32_t pad_token_id = 0;
    int32_t eos_token_id = 2;

    int32_t batch_size = 8;

    std::vector<int32_t> bucket_boundaries = {
        64, 128, 256, 512, 1024, 2048
    };

    int32_t num_workers    = 4;
    int32_t prefetch_factor = 2;  
    bool    shuffle        = true;
    int64_t seed           = 42;
};

// ─────────────────────────────────────────────────────────────
// Binary reader
// ─────────────────────────────────────────────────────────────
class BinaryDatasetReader {
public:

    static std::vector<RawSample> load(
        const std::string& instructions_path,
        const std::string& responses_path)
    {
        auto instructions = load_file(instructions_path);
        auto responses    = load_file(responses_path);

        if (instructions.size() != responses.size())
            throw std::runtime_error(
                "Dataset mismatch: instructions != responses");

        std::vector<RawSample> out;
        out.reserve(instructions.size());

        for (size_t i = 0; i < instructions.size(); ++i) {
            RawSample s;
            s.instruction_ids = std::move(instructions[i]);
            s.response_ids    = std::move(responses[i]);
            out.push_back(std::move(s));
        }

        return out;
    }

private:

    static std::vector<std::vector<int32_t>>
    load_file(const std::string& path)
    {
        std::ifstream f(path, std::ios::binary);

        if (!f)
            throw std::runtime_error("Cannot open " + path);

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

    template<typename T>
    static void read_pod(std::ifstream& f, T& out)
    {
        f.read(reinterpret_cast<char*>(&out), sizeof(T));
        if (!f)
            throw std::runtime_error("Unexpected EOF");
    }
};

// ─────────────────────────────────────────────────────────────
// Dataset
// ─────────────────────────────────────────────────────────────
class InstructionDataset {
public:
    InstructionDataset(
        std::vector<RawSample> samples,
        const DataLoaderConfig& cfg)
        : cfg_(cfg),
          samples_(std::move(samples))
    {
        lengths_.reserve(samples_.size());

        for (const auto& s : samples_) {
            lengths_.push_back(
                std::min(s.total_len(), cfg_.max_seq_len)
            );
        }
    }

    std::size_t size() const { return samples_.size(); }

    const std::vector<int32_t>& lengths() const {
        return lengths_;
    }

    struct Item {
        torch::Tensor instruction_ids;
        torch::Tensor response_ids;
        int64_t instruction_length;
        int64_t response_length;
    };

    Item get(int64_t idx) const
    {
    const auto& s = samples_[static_cast<size_t>(idx)];

    std::vector<int32_t> instr = s.instruction_ids;
    std::vector<int32_t> resp  = s.response_ids;

    int32_t total = (int32_t)instr.size() + (int32_t)resp.size();

    if (total > cfg_.max_seq_len) {
        // Step 1: clamp instruction to at most half the budget
        int32_t instr_len = std::min((int32_t)instr.size(), cfg_.max_seq_len / 2);
        // Step 2: give response whatever is left
        int32_t resp_len  = cfg_.max_seq_len - instr_len;
        // Step 3: clamp response to its actual size (don't over-allocate)
        resp_len  = std::min(resp_len,  (int32_t)resp.size());
        // Step 4: now that we know the true resp_len, give instruction the real remainder
        instr_len = cfg_.max_seq_len - resp_len;
        instr_len = std::min(instr_len, (int32_t)instr.size());

        instr.resize(instr_len);
        resp.resize(resp_len);

        // restore EOS that truncation removed
        instr.back() = cfg_.eos_token_id;
        resp.back()  = cfg_.eos_token_id;
    }

    if (instr.empty()) instr.push_back(cfg_.eos_token_id);
    if (resp.empty())  resp.push_back(cfg_.eos_token_id);

    auto t_instr = torch::from_blob(
        instr.data(),
        {(int64_t)instr.size()},
        torch::kInt32
    ).to(torch::kInt64).clone();

    auto t_resp = torch::from_blob(
        resp.data(),
        {(int64_t)resp.size()},
        torch::kInt32
    ).to(torch::kInt64).clone();

    return {
        t_instr,
        t_resp,
        (int64_t)t_instr.size(0),
        (int64_t)t_resp.size(0)
    };
}

private:
    DataLoaderConfig cfg_;
    std::vector<RawSample> samples_;
    std::vector<int32_t> lengths_;
};

} // namespace dl
