#pragma once
// Batch structs:
//   SFTBatch         — input_ids, attention_mask, labels
//   RewardModelBatch — chosen/rejected input_ids + masks (no labels)
//   DPOBatch         — chosen/rejected input_ids + masks + labels
//   RolloutBatch     — prompt_ids, prompt_mask (left-padded)
//
// UnifiedCollator dispatches to the right logic based on mode.
// ============================================================

#include "dataset.h"
#include <vector>
#include <algorithm>
#include <torch/torch.h>

namespace dl {

// SFT batch  (same as original Batch, kept compatible)
struct SFTBatch {
    torch::Tensor input_ids;       // (B, L)
    torch::Tensor attention_mask; 
    torch::Tensor labels;         // prompt tokens = -100
};
using Batch = SFTBatch;  // backward compat alias
struct RewardModelBatch {
    torch::Tensor chosen_input_ids;      
    torch::Tensor chosen_attention_mask;
    torch::Tensor rejected_input_ids;   
    torch::Tensor rejected_attention_mask; 
};

// DPO batch — same layout + token labels on both sides
struct DPOBatch {
    torch::Tensor chosen_input_ids;
    torch::Tensor chosen_attention_mask;
    torch::Tensor chosen_labels;          // prompt portion = -100
    torch::Tensor rejected_input_ids;
    torch::Tensor rejected_attention_mask;
    torch::Tensor rejected_labels;        // prompt portion = -100
};
struct RolloutBatch {
    torch::Tensor prompt_ids;   // (B, max_prompt_len + max_gen_len)
    torch::Tensor prompt_mask;//same
};
// (classic DynamicPaddingCollator, preserved)
class DynamicPaddingCollator {
public:
    explicit DynamicPaddingCollator(int64_t pad_token_id)
        : pad_id_(pad_token_id) {}

    SFTBatch operator()(
        const std::vector<std::pair<torch::Tensor, torch::Tensor>>& batch) const
    {
        const int64_t B = static_cast<int64_t>(batch.size());
        std::vector<torch::Tensor> inputs, label_seqs;
        inputs.reserve(B); label_seqs.reserve(B);
        int64_t max_len = 0;

        for (const auto& [instr, resp] : batch) {
            auto full = torch::cat({instr, resp}, 0);
            auto lbl  = torch::cat({
                torch::full({instr.size(0)}, -100LL, torch::kInt64),
                resp
            }, 0);
            max_len = std::max(max_len, full.size(0));
            inputs.push_back(full);
            label_seqs.push_back(lbl);
        }

        auto input_ids    = torch::full({B, max_len}, pad_id_,  torch::kInt64);
        auto attn_mask    = torch::zeros({B, max_len},           torch::kInt64);
        auto label_tensor = torch::full({B, max_len}, -100LL,   torch::kInt64);

        for (int64_t i = 0; i < B; ++i) {
            int64_t len = inputs[i].size(0);
            input_ids[i].slice(0,0,len).copy_(inputs[i]);
            attn_mask[i].slice(0,0,len).fill_(1);
            label_tensor[i].slice(0,0,len).copy_(label_seqs[i]);
        }
        return {input_ids, attn_mask, label_tensor};
    }

private:
    int64_t pad_id_;
};

namespace detail {
// Concatenate prompt + side and pad a batch to max_len
inline std::tuple<torch::Tensor, torch::Tensor>
pad_sequences(const std::vector<torch::Tensor>& seqs,
              int64_t pad_id)
{
    int64_t B = seqs.size();
    int64_t max_len = 0;
    for (auto& s : seqs) max_len = std::max(max_len, s.size(0));

    auto ids  = torch::full({B, max_len}, pad_id, torch::kInt64);
    auto mask = torch::zeros({B, max_len},         torch::kInt64);
    for (int64_t i = 0; i < B; ++i) {
        int64_t l = seqs[i].size(0);
        ids[i].slice(0,0,l).copy_(seqs[i]);
        mask[i].slice(0,0,l).fill_(1);
    }
    return {ids, mask};
}

// Build label tensor: mask prompt prefix with -100, keep rest
inline torch::Tensor build_labels(const torch::Tensor& input_ids,
                                   int64_t prompt_len)
{
    auto lbl = input_ids.clone();
    lbl.slice(1, 0, prompt_len).fill_(-100);
    return lbl;
}

} 
// Unified collator — dispatches on mode
class UnifiedCollator {
public:
    explicit UnifiedCollator(const DataLoaderConfig& cfg) : cfg_(cfg) {}

    //SFT  
    SFTBatch collate_sft(
        const std::vector<UnifiedDataset::SFTItem>& items) const
    {
        const int64_t B = items.size();
        std::vector<torch::Tensor> seqs, lbls;
        seqs.reserve(B); lbls.reserve(B);
        int64_t max_len = 0;

        for (const auto& it : items) {
            auto full = torch::cat({it.prompt_ids, it.response_ids}, 0);
            auto lbl  = torch::cat({
                torch::full({it.prompt_ids.size(0)}, -100LL, torch::kInt64),
                it.response_ids
            }, 0);
            max_len = std::max(max_len, full.size(0));
            seqs.push_back(full);
            lbls.push_back(lbl);
        }

        auto ids  = torch::full({B, max_len}, (int64_t)cfg_.pad_token_id, torch::kInt64);
        auto mask = torch::zeros({B, max_len}, torch::kInt64);
        auto lbl  = torch::full({B, max_len}, -100LL, torch::kInt64);

        for (int64_t i = 0; i < B; ++i) {
            int64_t l = seqs[i].size(0);
            ids[i].slice(0,0,l).copy_(seqs[i]);
            mask[i].slice(0,0,l).fill_(1);
            lbl[i].slice(0,0,l).copy_(lbls[i]);
        }
        return {ids, mask, lbl};
    }

    // Reward Model  
    RewardModelBatch collate_reward_model(
        const std::vector<UnifiedDataset::PreferenceItem>& items) const
    {
        const int64_t B = items.size();
        std::vector<torch::Tensor> cho_seqs, rej_seqs;
        cho_seqs.reserve(B); rej_seqs.reserve(B);

        for (const auto& it : items) {
            cho_seqs.push_back(torch::cat({it.prompt_ids, it.chosen_ids},   0));
            rej_seqs.push_back(torch::cat({it.prompt_ids, it.rejected_ids}, 0));
        }

        auto [cho_ids, cho_mask] = detail::pad_sequences(cho_seqs, cfg_.pad_token_id);
        auto [rej_ids, rej_mask] = detail::pad_sequences(rej_seqs, cfg_.pad_token_id);

        return {cho_ids, cho_mask, rej_ids, rej_mask};
    }

    // DPO  
    DPOBatch collate_dpo(
        const std::vector<UnifiedDataset::PreferenceItem>& items) const
    {
        const int64_t B = items.size();
        std::vector<torch::Tensor> cho_seqs, rej_seqs;
        std::vector<int64_t> prompt_lens;
        cho_seqs.reserve(B); rej_seqs.reserve(B); prompt_lens.reserve(B);

        for (const auto& it : items) {
            prompt_lens.push_back(it.prompt_ids.size(0));
            cho_seqs.push_back(torch::cat({it.prompt_ids, it.chosen_ids},   0));
            rej_seqs.push_back(torch::cat({it.prompt_ids, it.rejected_ids}, 0));
        }

        auto [cho_ids, cho_mask] = detail::pad_sequences(cho_seqs, cfg_.pad_token_id);
        auto [rej_ids, rej_mask] = detail::pad_sequences(rej_seqs, cfg_.pad_token_id);

        // Build labels: mask prompt prefix per sample
        auto cho_lbl = cho_ids.clone().fill_(-100);
        auto rej_lbl = rej_ids.clone().fill_(-100);
        for (int64_t i = 0; i < B; ++i) {
            int64_t pl = prompt_lens[i];
            int64_t cho_end = cho_seqs[i].size(0);
            int64_t rej_end = rej_seqs[i].size(0);
            if (cho_end > pl)
                cho_lbl[i].slice(0, pl, cho_end).copy_(cho_seqs[i].slice(0, pl, cho_end));
            if (rej_end > pl)
                rej_lbl[i].slice(0, pl, rej_end).copy_(rej_seqs[i].slice(0, pl, rej_end));
        }

        return {cho_ids, cho_mask, cho_lbl, rej_ids, rej_mask, rej_lbl};
    }

    // Rollout 
    RolloutBatch collate_rollout(
        const std::vector<UnifiedDataset::RolloutItem>& items) const
    {
        // Items are already individually left-padded to (max_prompt_len + max_gen_len)
        // Just stack them
        const int64_t B = items.size();
        std::vector<torch::Tensor> ids_vec, mask_vec;
        ids_vec.reserve(B); mask_vec.reserve(B);
        for (const auto& it : items) {
            ids_vec.push_back(it.prompt_ids.unsqueeze(0));
            mask_vec.push_back(it.prompt_mask.unsqueeze(0));
        }
        return {torch::cat(ids_vec, 0), torch::cat(mask_vec, 0)};
    }

private:
    DataLoaderConfig cfg_;
};

} 