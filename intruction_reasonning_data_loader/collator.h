#pragma once
//  Input:
//    batch of (instruction_ids, response_ids) pairs
//
//  Output:
//    input_ids      = instruction + response
//    attention_mask  = 1 for real tokens
//    labels          = -100 for instruction, response tokens for learning
// ============================================================

#include <vector>
#include <algorithm>
#include <torch/torch.h>

namespace dl {

struct Batch {
    torch::Tensor input_ids;       // (B, L)
    torch::Tensor attention_mask;  // (B, L)
    torch::Tensor labels;          // (B, L)
};

class DynamicPaddingCollator {
public:
    explicit DynamicPaddingCollator(int64_t pad_token_id)
        : pad_id_(pad_token_id) {}

    // FIX: accept vector<pair<Tensor,Tensor>> to match what
    //      CUDAPrefetcher passes (instruction_ids, response_ids)
    Batch operator()(
        const std::vector<
            std::pair<torch::Tensor, torch::Tensor>
        >& batch) const
    {
        const int64_t B = static_cast<int64_t>(batch.size());

        std::vector<torch::Tensor> inputs;
        std::vector<torch::Tensor> label_seqs;

        inputs.reserve(B);
        label_seqs.reserve(B);

        int64_t max_len = 0;

        for (const auto& item : batch) {

            const auto& instr = item.first;
            const auto& resp  = item.second;

            auto full = torch::cat({instr, resp}, 0);

            auto lbl = torch::cat({
                torch::full(
                    {instr.size(0)},
                    static_cast<int64_t>(-100),
                    torch::kInt64
                ),
                resp
            }, 0);

            max_len = std::max(max_len, full.size(0));

            inputs.push_back(full);
            label_seqs.push_back(lbl);
        }

        auto input_ids = torch::full(
            {B, max_len},
            pad_id_,
            torch::kInt64
        );

        auto attention_mask = torch::zeros(
            {B, max_len},
            torch::kInt64
        );

        auto label_tensor = torch::full(
            {B, max_len},
            static_cast<int64_t>(-100),
            torch::kInt64
        );

        for (int64_t i = 0; i < B; ++i) {

            int64_t len = inputs[i].size(0);

            input_ids[i].slice(0, 0, len).copy_(inputs[i]);
            attention_mask[i].slice(0, 0, len).fill_(1);
            label_tensor[i].slice(0, 0, len).copy_(label_seqs[i]);
        }

        return {
            input_ids,
            attention_mask,
            label_tensor
        };
    }

private:
    int64_t pad_id_;
};

} // namespace dl
