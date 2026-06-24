#pragma once
//  collator.h — Per-batch trim padding collator
//
//  Receives raw unpadded tensors per sample.
//  Pads each batch only to its own maximum real token length.
//  Builds attention mask at batch time.
//
//  Input per item: (prompt_ids[prompt_len], response_ids[response_len],
//                   prompt_len, response_len)
//
//  Output tensors shape: [B, batch_max_len]
//  Labels: -100 for prompt tokens, token_id for response tokens.

#include <algorithm>
#include <tuple>
#include <vector>
#include <torch/torch.h>

namespace dl {

struct Batch {
    torch::Tensor input_ids;       // [B, batch_max_len]
    torch::Tensor attention_mask;  // [B, batch_max_len]
    torch::Tensor labels;          // [B, batch_max_len]  -100 for prompt
    int64_t       batch_max_len;
};

class TrimPaddingCollator {
public:
    explicit TrimPaddingCollator(int64_t pad_token_id, int32_t max_seq_len)
        : pad_id_(pad_token_id), max_seq_len_(max_seq_len) {}

    // items: vector of (prompt_ids[prompt_len], response_ids[response_len],
    //                   prompt_len, response_len)
    // No attention_mask in input — built here from real lengths.
    Batch operator()(
        const std::vector<std::tuple
            torch::Tensor,   // prompt_ids   [prompt_len]
            torch::Tensor,   // response_ids [response_len]
            int64_t,         // prompt_len
            int64_t          // response_len
        >>& items) const
    {
        const int64_t B = static_cast<int64_t>(items.size());

        // ── Step 1: find batch_max_len ────────────────────────
        int64_t batch_max_len = 0;
        for (const auto& it : items) {
            int64_t real = std::get<2>(it) + std::get<3>(it);
            batch_max_len = std::max(batch_max_len, real);
        }
        batch_max_len = std::min(batch_max_len,
                                  static_cast<int64_t>(max_seq_len_));

        // ── Step 2: allocate output tensors ───────────────────
        auto input_ids      = torch::full(
            {B, batch_max_len}, pad_id_, torch::kInt64);
        auto attention_mask = torch::zeros(
            {B, batch_max_len}, torch::kInt64);
        auto labels         = torch::full(
            {B, batch_max_len}, static_cast<int64_t>(-100), torch::kInt64);

        // ── Step 3: fill each sample ──────────────────────────
        for (int64_t i = 0; i < B; ++i) {
            const auto& t_prompt   = std::get<0>(items[i]);
            const auto& t_response = std::get<1>(items[i]);
            const int64_t p_len    = std::get<2>(items[i]);
            const int64_t r_len    = std::get<3>(items[i]);

            // clamp to batch_max_len
            const int64_t p_actual = std::min(p_len, batch_max_len);
            const int64_t r_actual = std::min(r_len, batch_max_len - p_actual);
            const int64_t real     = p_actual + r_actual;

            // prompt → input_ids[i, 0:p_actual]
            if (p_actual > 0)
                input_ids[i].slice(0, 0, p_actual)
                    .copy_(t_prompt.slice(0, 0, p_actual));

            // response → input_ids[i, p_actual:p_actual+r_actual]
            if (r_actual > 0)
                input_ids[i].slice(0, p_actual, p_actual + r_actual)
                    .copy_(t_response.slice(0, 0, r_actual));

            // attention mask: 1 for all real tokens
            attention_mask[i].slice(0, 0, real).fill_(1);

            // labels: -100 for prompt positions, real ids for response
            if (r_actual > 0)
                labels[i].slice(0, p_actual, p_actual + r_actual)
                    .copy_(t_response.slice(0, 0, r_actual));
        }

        return { input_ids, attention_mask, labels, batch_max_len };
    }

private:
    int64_t pad_id_;
    int32_t max_seq_len_;
};

} // namespace dl