#pragma once
//  collator.h — Batch collation with global pad + per-batch trim
//
//  Strategy
//  All samples on disk are ALREADY padded to max_seq_len
//  (done at tokenization time).  The collator:
//
//  1. Concatenates prompt_ids + response_ids into one
//     input sequence of length max_seq_len (combined).
//     Actually we store prompt and response separately,
//     so concatenation is done here.
//
//  2. Finds the maximum REAL token length across the batch
//     (prompt_len[i] + response_len[i] for each sample i).
//
//  3. TRIMS the padded tensors to that batch maximum —
//     removing trailing padding that no sample in this
//     batch actually uses.
//
//  Output tensors shape: [B, batch_max_len]
//  where batch_max_len = max(prompt_len[i] + response_len[i]) over batch.
//
//  Labels: -100 for prompt tokens (masked from loss),
//           token_id for response tokens.
// ============================================================

#include <algorithm>
#include <vector>
#include <torch/torch.h>

namespace dl {

struct Batch {
    torch::Tensor input_ids;       // [B, batch_max_len]
    torch::Tensor attention_mask;  // [B, batch_max_len]
    torch::Tensor labels;          // [B, batch_max_len]  -100 for prompt
    int64_t       batch_max_len;   // actual sequence length after trimming
};

class TrimPaddingCollator {
public:
    explicit TrimPaddingCollator(int64_t pad_token_id, int32_t max_seq_len)
        : pad_id_(pad_token_id), max_seq_len_(max_seq_len) {}

    // items: vector of (prompt_ids[max_seq_len], response_ids[max_seq_len],
    //                   attention_mask[max_seq_len], prompt_len, response_len)
    Batch operator()(
        const std::vector<std::tuple<
            torch::Tensor,   // prompt_ids    [max_seq_len]
            torch::Tensor,   // response_ids  [max_seq_len]
            torch::Tensor,   // attention_mask[max_seq_len]
            int64_t,         // prompt_len
            int64_t          // response_len
        >>& items) const
    {
        const int64_t B = static_cast<int64_t>(items.size());

        
                // Step 1: find batch maximum real length
        int64_t batch_max_len = 0;
        for (const auto& it : items) {
            int64_t real = it.prompt_ids.size(0) + it.response_ids.size(0);
            batch_max_len = std::max(batch_max_len, real);
        }
        batch_max_len = std::min(batch_max_len, (int64_t)max_seq_len_);

        // Step 2: allocate padded tensors at batch_max_len
        auto input_ids = torch::full({B, batch_max_len}, pad_id_, torch::kInt64);
        auto attention_mask = torch::zeros({B, batch_max_len}, torch::kInt64);
        auto labels = torch::full({B, batch_max_len}, -100LL, torch::kInt64);

        // Step 3: fill each sample — concat prompt + response, build mask
        for (int64_t i = 0; i < B; ++i) {
            int64_t p_len = items[i].prompt_ids.size(0);
            int64_t r_len = items[i].response_ids.size(0);
            int64_t real  = std::min(p_len + r_len, batch_max_len);

            // prompt into input_ids[i, 0:p_len]
            input_ids[i].slice(0, 0, p_len).copy_(items[i].prompt_ids);

            // response into input_ids[i, p_len:p_len+r_len]
            int64_t r_actual = std::min(r_len, batch_max_len - p_len);
            if (r_actual > 0)
                input_ids[i].slice(0, p_len, p_len + r_actual)
                            .copy_(items[i].response_ids.slice(0, 0, r_actual));

            // attention mask: 1 for all real tokens
            attention_mask[i].slice(0, 0, real).fill_(1);

            // labels: -100 for prompt, real ids for response
            if (r_actual > 0)
                labels[i].slice(0, p_len, p_len + r_actual)
                        .copy_(items[i].response_ids.slice(0, 0, r_actual));
        }

        return { input_ids, attention_mask, labels, batch_max_len };
    }

private:
    int64_t pad_id_;
    int32_t max_seq_len_;
};

} // namespace dl
