#pragma once
// rollout_buffer.h — In-memory Rollout Buffer (PPO / GRPO)
// The RolloutBuffer sits between the two inner phases of PPO
// and GRPO:
//   Phase A (rollout collection)
//     Training loop calls push() after each rollout batch:
//       prompt_ids        (B, prompt_len+gen_len)
//         response_ids      (B, gen_len)
//         old_log_probs     (B, gen_len)  — actor log-probs at rollout
//        ref_log_probs     (B, gen_len)  — reference model log-probs
//         rewards           (B,)          — from reward model / rule
//      advantages        (B, gen_len)  — computed via GAE or group stats
//
//   Phase B (update epochs)
//     Training loop calls next_minibatch() to iterate over
//     the stored data in gradient-update-sized mini-batches.
//     Call clear() after all PPO epochs to reset for the next
//     rollout collection phase.
// GRPO note:
//   For GRPO the training loop expands each prompt to G copies
//   before pushing.  The buffer is agnostic to this: it stores
//   whatever tensors it receives.  After expansion the effective
//   batch dimension seen by the buffer is B × G.

#include <cstddef>
#include <optional>
#include <stdexcept>
#include <vector>
#include <torch/torch.h>

namespace dl
{
    // Per-step entry pushed by the training loop
    struct RolloutEntry
    {
        torch::Tensor prompt_ids;       // (B, prompt_len + gen_len)  — CPU or GPU
        torch::Tensor response_ids;// (B, gen_len)
        torch::Tensor old_log_probs;  // (B, gen_len)
        torch::Tensor ref_log_probs;   // (B, gen_len)
        torch::Tensor rewards;       // (B,)
        torch::Tensor advantages;    // (B, gen_len)
    };
    // Mini-batch returned during the update phase
    struct RolloutMiniBatch
    {
        torch::Tensor prompt_ids;
        torch::Tensor response_ids;
        torch::Tensor old_log_probs;
        torch::Tensor ref_log_probs;
        torch::Tensor rewards;
        torch::Tensor advantages;
    };
    // RolloutBuffer
    class RolloutBuffer
    {
    public:
        // minibatch_size  — number of samples per gradient-update step
        // normalize_adv   — zero-mean / unit-std normalise advantages
        //                   before iterating (standard PPO practice)
        explicit RolloutBuffer(int32_t minibatch_size = 8,
                               bool normalize_adv = true)
            : minibatch_size_(minibatch_size), normalize_adv_(normalize_adv), cursor_(0), sealed_(false)
        {
        }

        //  Write phase

        // Append one rollout batch.  All tensors must have the same
        // leading batch dimension.  Call as many times as needed.
        void push(RolloutEntry entry)
        {
            if (sealed_)
                throw std::runtime_error("RolloutBuffer: cannot push after seal()");
            entries_.push_back(std::move(entry));
        }

        // Concatenate all pushed entries into flat tensors and
        // optionally normalise advantages.  Must be called once
        // before iterating with next_minibatch().
        void seal()
        {
            if (entries_.empty())
                throw std::runtime_error("RolloutBuffer: no entries to seal");

            // Concatenate along dim 0
            auto cat = [&](auto member_fn)
            {
                std::vector<torch::Tensor> v;
                v.reserve(entries_.size());
                for (auto &e : entries_)
                    v.push_back(member_fn(e));
                return torch::cat(v, 0);
            };

            prompt_ids_ = cat([](const RolloutEntry &e)
                              { return e.prompt_ids; });
            response_ids_ = cat([](const RolloutEntry &e)
                                { return e.response_ids; });
            old_log_probs_ = cat([](const RolloutEntry &e)
                                 { return e.old_log_probs; });
            ref_log_probs_ = cat([](const RolloutEntry &e)
                                 { return e.ref_log_probs; });
            rewards_ = cat([](const RolloutEntry &e)
                           { return e.rewards; });
            advantages_ = cat([](const RolloutEntry &e)
                              { return e.advantages; });

            if (normalize_adv_)
            {
                auto mean = advantages_.mean();
                auto std = advantages_.std() + 1e-8f;
                advantages_ = (advantages_ - mean) / std;
            }

            total_samples_ = static_cast<int64_t>(prompt_ids_.size(0));
            cursor_ = 0;
            sealed_ = true;

            // Shuffle the flat index order once per seal
            perm_ = torch::randperm(total_samples_);
        }

        //  Read phase 

        // Returns the next mini-batch, or nullopt when one epoch
        // over the buffer is complete.  Call seal() before first use.
        std::optional<RolloutMiniBatch> next_minibatch()
        {
            if (!sealed_)
                throw std::runtime_error("RolloutBuffer: call seal() before iterating");
            if (cursor_ >= total_samples_)
                return std::nullopt;

            int64_t end = std::min(cursor_ + static_cast<int64_t>(minibatch_size_),
                                   total_samples_);
            auto idx = perm_.slice(0, cursor_, end);
            cursor_ = end;

            return RolloutMiniBatch{
                prompt_ids_.index_select(0, idx),
                response_ids_.index_select(0, idx),
                old_log_probs_.index_select(0, idx),
                ref_log_probs_.index_select(0, idx),
                rewards_.index_select(0, idx),
                advantages_.index_select(0, idx)};
        }

        // Reset the cursor to re-iterate the same sealed data
        // (for multiple PPO update epochs on the same rollout).
        void reset_cursor()
        {
            if (!sealed_)
                throw std::runtime_error("RolloutBuffer: call seal() before reset_cursor()");
            // Re-shuffle for each PPO epoch
            perm_ = torch::randperm(total_samples_);
            cursor_ = 0;
        }

        // Clear everything — call after all PPO update epochs are done.
        void clear()
        {
            entries_.clear();
            prompt_ids_ = torch::Tensor();
            response_ids_ = torch::Tensor();
            old_log_probs_ = torch::Tensor();
            ref_log_probs_ = torch::Tensor();
            rewards_ = torch::Tensor();
            advantages_ = torch::Tensor();
            perm_ = torch::Tensor();
            cursor_ = 0;
            total_samples_ = 0;
            sealed_ = false;
        }

        // ── Accessors ────────────────────────────────────────────

        bool is_sealed() const { return sealed_; }
        int64_t total_samples() const { return total_samples_; }
        int64_t cursor() const { return cursor_; }

        // Number of complete mini-batches per epoch
        int64_t num_minibatches() const
        {
            if (!sealed_ || total_samples_ == 0)
                return 0;
            return (total_samples_ + minibatch_size_ - 1) / minibatch_size_;
        }

    private:
        int32_t minibatch_size_;
        bool normalize_adv_;
        bool sealed_;
        int64_t cursor_ = 0;
        int64_t total_samples_ = 0;

        std::vector<RolloutEntry> entries_; // raw pushes (freed after seal)

        // Sealed / concatenated tensors
        torch::Tensor prompt_ids_;
        torch::Tensor response_ids_;
        torch::Tensor old_log_probs_;
        torch::Tensor ref_log_probs_;
        torch::Tensor rewards_;
        torch::Tensor advantages_;
        torch::Tensor perm_; // shuffled index permutation
    };

} 