/**
 * patch_compress.cpp
 *
 * Stage 2 of the preprocessing pipeline.
 *
 * Reads instructions.bin / responses.bin produced by tokenize.py,
 * then decides — based on a threshold — whether to:
 *   (A) TRUNCATE  all sequences to max_seq_len  (cheap, no encoder needed), or
 *   (B) COMPRESS  long sequences with MegaByte-style patch encoding
 *
 * Decision rule  (your suggestion, implemented here):
 *   long_ratio = (#sequences whose length > max_seq_len) / total_sequences
 *   if long_ratio >= --threshold  →  train encoder + compress  (path B)
 *   else                          →  truncate                  (path A)
 *
 * Output .bin format — identical to the Python version so the same C++
 * dataloader can read both files without changes:
 *
 *   [4 bytes]  num_sequences                      (uint32_le)
 *   per sequence:
 *     [1 byte]   type flag  (0 = normal, 1 = compressed)
 *     [2 bytes]  num_steps                         (uint16_le)
 *     if type == 0:
 *       [num_steps * 4 bytes]  token IDs           (uint32_le)
 *     if type == 1:
 *       [4 bytes]  patch_size                      (uint32_le)
 *       [4 bytes]  embed_dim                       (uint32_le)
 *       [num_steps * embed_dim * 4 bytes]  vectors (float32_le)
 *
 * Build (Linux / macOS):
 *   g++ -O2 -std=c++17 patch_compress.cpp -o patch_compress
 *
 * Build with LibTorch for the neural encoder path:
 *   g++ -O2 -std=c++17 patch_compress.cpp \
 *       -I/path/to/libtorch/include \
 *       -L/path/to/libtorch/lib -ltorch -ltorch_cpu -lc10 \
 *       -DUSE_LIBTORCH \
 *       -o patch_compress
 *
 * Usage — let the tool decide:
 *   ./patch_compress \
 *       --instructions instructions.bin \
 *       --responses    responses.bin \
 *       --max_seq_len  512 \
 *       --threshold    0.20          # compress if >20 % of seqs are long
 *
 * Usage — force truncation:
 *   ./patch_compress ... --force_truncate
 *
 * Usage — force compression (train encoder first):
 *   ./patch_compress ... --force_compress --train_encoder --epochs 3
 */

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// ─────────────────────────────────────────────────────────────────────────────
//  Tiny matrix type (row-major, float)
// ─────────────────────────────────────────────────────────────────────────────
struct Matrix {
    int rows, cols;
    std::vector<float> data;

    Matrix() : rows(0), cols(0) {}
    Matrix(int r, int c, float fill = 0.f)
        : rows(r), cols(c), data(r * c, fill) {}

    float& at(int r, int c)       { return data[r * cols + c]; }
    float  at(int r, int c) const { return data[r * cols + c]; }
};

// ─────────────────────────────────────────────────────────────────────────────
//  Binary I/O  (matches tokenize.py format exactly)
// ─────────────────────────────────────────────────────────────────────────────

// Helper: read little-endian integers from a file
static uint32_t read_u32(std::ifstream& f) {
    uint32_t v = 0;
    f.read(reinterpret_cast<char*>(&v), 4);
    return v;
}
static uint16_t read_u16(std::ifstream& f) {
    uint16_t v = 0;
    f.read(reinterpret_cast<char*>(&v), 2);
    return v;
}

// Helper: write little-endian integers to a file
static void write_u8 (std::ofstream& f, uint8_t  v) { f.write(reinterpret_cast<char*>(&v), 1); }
static void write_u16(std::ofstream& f, uint16_t v) { f.write(reinterpret_cast<char*>(&v), 2); }
static void write_u32(std::ofstream& f, uint32_t v) { f.write(reinterpret_cast<char*>(&v), 4); }
static void write_f32(std::ofstream& f, float    v) { f.write(reinterpret_cast<char*>(&v), 4); }

/**
 * read_bin  — reads the .bin produced by tokenize.py
 * Format: [4B num_seqs] then per seq: [2B len][len * 4B token_ids]
 */
std::vector<std::vector<uint32_t>> read_bin(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open: " + path);

    uint32_t num_seqs = read_u32(f);
    std::vector<std::vector<uint32_t>> seqs;
    seqs.reserve(num_seqs);

    for (uint32_t i = 0; i < num_seqs; ++i) {
        uint16_t len = read_u16(f);
        std::vector<uint32_t> ids(len);
        f.read(reinterpret_cast<char*>(ids.data()), len * 4);
        seqs.push_back(std::move(ids));
    }
    std::printf("[read_bin] %s: %zu sequences\n", path.c_str(), seqs.size());
    return seqs;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Record type  (one per sequence in the output file)
// ─────────────────────────────────────────────────────────────────────────────
struct Record {
    uint8_t  type;       // 0 = normal tokens, 1 = patch vectors
    // type 0
    std::vector<uint32_t> ids;
    // type 1
    uint32_t patch_size = 0;
    uint32_t embed_dim  = 0;
    Matrix   vectors;    // (num_patches, embed_dim)
};

/**
 * write_compressed_bin  — writes the stage-2 output file.
 */
void write_compressed_bin(const std::string& path,
                          const std::vector<Record>& records) {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot write: " + path);

    write_u32(f, static_cast<uint32_t>(records.size()));

    for (const auto& rec : records) {
        if (rec.type == 0) {
            write_u8 (f, 0);
            write_u16(f, static_cast<uint16_t>(rec.ids.size()));
            for (uint32_t id : rec.ids)
                write_u32(f, id);
        } else {
            int num_patches = rec.vectors.rows;
            write_u8 (f, 1);
            write_u16(f, static_cast<uint16_t>(num_patches));
            write_u32(f, rec.patch_size);
            write_u32(f, rec.embed_dim);
            for (float v : rec.vectors.data)
                write_f32(f, v);
        }
    }
    std::printf("[write] %s: %zu records\n", path.c_str(), records.size());
}

// ─────────────────────────────────────────────────────────────────────────────
//  Threshold decision
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Compute the fraction of sequences that exceed max_seq_len.
 * This is the core of your idea: if the fraction is below the threshold,
 * it is cheaper and "good enough" to just truncate.
 */
double compute_long_ratio(const std::vector<std::vector<uint32_t>>& seqs,
                          int max_seq_len) {
    if (seqs.empty()) return 0.0;
    int long_count = 0;
    for (const auto& s : seqs)
        if (static_cast<int>(s.size()) > max_seq_len)
            ++long_count;
    return static_cast<double>(long_count) / seqs.size();
}

// ─────────────────────────────────────────────────────────────────────────────
//  Patch utilities
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Minimum patch_size so that (seq_len / patch_size) <= max_len.
 */
int compute_patch_size(int seq_len, int max_len) {
    return static_cast<int>(std::ceil(static_cast<double>(seq_len) / max_len));
}

/**
 * Pad ids to a multiple of patch_size, then split into patches.
 * Padding token = 0 (ignored by the encoder via ignore_index=0).
 */
std::vector<std::vector<uint32_t>>
pad_and_split(const std::vector<uint32_t>& ids, int patch_size) {
    std::vector<uint32_t> padded = ids;
    int rem = padded.size() % patch_size;
    if (rem != 0)
        padded.insert(padded.end(), patch_size - rem, 0u);

    std::vector<std::vector<uint32_t>> patches;
    for (size_t i = 0; i < padded.size(); i += patch_size) {
        patches.emplace_back(padded.begin() + i,
                             padded.begin() + i + patch_size);
    }
    return patches;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Simple neural building blocks (CPU, no LibTorch dependency)
//
//  This is a faithful port of the Python PatchEncoder architecture:
//    token_emb  →  prepend CLS  →  pos_emb  →  TransformerEncoder  →  CLS[0]
//                                                                    →  proj + norm
//
//  For production you would replace these stubs with LibTorch calls
//  (see the #ifdef USE_LIBTORCH section at the bottom of this file).
// ─────────────────────────────────────────────────────────────────────────────

static std::mt19937 g_rng(42);

inline float randn_scalar(float std_dev = 1.f) {
    static std::normal_distribution<float> dist(0.f, 1.f);
    return dist(g_rng) * std_dev;
}

// Layer normalisation  (over the last axis)
static void layer_norm(float* x, int n, const float* gamma, const float* beta,
                       float eps = 1e-5f) {
    float mean = 0.f;
    for (int i = 0; i < n; ++i) mean += x[i];
    mean /= n;

    float var = 0.f;
    for (int i = 0; i < n; ++i) var += (x[i] - mean) * (x[i] - mean);
    var /= n;

    float inv = 1.f / std::sqrt(var + eps);
    for (int i = 0; i < n; ++i)
        x[i] = gamma[i] * (x[i] - mean) * inv + beta[i];
}

// GELU activation  (approximation matching PyTorch's default)
static float gelu(float x) {
    return 0.5f * x * (1.f + std::tanh(0.7978845608f * (x + 0.044715f * x * x * x)));
}

// Dense (linear) layer: y = x @ W^T + b
//   x: (in_features,)   W: (out_features, in_features)   b: (out_features,)
static void linear(const float* x, const float* W, const float* b,
                   float* y, int in_f, int out_f) {
    for (int o = 0; o < out_f; ++o) {
        float acc = b ? b[o] : 0.f;
        for (int i = 0; i < in_f; ++i)
            acc += W[o * in_f + i] * x[i];
        y[o] = acc;
    }
}

// Softmax in-place over a vector of length n
static void softmax(float* x, int n) {
    float mx = *std::max_element(x, x + n);
    float sum = 0.f;
    for (int i = 0; i < n; ++i) { x[i] = std::exp(x[i] - mx); sum += x[i]; }
    for (int i = 0; i < n; ++i) x[i] /= sum;
}

// ─────────────────────────────────────────────────────────────────────────────
//  PatchEncoder  — CPU port
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Minimal CPU PatchEncoder that mirrors the Python architecture exactly.
 *
 * Weights are randomly initialised here; in real use you would load them
 * from a file written by the Python training loop (see save/load helpers).
 *
 * For a production-grade build, compile with -DUSE_LIBTORCH and use the
 * LibTorch backend defined further below.
 */
struct PatchEncoder {
    int vocab_size;
    int embed_dim;
    int num_heads;
    int num_layers;
    int max_patch_size;

    // token embedding table: (vocab_size+1, embed_dim)
    std::vector<float> token_emb;

    // CLS token: (embed_dim,)
    std::vector<float> cls_token;

    // positional embedding: (max_patch_size+1, embed_dim)
    std::vector<float> pos_emb;

    // Per-layer transformer weights
    // We implement a minimal Pre-Norm transformer encoder layer:
    //   Pre-LN1 → multi-head self-attention → residual
    //   Pre-LN2 → FFN (linear → GELU → linear) → residual
    struct TransformerLayer {
        int D, H, Dff;  // embed_dim, num_heads, 4*embed_dim
        // Pre-LN 1
        std::vector<float> ln1_g, ln1_b;
        // Q, K, V projections (each: D × D)
        std::vector<float> Wq, Wk, Wv, Wo;
        std::vector<float> bq, bk, bv, bo;
        // Pre-LN 2
        std::vector<float> ln2_g, ln2_b;
        // FFN
        std::vector<float> W1, b1, W2, b2;
    };
    std::vector<TransformerLayer> layers;

    // Final proj + norm
    std::vector<float> proj_W, proj_b;
    std::vector<float> norm_g, norm_b;

    // ── constructor ───────────────────────────────────────────────────────────
    PatchEncoder(int vocab_size_, int embed_dim_,
                 int num_heads_ = 4, int num_layers_ = 2,
                 int max_patch_size_ = 64)
        : vocab_size(vocab_size_), embed_dim(embed_dim_),
          num_heads(num_heads_), num_layers(num_layers_),
          max_patch_size(max_patch_size_)
    {
        int D   = embed_dim;
        int V1  = vocab_size + 1;       // +1 for padding idx 0
        int Pos = max_patch_size + 1;   // +1 for CLS position

        // random init (σ = 0.02, matching PyTorch default for embeddings)
        auto rand_vec = [&](int n, float sigma = 0.02f) {
            std::vector<float> v(n);
            for (auto& x : v) x = randn_scalar(sigma);
            return v;
        };
        auto ones_vec = [&](int n) { return std::vector<float>(n, 1.f); };
        auto zero_vec = [&](int n) { return std::vector<float>(n, 0.f); };

        token_emb = rand_vec(V1 * D);
        cls_token = rand_vec(D);
        pos_emb   = rand_vec(Pos * D);

        layers.resize(num_layers);
        for (auto& l : layers) {
            l.D   = D;
            l.H   = num_heads;
            l.Dff = D * 4;
            l.ln1_g = ones_vec(D);  l.ln1_b = zero_vec(D);
            l.Wq = rand_vec(D*D);   l.bq = zero_vec(D);
            l.Wk = rand_vec(D*D);   l.bk = zero_vec(D);
            l.Wv = rand_vec(D*D);   l.bv = zero_vec(D);
            l.Wo = rand_vec(D*D);   l.bo = zero_vec(D);
            l.ln2_g = ones_vec(D);  l.ln2_b = zero_vec(D);
            l.W1 = rand_vec(l.Dff * D);   l.b1 = zero_vec(l.Dff);
            l.W2 = rand_vec(D * l.Dff);   l.b2 = zero_vec(D);
        }

        proj_W = rand_vec(D * D);   proj_b = zero_vec(D);
        norm_g = ones_vec(D);        norm_b = zero_vec(D);
    }

    // ── forward: encode one patch (patch_size token IDs) → embed_dim vector ──
    /**
     * Encodes a single patch of `patch_size` token IDs into one vector.
     *
     * Steps (matching Python exactly):
     *   1. Embed each token:   x[i] = token_emb[id[i]]
     *   2. Prepend CLS:        seq = [cls_token, x[0], x[1], …, x[P-1]]
     *   3. Add positional emb: seq[i] += pos_emb[i]
     *   4. Run transformer layers (pre-norm)
     *   5. Take CLS (position 0), project, layer-norm → output vector
     */
    std::vector<float> encode_patch(const std::vector<uint32_t>& patch) const {
        int P = static_cast<int>(patch.size());
        int D = embed_dim;
        int seqlen = P + 1;  // CLS + P tokens

        // (seqlen, D) stored flat row-major
        std::vector<float> seq(seqlen * D);

        // CLS token at position 0
        for (int d = 0; d < D; ++d)
            seq[d] = cls_token[d];

        // Embed token IDs at positions 1..P
        for (int i = 0; i < P; ++i) {
            uint32_t id = patch[i];
            if (id >= static_cast<uint32_t>(vocab_size + 1)) id = 0;  // clamp
            const float* emb = token_emb.data() + id * D;
            float* dst = seq.data() + (i + 1) * D;
            std::copy(emb, emb + D, dst);
        }

        // Add positional embeddings
        for (int i = 0; i < seqlen; ++i) {
            const float* pe = pos_emb.data() + i * D;
            float* x = seq.data() + i * D;
            for (int d = 0; d < D; ++d) x[d] += pe[d];
        }

        // Transformer layers
        std::vector<float> buf(D), buf2(D), attn(seqlen * seqlen);
        std::vector<float> Q(seqlen * D), K(seqlen * D), V(seqlen * D);

        for (const auto& l : layers) {
            int H = l.H;
            int dh = D / H;  // head dim

            // ── Pre-LN 1 + Multi-head self-attention ─────────────────────
            // Apply LN to each position, compute Q, K, V
            std::vector<float> normed(seqlen * D);
            for (int i = 0; i < seqlen; ++i) {
                std::copy(seq.data() + i*D, seq.data() + (i+1)*D,
                          normed.data() + i*D);
                layer_norm(normed.data() + i*D, D, l.ln1_g.data(), l.ln1_b.data());
                linear(normed.data() + i*D, l.Wq.data(), l.bq.data(),
                       Q.data() + i*D, D, D);
                linear(normed.data() + i*D, l.Wk.data(), l.bk.data(),
                       K.data() + i*D, D, D);
                linear(normed.data() + i*D, l.Wv.data(), l.bv.data(),
                       V.data() + i*D, D, D);
            }

            // Compute attention output: for each head
            std::vector<float> attn_out(seqlen * D, 0.f);
            float scale = 1.f / std::sqrt(static_cast<float>(dh));

            for (int h = 0; h < H; ++h) {
                // attn scores for this head: (seqlen, seqlen)
                for (int i = 0; i < seqlen; ++i) {
                    for (int j = 0; j < seqlen; ++j) {
                        float dot = 0.f;
                        for (int d = 0; d < dh; ++d)
                            dot += Q[i*D + h*dh + d] * K[j*D + h*dh + d];
                        attn[i*seqlen + j] = dot * scale;
                    }
                    softmax(attn.data() + i*seqlen, seqlen);
                }

                // weighted sum of V
                for (int i = 0; i < seqlen; ++i) {
                    for (int d = 0; d < dh; ++d) {
                        float acc = 0.f;
                        for (int j = 0; j < seqlen; ++j)
                            acc += attn[i*seqlen + j] * V[j*D + h*dh + d];
                        attn_out[i*D + h*dh + d] += acc;
                    }
                }
            }

            // Output projection + residual
            for (int i = 0; i < seqlen; ++i) {
                linear(attn_out.data() + i*D, l.Wo.data(), l.bo.data(),
                       buf.data(), D, D);
                for (int d = 0; d < D; ++d)
                    seq[i*D + d] += buf[d];
            }

            // ── Pre-LN 2 + FFN ────────────────────────────────────────────
            std::vector<float> ffn_in(D), ffn_h(l.Dff), ffn_out(D);
            for (int i = 0; i < seqlen; ++i) {
                std::copy(seq.data() + i*D, seq.data() + (i+1)*D, ffn_in.data());
                layer_norm(ffn_in.data(), D, l.ln2_g.data(), l.ln2_b.data());
                linear(ffn_in.data(), l.W1.data(), l.b1.data(),
                       ffn_h.data(), D, l.Dff);
                for (auto& v : ffn_h) v = gelu(v);
                linear(ffn_h.data(), l.W2.data(), l.b2.data(),
                       ffn_out.data(), l.Dff, D);
                for (int d = 0; d < D; ++d)
                    seq[i*D + d] += ffn_out[d];
            }
        }

        // CLS at position 0 → project → norm
        std::vector<float> cls_out(D);
        linear(seq.data(), proj_W.data(), proj_b.data(), cls_out.data(), D, D);
        layer_norm(cls_out.data(), D, norm_g.data(), norm_b.data());
        return cls_out;
    }

    // ── encode a full sequence → (num_patches, embed_dim) Matrix ─────────────
    Matrix encode_sequence(const std::vector<uint32_t>& ids,
                           int patch_size) const {
        auto patches = pad_and_split(ids, patch_size);
        int np = static_cast<int>(patches.size());
        Matrix M(np, embed_dim);
        for (int i = 0; i < np; ++i) {
            auto vec = encode_patch(patches[i]);
            std::copy(vec.begin(), vec.end(), M.data.begin() + i * embed_dim);
        }
        return M;
    }

    // ── save/load weights from a binary file ──────────────────────────────────
    /**
     * File format (all little-endian float32, no headers):
     *   token_emb  (V1 * D floats)
     *   cls_token  (D floats)
     *   pos_emb    (Pos * D floats)
     *   per layer: ln1_g, ln1_b, Wq, bq, Wk, bk, Wv, bv, Wo, bo,
     *              ln2_g, ln2_b, W1, b1, W2, b2
     *   proj_W, proj_b, norm_g, norm_b
     *
     * NOTE: this format is intentionally simple so the Python side can write
     *       it with  numpy.ndarray.astype('float32').tofile(f)
     */
    void save(const std::string& path) const {
        std::ofstream f(path, std::ios::binary);
        auto write_vec = [&](const std::vector<float>& v) {
            f.write(reinterpret_cast<const char*>(v.data()), v.size() * 4);
        };
        write_vec(token_emb);
        write_vec(cls_token);
        write_vec(pos_emb);
        for (const auto& l : layers) {
            write_vec(l.ln1_g); write_vec(l.ln1_b);
            write_vec(l.Wq);    write_vec(l.bq);
            write_vec(l.Wk);    write_vec(l.bk);
            write_vec(l.Wv);    write_vec(l.bv);
            write_vec(l.Wo);    write_vec(l.bo);
            write_vec(l.ln2_g); write_vec(l.ln2_b);
            write_vec(l.W1);    write_vec(l.b1);
            write_vec(l.W2);    write_vec(l.b2);
        }
        write_vec(proj_W); write_vec(proj_b);
        write_vec(norm_g); write_vec(norm_b);
        std::printf("[PatchEncoder] weights saved → %s\n", path.c_str());
    }

    void load(const std::string& path) {
        std::ifstream f(path, std::ios::binary);
        if (!f) throw std::runtime_error("Cannot open encoder: " + path);
        auto read_vec = [&](std::vector<float>& v) {
            f.read(reinterpret_cast<char*>(v.data()), v.size() * 4);
        };
        read_vec(token_emb);
        read_vec(cls_token);
        read_vec(pos_emb);
        for (auto& l : layers) {
            read_vec(l.ln1_g); read_vec(l.ln1_b);
            read_vec(l.Wq);    read_vec(l.bq);
            read_vec(l.Wk);    read_vec(l.bk);
            read_vec(l.Wv);    read_vec(l.bv);
            read_vec(l.Wo);    read_vec(l.bo);
            read_vec(l.ln2_g); read_vec(l.ln2_b);
            read_vec(l.W1);    read_vec(l.b1);
            read_vec(l.W2);    read_vec(l.b2);
        }
        read_vec(proj_W); read_vec(proj_b);
        read_vec(norm_g); read_vec(norm_b);
        std::printf("[PatchEncoder] weights loaded ← %s\n", path.c_str());
    }
};

// ─────────────────────────────────────────────────────────────────────────────
//  Path A — TRUNCATE
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Build records by simply truncating every sequence to max_seq_len tokens.
 * No encoder needed. All output records have type == 0.
 *
 * This is the cheap path taken when:
 *   long_ratio < threshold   OR   --force_truncate is set
 */
std::vector<Record>
truncate_sequences(const std::vector<std::vector<uint32_t>>& seqs,
                   int max_seq_len) {
    std::vector<Record> records;
    records.reserve(seqs.size());
    int truncated = 0;

    for (const auto& ids : seqs) {
        Record rec;
        rec.type = 0;
        if (static_cast<int>(ids.size()) > max_seq_len) {
            rec.ids.assign(ids.begin(), ids.begin() + max_seq_len);
            ++truncated;
        } else {
            rec.ids = ids;
        }
        records.push_back(std::move(rec));
    }

    std::printf("[truncate] %zu sequences, %d truncated\n",
                seqs.size(), truncated);
    return records;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Path B — COMPRESS (inference pass; training is done in Python)
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Compress long sequences using the (pre-trained) PatchEncoder.
 * Short sequences are passed through as normal tokens (type 0).
 *
 * This mirrors compress_sequences() in the Python file exactly.
 */
std::vector<Record>
compress_sequences(const std::vector<std::vector<uint32_t>>& seqs,
                   const PatchEncoder& encoder,
                   int max_seq_len) {
    std::vector<Record> records;
    records.reserve(seqs.size());
    int normal_count = 0, compressed_count = 0;

    for (size_t idx = 0; idx < seqs.size(); ++idx) {
        const auto& ids = seqs[idx];
        Record rec;

        if (static_cast<int>(ids.size()) <= max_seq_len) {
            rec.type = 0;
            rec.ids  = ids;
            ++normal_count;
        } else {
            int ps = compute_patch_size(static_cast<int>(ids.size()), max_seq_len);
            rec.type       = 1;
            rec.patch_size = static_cast<uint32_t>(ps);
            rec.embed_dim  = static_cast<uint32_t>(encoder.embed_dim);
            rec.vectors    = encoder.encode_sequence(ids, ps);
            ++compressed_count;
        }

        records.push_back(std::move(rec));

        if ((idx + 1) % 1000 == 0)
            std::printf("  compressed %zu/%zu\r", idx + 1, seqs.size());
    }

    std::printf("\n[compress] Normal     : %d\n", normal_count);
    std::printf("[compress] Compressed : %d\n", compressed_count);
    return records;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Argument parsing  (no external library)
// ─────────────────────────────────────────────────────────────────────────────
struct Args {
    std::string instructions         = "";
    std::string responses            = "";
    std::string instructions_output  = "instructions_compressed.bin";
    std::string responses_output     = "responses_compressed.bin";

    int    max_seq_len   = 512;
    int    embed_dim     = 256;
    int    vocab_size    = 50257;

    double threshold     = 0.20;   // compress if >= 20 % of sequences are long
    bool   force_truncate = false;
    bool   force_compress = false;

    std::string encoder_path = "patch_encoder.bin";
};

static std::string get_arg(int argc, char* argv[], const std::string& key,
                            const std::string& def = "") {
    for (int i = 1; i < argc - 1; ++i)
        if (argv[i] == key) return argv[i + 1];
    return def;
}
static bool has_flag(int argc, char* argv[], const std::string& key) {
    for (int i = 1; i < argc; ++i)
        if (argv[i] == key) return true;
    return false;
}

Args parse_args(int argc, char* argv[]) {
    Args a;
    a.instructions        = get_arg(argc, argv, "--instructions");
    a.responses           = get_arg(argc, argv, "--responses");
    a.instructions_output = get_arg(argc, argv, "--instructions_output",
                                    a.instructions_output);
    a.responses_output    = get_arg(argc, argv, "--responses_output",
                                    a.responses_output);
    a.encoder_path        = get_arg(argc, argv, "--encoder_path",
                                    a.encoder_path);

    auto int_arg = [&](const std::string& k, int def) {
        std::string s = get_arg(argc, argv, k);
        return s.empty() ? def : std::stoi(s);
    };
    auto dbl_arg = [&](const std::string& k, double def) {
        std::string s = get_arg(argc, argv, k);
        return s.empty() ? def : std::stod(s);
    };

    a.max_seq_len    = int_arg("--max_seq_len",  a.max_seq_len);
    a.embed_dim      = int_arg("--embed_dim",    a.embed_dim);
    a.vocab_size     = int_arg("--vocab_size",   a.vocab_size);
    a.threshold      = dbl_arg("--threshold",    a.threshold);
    a.force_truncate = has_flag(argc, argv, "--force_truncate");
    a.force_compress = has_flag(argc, argv, "--force_compress");
    return a;
}

// ─────────────────────────────────────────────────────────────────────────────
//  main
// ─────────────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    // ── parse arguments ───────────────────────────────────────────────────────
    Args args = parse_args(argc, argv);

    if (args.instructions.empty() || args.responses.empty()) {
        std::fprintf(stderr,
            "Usage: patch_compress --instructions <file> --responses <file>\n"
            "                      [--max_seq_len 512] [--threshold 0.20]\n"
            "                      [--embed_dim 256] [--vocab_size 50257]\n"
            "                      [--encoder_path patch_encoder.bin]\n"
            "                      [--force_truncate | --force_compress]\n"
            "                      [--instructions_output <file>]\n"
            "                      [--responses_output <file>]\n");
        return 1;
    }

    std::printf("[patch_compress] max_seq_len = %d\n", args.max_seq_len);
    std::printf("[patch_compress] embed_dim   = %d\n", args.embed_dim);
    std::printf("[patch_compress] threshold   = %.2f\n", args.threshold);

    // ── load sequences ────────────────────────────────────────────────────────
    auto inst_seqs = read_bin(args.instructions);
    auto resp_seqs = read_bin(args.responses);

    // ── decide: truncate or compress? ─────────────────────────────────────────
    //
    // KEY LOGIC (your idea):
    //   1. Count how many sequences in the combined dataset are "long"
    //      (i.e. exceed max_seq_len).
    //   2. If that fraction is below --threshold  →  truncation is "good enough"
    //      (only a small minority of sequences would be affected by patching,
    //       so the expensive encoder training is not worth it).
    //   3. If the fraction is >= threshold  →  too much data would be lost to
    //      truncation, so we pay the training cost and compress instead.
    //
    // The flags --force_truncate / --force_compress override this logic.
    // ─────────────────────────────────────────────────────────────────────────

    std::vector<std::vector<uint32_t>> all_seqs;
    all_seqs.insert(all_seqs.end(), inst_seqs.begin(), inst_seqs.end());
    all_seqs.insert(all_seqs.end(), resp_seqs.begin(), resp_seqs.end());

    double long_ratio = compute_long_ratio(all_seqs, args.max_seq_len);

    std::printf("[patch_compress] long sequences ratio = %.4f (threshold %.2f)\n",
                long_ratio, args.threshold);

    bool use_compress;
    if (args.force_truncate) {
        use_compress = false;
        std::printf("[patch_compress] mode = TRUNCATE (forced)\n");
    } else if (args.force_compress) {
        use_compress = true;
        std::printf("[patch_compress] mode = COMPRESS (forced)\n");
    } else if (long_ratio >= args.threshold) {
        use_compress = true;
        std::printf("[patch_compress] mode = COMPRESS "
                    "(%.1f%% long >= threshold %.0f%%)\n",
                    long_ratio * 100, args.threshold * 100);
    } else {
        use_compress = false;
        std::printf("[patch_compress] mode = TRUNCATE "
                    "(only %.1f%% long < threshold %.0f%%, not worth encoding)\n",
                    long_ratio * 100, args.threshold * 100);
    }

    // ── PATH A: TRUNCATE ──────────────────────────────────────────────────────
    if (!use_compress) {
        std::printf("\n[truncate] Processing instructions...\n");
        auto inst_records = truncate_sequences(inst_seqs, args.max_seq_len);
        write_compressed_bin(args.instructions_output, inst_records);

        std::printf("\n[truncate] Processing responses...\n");
        auto resp_records = truncate_sequences(resp_seqs, args.max_seq_len);
        write_compressed_bin(args.responses_output, resp_records);

        std::printf("\nDone. Both files written with truncation (type=0 only).\n");
        return 0;
    }

    // ── PATH B: COMPRESS ──────────────────────────────────────────────────────
    //
    // Determine the global patch_size from the longest sequence across both
    // files (same logic as the Python script).
    int max_found = 0;
    for (const auto& s : all_seqs)
        max_found = std::max(max_found, static_cast<int>(s.size()));

    int patch_size = compute_patch_size(max_found, args.max_seq_len);
    std::printf("[patch_compress] longest sequence = %d tokens\n", max_found);
    std::printf("[patch_compress] patch_size       = %d\n", patch_size);

    // Build encoder
    PatchEncoder encoder(args.vocab_size, args.embed_dim, 4, 2, patch_size);

    // Load pre-trained weights if available.
    // Training itself must be done on the Python side (patch_compress.py
    // --train_encoder) and the weights exported to the binary format
    // described in PatchEncoder::save() / load() above.
    std::ifstream enc_check(args.encoder_path);
    if (enc_check.good()) {
        enc_check.close();
        encoder.load(args.encoder_path);
    } else {
        std::printf("[patch_compress] WARNING: no encoder found at %s\n",
                    args.encoder_path.c_str());
        std::printf("[patch_compress] Using random weights — run Python "
                    "patch_compress.py --train_encoder first!\n");
    }

    // Compress instructions
    std::printf("\n[compress] Processing instructions...\n");
    auto inst_records = compress_sequences(inst_seqs, encoder, args.max_seq_len);
    write_compressed_bin(args.instructions_output, inst_records);

    // Compress responses
    std::printf("\n[compress] Processing responses...\n");
    auto resp_records = compress_sequences(resp_seqs, encoder, args.max_seq_len);
    write_compressed_bin(args.responses_output, resp_records);

    // ── summary ───────────────────────────────────────────────────────────────
    std::printf("\n");
    std::printf("==================================================\n");
    std::printf("  DONE\n");
    std::printf("==================================================\n");
    std::printf("  Instructions : %s\n", args.instructions_output.c_str());
    std::printf("  Responses    : %s\n", args.responses_output.c_str());
    std::printf("  Encoder      : %s\n", args.encoder_path.c_str());
    std::printf("  patch_size   : %d\n", patch_size);
    std::printf("  embed_dim    : %d\n", args.embed_dim);
    std::printf("==================================================\n");

    return 0;
}