# Orbit · Sentinel

**Sentinel** LLM on the **Orbit** stack — **Rust**, custom **Tesla** kernel, **DoBu** data, **Bhutu** runtime.

> No third-party kernels. No CUDA/cuDNN. No FlashAttention. No HuggingFace.  
> Pure Rust. Fully owned. "Mere khuda ka kernel."

---

## Orbit names

| Name | Role |
|------|------|
| **Orbit** | Full platform / stack |
| **Sentinel** | The LLM (MoE + looped Transformer) |
| **Tesla** | Custom compute kernel (Tesla-1 CPU, Tesla-2 CUDA) |
| **DoBu** | Data + tokenizer pipeline |
| **Bhutu** | Inference runtime (prefill / decode / serve) |


## Architecture

| Component              | Description                                      |
|------------------------|--------------------------------------------------|
| **Model**              | Sentinel (Orbit model)                           |
| **Core Architecture**  | Mixture-of-Experts + Looped Transformer          |
| **Kernel**             | **Tesla-1** (custom pure-Rust)                   |
| **Attention**          | Multi-head + RoPE + SDPA (Tesla kernel)          |
| **FFN**                | SwiGLU-style Experts + optional Shared Expert    |
| **Looping**            | Same set of layers applied `n_loops` times       |
| **Normalization**      | RMSNorm                                          |
| **Inference**           | Prefill → Decode (KV cache)                      |

### Why Looped Transformer?
Instead of 32 unique layers, we keep e.g. 8 unique layers and loop them 4×.  
This gives high effective depth with far fewer parameters and better parameter efficiency.

### Why Custom Tesla Kernel?
- Full control & auditability
- No dependency on closed or heavy external kernels
- Clean path to later SIMD / custom GPU backends under the same API

---

## Project Structure

```
sentinel/
├── crates/
│   ├── sentinel-core/          # Tensor, Config, Error types
│   ├── sentinel-kernel/        # Tesla Kernel 1 (matmul, attention, rope, norm...)
│   ├── sentinel-moe/           # Router + Experts + MoE layer
│   ├── sentinel-transformer/   # Looped Transformer + SentinelModel
│   ├── sentinel-train/         # Training loop (skeleton)
│   └── sentinel-infer/         # CLI binary `sentinel`
├── configs/
├── models/
└── scripts/
```

---

## Quick Start

```bash
# From the workspace root
cargo build --release

# Show model architecture
cargo run -p sentinel-infer -- info

# Smoke test (forward pass)
cargo run -p sentinel-infer -- smoke --seq-len 32

# Generate (random weights — architecture demo only)
cargo run -p sentinel-infer -- generate --max-tokens 16
```

---

## Model Configs

| Name           | d_model | layers | loops | experts | ~params |
|----------------|---------|--------|-------|---------|---------|
| Sentinel-1B    | 1024    | 8      | 4     | 8       | ~1B*    |
| Sentinel-3B    | 2048    | 12     | 6     | 16      | ~3B*    |

\*Rough estimate. Actual count depends on tying and shared expert.

---

## Tesla Kernel Roadmap

- [x] Tesla-1 : Pure Rust + Rayon (CPU)
- [ ] Tesla-2 : SIMD (AVX2 / NEON) kernels
- [ ] Tesla-3 : Custom GPU backend (Vulkan / CUDA driver API)
- [ ] Tesla-4 : Quantized kernels (INT8 / FP8)

---


---

## Tokenizer

Sentinel ships with a flexible tokenizer layer (`sentinel-tokenizer`):

| Backend | When used | Notes |
|---------|-----------|-------|
| **HuggingFace tokenizers** | `--tokenizer path/to/tokenizer.json` | Full BPE / WordPiece / Unigram |
| **SimpleTokenizer** | Default (no file) | Character + whitespace, good for demos |

### Inference path: Prefill → Decode

There is **no model-level encode**. Flow is:

1. **Prefill** — full prompt in one shot → last logits + KV cache
2. **Decode** — one new token at a time (autoregressive)

```bash
# Prefill only
cargo run -p sentinel-infer -- prefill --prompt "Hello Sentinel"

# Full generation (prefill + decode loop)
cargo run -p sentinel-infer -- generate --prompt "The future of AI" --max-tokens 20

# Smoke: prefill + N decode steps
cargo run -p sentinel-infer -- smoke --prompt "Hello" --steps 4

# Tokenizer helpers (text ↔ ids only, not model encode)
cargo run -p sentinel-infer -- tok "Hello Sentinel" --bos
cargo run -p sentinel-infer -- detok 12 45 67 3

# Real HuggingFace tokenizer.json
cargo run -p sentinel-infer -- --tokenizer ./tokenizer.json generate --prompt "Hi"
```

## License

MIT OR Apache-2.0
