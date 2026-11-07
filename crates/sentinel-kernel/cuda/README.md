# Tesla-2 CUDA kernels

Custom kernels for Sentinel / Orbit. **No cuBLAS / cuDNN / FlashAttention.**

| File | Kernel |
|------|--------|
| `gemm.cu` | `tesla_gemm_f32` |
| `sdpa_scores.cu` | `tesla_causal_mask_f32` |

## Build PTX

```bash
nvcc -O3 -arch=sm_75 --ptx gemm.cu -o gemm.ptx
nvcc -O3 -arch=sm_75 --ptx sdpa_scores.cu -o sdpa_scores.ptx
```

## Rust

```bash
cargo build -p sentinel-kernel --features cuda
cargo run -p sentinel-infer -- --device cuda gpu-smoke
```

## Status

- HostMirror path: API complete, runs without GPU
- TeslaCuda: feature flag + kernel sources; driver launch next
