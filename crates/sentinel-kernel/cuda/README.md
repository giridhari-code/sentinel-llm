# Tesla-2 CUDA kernels

Custom kernels for Sentinel. **No cuBLAS / cuDNN / FlashAttention.**

| File | Kernel |
|------|--------|
| `gemm.cu` | `tesla_gemm_f32` — FP32 matmul |

## Build PTX

```bash
nvcc -O3 -arch=sm_75 --ptx gemm.cu -o gemm.ptx
```

## Enable in Rust

```bash
cargo build -p sentinel-kernel --features cuda
cargo run -p sentinel-infer -- features... --device cuda
```

## Status

- Tesla-2.0: host-mirror + kernel source landed
- Next: CUDA driver load PTX, VRAM buffers, SDPA on GPU
