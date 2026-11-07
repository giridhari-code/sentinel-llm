# Tesla-2 CUDA kernels

Custom kernels only — **no cuBLAS / FlashAttention**.

| Artifact | Role |
|----------|------|
| `gemm.cu` | Source of truth for `tesla_gemm_f32` |
| `sdpa_scores.cu` | Causal mask helper |
| `src/gpu/ptx/tesla_gemm_f32.ptx` | Embedded PTX loaded by CUDA Driver API |

## Launch path

1. `dlopen(libcuda.so)`
2. `cuInit` → `cuCtxCreate` → `cuModuleLoadData(PTX)`
3. `cuMemAlloc` / `cuMemcpyHtoD`
4. `cuLaunchKernel(tesla_gemm_f32)`
5. `cuMemcpyDtoH`

If the driver is missing, Tesla automatically uses **HostMirror**.

## Rebuild PTX from source (optional)

```bash
nvcc -O3 -arch=sm_70 --ptx gemm.cu -o ../src/gpu/ptx/tesla_gemm_f32.ptx
```

## Smoke

```bash
cargo run -p sentinel-infer -- --device cuda gpu-smoke
```
