// Tesla-2 custom GEMM — NOT cuBLAS.
// Build (when CUDA toolkit installed):
//   nvcc -O3 -arch=sm_75 --ptx crates/sentinel-kernel/cuda/gemm.cu -o gemm.ptx
//
// C = A @ B, row-major, float32
// A: [M,K], B: [K,N], C: [M,N]

extern "C" __global__ void tesla_gemm_f32(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;

    float acc = 0.f;
    for (int k = 0; k < K; ++k) {
        acc += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = acc;
}
