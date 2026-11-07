// Tesla-2: causal score fill helper (custom, not FlashAttention)
// scores[i,j] = -inf if j > i + offset else scores[i,j]

extern "C" __global__ void tesla_causal_mask_f32(
    float* scores, int Tq, int Tk, int pos_offset
) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= Tq || j >= Tk) return;
    int q_pos = pos_offset + i;
    if (j > q_pos) {
        scores[i * Tk + j] = -1.0e30f;
    }
}
