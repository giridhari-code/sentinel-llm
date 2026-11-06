//! Multi-head self-attention with real KV cache (Prefill + Decode).

use sentinel_core::tensor::Tensor;
use sentinel_core::{Result, ModelConfig, SentinelError};
use sentinel_kernel::{
    matmul, rms_norm, apply_rope, apply_rope_with_offset, scaled_dot_product_attention,
};
use crate::kv_cache::LayerKV;

pub struct MultiHeadAttention {
    pub wq: Tensor,
    pub wk: Tensor,
    pub wv: Tensor,
    pub wo: Tensor,
    pub norm_weight: Tensor,
    pub n_heads: usize,
    pub d_head: usize,
    pub d_model: usize,
    pub rope_theta: f32,
    pub norm_eps: f32,
}

impl MultiHeadAttention {
    pub fn new(config: &ModelConfig) -> Self {
        let d = config.d_model;
        let std = (1.0 / d as f32).sqrt();
        Self {
            wq: Tensor::randn(&[d, d], 0.0, std),
            wk: Tensor::randn(&[d, d], 0.0, std),
            wv: Tensor::randn(&[d, d], 0.0, std),
            wo: Tensor::randn(&[d, d], 0.0, std),
            norm_weight: Tensor::ones(&[d]),
            n_heads: config.n_heads,
            d_head: config.d_head,
            d_model: d,
            rope_theta: config.rope_theta,
            norm_eps: config.norm_eps,
        }
    }

    /// Project x [seq, d_model] → Q,K,V as [1, heads, seq, head_dim], apply RoPE with offset.
    fn project_qkv(
        &self,
        x: &Tensor,
        pos_offset: usize,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let seq = x.shape[0];
        let h = self.n_heads;
        let dh = self.d_head;

        let q = matmul(x, &self.wq)?;
        let k = matmul(x, &self.wk)?;
        let v = matmul(x, &self.wv)?;

        let q = q.reshape(&[1, h, seq, dh])?;
        let k = k.reshape(&[1, h, seq, dh])?;
        let v = v.reshape(&[1, h, seq, dh])?;

        let q = apply_rope_with_offset(&q, self.rope_theta, pos_offset)?;
        let k = apply_rope_with_offset(&k, self.rope_theta, pos_offset)?;

        Ok((q, k, v))
    }

    /// Convert [1, H, T, D] → [H, T, D] for cache storage.
    fn squeeze_batch(t: &Tensor) -> Result<Tensor> {
        // t shape [1, H, T, D]
        let h = t.shape[1];
        let seq = t.shape[2];
        let dh = t.shape[3];
        t.reshape(&[h, seq, dh])
    }

    /// Convert cached [H, T, D] → [1, H, T, D] for attention kernel.
    fn unsqueeze_batch(t: &Tensor) -> Result<Tensor> {
        let h = t.shape[0];
        let seq = t.shape[1];
        let dh = t.shape[2];
        t.reshape(&[1, h, seq, dh])
    }

    /// Full forward without cache (legacy / training path).
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (out, _kv) = self.forward_prefill(x)?;
        Ok(out)
    }

    /// **Prefill**: process full sequence, return output + K/V to seed the cache.
    /// x: [seq, d_model]
    /// returns (out [seq, d_model], LayerKV with K,V for this sequence)
    pub fn forward_prefill(&self, x: &Tensor) -> Result<(Tensor, LayerKV)> {
        let residual = x.clone();
        let x_n = rms_norm(x, &self.norm_weight, self.norm_eps)?;
        let seq = x_n.shape[0];
        let d = self.d_model;

        let (q, k, v) = self.project_qkv(&x_n, 0)?;
        let attn = scaled_dot_product_attention(&q, &k, &v, None)?;
        let attn = attn.reshape(&[seq, d])?;
        let out = matmul(&attn, &self.wo)?;
        let out = residual.add(&out)?;

        let layer_kv = LayerKV {
            keys: Self::squeeze_batch(&k)?,
            values: Self::squeeze_batch(&v)?,
        };
        Ok((out, layer_kv))
    }

    /// **Decode**: process only new tokens (usually seq=1), attend over cache + new K/V.
    /// x: [new_seq, d_model]
    /// cache: previous LayerKV
    /// returns (out [new_seq, d_model], updated LayerKV)
    pub fn forward_decode(&self, x: &Tensor, cache: &LayerKV) -> Result<(Tensor, LayerKV)> {
        let residual = x.clone();
        let x_n = rms_norm(x, &self.norm_weight, self.norm_eps)?;
        let new_seq = x_n.shape[0];
        let d = self.d_model;
        let pos_offset = cache.len();

        let (q, k_new, v_new) = self.project_qkv(&x_n, pos_offset)?;

        // Full K/V = cache || new
        let k_full_cached = Self::unsqueeze_batch(&cache.keys)?;
        let v_full_cached = Self::unsqueeze_batch(&cache.values)?;

        // k_new/v_new are [1,H,new_seq,D]; need concat on seq dim (dim=2)
        // Our Tensor::concat works on any dim.
        // Reshape path: squeeze to [H, T, D], concat dim=1, unsqueeze back.
        let k_new_s = Self::squeeze_batch(&k_new)?;
        let v_new_s = Self::squeeze_batch(&v_new)?;
        let updated = cache.append(&k_new_s, &v_new_s)?;

        let k_full = Self::unsqueeze_batch(&updated.keys)?;
        let v_full = Self::unsqueeze_batch(&updated.values)?;

        // Q is only the new positions: [1, H, new_seq, D]
        let attn = scaled_dot_product_attention(&q, &k_full, &v_full, None)?;
        let attn = attn.reshape(&[new_seq, d])?;
        let out = matmul(&attn, &self.wo)?;
        let out = residual.add(&out)?;

        // silence unused (kept for clarity of full path)
        let _ = (k_full_cached, v_full_cached);

        Ok((out, updated))
    }
}
