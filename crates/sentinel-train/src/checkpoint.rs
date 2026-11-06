//! Model checkpoint save / load (custom binary format, no safetensors dep).

use sentinel_core::config::ModelConfig;
use sentinel_core::tensor::Tensor;
use sentinel_core::{Result, SentinelError};
use sentinel_transformer::SentinelModel;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

const MAGIC: &[u8; 8] = b"SENTCKP1";

#[derive(Serialize, Deserialize)]
struct CkptHeader {
    config: ModelConfig,
    tensors: Vec<TensorMeta>,
}

#[derive(Serialize, Deserialize)]
struct TensorMeta {
    name: String,
    shape: Vec<usize>,
    numel: usize,
}

fn collect_tensors(model: &SentinelModel) -> Vec<(String, &Tensor)> {
    let mut out = Vec::new();
    out.push(("embed.weight".into(), &model.embed.weight));
    out.push(("lm_head".into(), &model.lm_head));
    out.push(("final_norm_weight".into(), &model.final_norm_weight));
    for (li, layer) in model.layers.iter().enumerate() {
        let p = format!("layers.{}", li);
        out.push((format!("{}.attn.wq", p), &layer.attn.wq));
        out.push((format!("{}.attn.wk", p), &layer.attn.wk));
        out.push((format!("{}.attn.wv", p), &layer.attn.wv));
        out.push((format!("{}.attn.wo", p), &layer.attn.wo));
        out.push((format!("{}.attn.norm_weight", p), &layer.attn.norm_weight));
        out.push((format!("{}.moe.norm_weight", p), &layer.moe.norm_weight));
        out.push((format!("{}.moe.router.weight", p), &layer.moe.router.weight));
        for (ei, expert) in layer.moe.experts.iter().enumerate() {
            out.push((format!("{}.moe.experts.{}.gate", p, ei), &expert.gate));
            out.push((format!("{}.moe.experts.{}.up", p, ei), &expert.up));
            out.push((format!("{}.moe.experts.{}.down", p, ei), &expert.down));
        }
        if let Some(ref shared) = layer.moe.shared_expert {
            out.push((format!("{}.moe.shared.gate", p), &shared.gate));
            out.push((format!("{}.moe.shared.up", p), &shared.up));
            out.push((format!("{}.moe.shared.down", p), &shared.down));
        }
    }
    out
}

fn set_tensor(model: &mut SentinelModel, name: &str, data: Vec<f32>, shape: &[usize]) -> Result<()> {
    let t = Tensor::from_vec(data, shape)?;
    // Match by name — assign into model
    if name == "embed.weight" {
        model.embed.weight = t;
        return Ok(());
    }
    if name == "lm_head" {
        model.lm_head = t;
        return Ok(());
    }
    if name == "final_norm_weight" {
        model.final_norm_weight = t;
        return Ok(());
    }
    for (li, layer) in model.layers.iter_mut().enumerate() {
        let p = format!("layers.{}", li);
        if name == format!("{}.attn.wq", p) { layer.attn.wq = t; return Ok(()); }
        if name == format!("{}.attn.wk", p) { layer.attn.wk = t; return Ok(()); }
        if name == format!("{}.attn.wv", p) { layer.attn.wv = t; return Ok(()); }
        if name == format!("{}.attn.wo", p) { layer.attn.wo = t; return Ok(()); }
        if name == format!("{}.attn.norm_weight", p) { layer.attn.norm_weight = t; return Ok(()); }
        if name == format!("{}.moe.norm_weight", p) { layer.moe.norm_weight = t; return Ok(()); }
        if name == format!("{}.moe.router.weight", p) { layer.moe.router.weight = t; return Ok(()); }
        for (ei, expert) in layer.moe.experts.iter_mut().enumerate() {
            if name == format!("{}.moe.experts.{}.gate", p, ei) { expert.gate = t; return Ok(()); }
            if name == format!("{}.moe.experts.{}.up", p, ei) { expert.up = t; return Ok(()); }
            if name == format!("{}.moe.experts.{}.down", p, ei) { expert.down = t; return Ok(()); }
        }
        if let Some(ref mut shared) = layer.moe.shared_expert {
            if name == format!("{}.moe.shared.gate", p) { shared.gate = t; return Ok(()); }
            if name == format!("{}.moe.shared.up", p) { shared.up = t; return Ok(()); }
            if name == format!("{}.moe.shared.down", p) { shared.down = t; return Ok(()); }
        }
    }
    Err(SentinelError::Other(format!("unknown tensor in ckpt: {}", name)))
}

pub fn save_checkpoint(model: &SentinelModel, path: &Path) -> Result<()> {
    let tensors = collect_tensors(model);
    let header = CkptHeader {
        config: model.config.clone(),
        tensors: tensors
            .iter()
            .map(|(n, t)| TensorMeta {
                name: n.clone(),
                shape: t.shape.clone(),
                numel: t.numel(),
            })
            .collect(),
    };
    let header_bytes = serde_json::to_vec(&header)?;
    let mut f = File::create(path)?;
    f.write_all(MAGIC)?;
    f.write_all(&(header_bytes.len() as u64).to_le_bytes())?;
    f.write_all(&header_bytes)?;
    for (_, t) in &tensors {
        for &x in &t.data {
            f.write_all(&x.to_le_bytes())?;
        }
    }
    log::info!("Saved checkpoint → {} ({} tensors)", path.display(), tensors.len());
    Ok(())
}

pub fn load_checkpoint(path: &Path) -> Result<SentinelModel> {
    let mut f = File::open(path)?;
    let mut magic = [0u8; 8];
    f.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(SentinelError::Other("invalid checkpoint magic".into()));
    }
    let mut len_buf = [0u8; 8];
    f.read_exact(&mut len_buf)?;
    let hlen = u64::from_le_bytes(len_buf) as usize;
    let mut header_bytes = vec![0u8; hlen];
    f.read_exact(&mut header_bytes)?;
    let header: CkptHeader = serde_json::from_slice(&header_bytes)?;

    let mut model = SentinelModel::new(header.config);
    for meta in &header.tensors {
        let mut data = vec![0.0f32; meta.numel];
        for x in data.iter_mut() {
            let mut b = [0u8; 4];
            f.read_exact(&mut b)?;
            *x = f32::from_le_bytes(b);
        }
        set_tensor(&mut model, &meta.name, data, &meta.shape)?;
    }
    log::info!("Loaded checkpoint ← {} ({} tensors)", path.display(), header.tensors.len());
    Ok(model)
}
