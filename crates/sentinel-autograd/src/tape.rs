//! Computation tape and variable handles.

use sentinel_core::tensor::Tensor;
use sentinel_core::types::Float;
use sentinel_core::{Result, SentinelError};
use std::collections::HashMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Var(pub(crate) usize);

#[derive(Clone, Debug)]
pub(crate) enum Op {
    /// Leaf parameter or constant input
    Leaf { is_param: bool },
    Matmul { a: Var, b: Var },
    Add { a: Var, b: Var },
    Mul { a: Var, b: Var },
    Scale { x: Var, factor: Float },
    Silu { x: Var },
    /// RMSNorm: y = x / rms * weight (weight is Var)
    RmsNorm { x: Var, weight: Var, eps: Float },
    /// Embedding gather: weight [vocab, d], indices stored separately
    Embed { weight: Var, indices: Vec<u32> },
    /// Softmax CE loss (scalar). logits [seq,vocab], targets next tokens
    SoftmaxCe { logits: Var, targets: Vec<u32> },
    Reshape { x: Var, shape: Vec<usize> },
}

struct Node {
    value: Tensor,
    grad: Option<Tensor>,
    op: Op,
}

pub struct Tape {
    nodes: Vec<Node>,
}

/// Map from parameter Var → gradient tensor
pub type GradMap = HashMap<Var, Tensor>;

impl Tape {
    pub fn new() -> Self {
        Self { nodes: Vec::new() }
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    fn push(&mut self, value: Tensor, op: Op) -> Var {
        let id = self.nodes.len();
        self.nodes.push(Node {
            value,
            grad: None,
            op,
        });
        Var(id)
    }

    /// Constant (no grad)
    pub fn constant(&mut self, t: Tensor) -> Var {
        self.push(t, Op::Leaf { is_param: false })
    }

    /// Learnable parameter
    pub fn param(&mut self, t: Tensor) -> Var {
        self.push(t, Op::Leaf { is_param: true })
    }

    pub fn value(&self, v: Var) -> &Tensor {
        &self.nodes[v.0].value
    }

    pub fn grad(&self, v: Var) -> Option<&Tensor> {
        self.nodes[v.0].grad.as_ref()
    }

    pub fn is_param(&self, v: Var) -> bool {
        matches!(self.nodes[v.0].op, Op::Leaf { is_param: true })
    }

    // ----- ops -----

    pub fn matmul(&mut self, a: Var, b: Var) -> Result<Var> {
        let va = &self.nodes[a.0].value;
        let vb = &self.nodes[b.0].value;
        if va.rank() != 2 || vb.rank() != 2 || va.shape[1] != vb.shape[0] {
            return Err(SentinelError::ShapeMismatch {
                expected: va.shape.clone(),
                got: vb.shape.clone(),
            });
        }
        let m = va.shape[0];
        let k = va.shape[1];
        let n = vb.shape[1];
        let mut data = vec![0.0; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut s = 0.0f32;
                for p in 0..k {
                    s += va.data[i * k + p] * vb.data[p * n + j];
                }
                data[i * n + j] = s;
            }
        }
        let out = Tensor::from_vec(data, &[m, n])?;
        Ok(self.push(out, Op::Matmul { a, b }))
    }

    pub fn add(&mut self, a: Var, b: Var) -> Result<Var> {
        let out = self.nodes[a.0].value.add(&self.nodes[b.0].value)?;
        Ok(self.push(out, Op::Add { a, b }))
    }

    pub fn mul(&mut self, a: Var, b: Var) -> Result<Var> {
        let va = &self.nodes[a.0].value;
        let vb = &self.nodes[b.0].value;
        if va.shape != vb.shape {
            return Err(SentinelError::ShapeMismatch {
                expected: va.shape.clone(),
                got: vb.shape.clone(),
            });
        }
        let data: Vec<Float> = va
            .data
            .iter()
            .zip(vb.data.iter())
            .map(|(x, y)| x * y)
            .collect();
        let out = Tensor::from_vec(data, &va.shape)?;
        Ok(self.push(out, Op::Mul { a, b }))
    }

    pub fn scale(&mut self, x: Var, factor: Float) -> Result<Var> {
        let mut t = self.nodes[x.0].value.clone();
        t.scale_inplace(factor);
        Ok(self.push(t, Op::Scale { x, factor }))
    }

    pub fn silu(&mut self, x: Var) -> Result<Var> {
        let v = &self.nodes[x.0].value;
        let data: Vec<Float> = v
            .data
            .iter()
            .map(|&a| {
                let sig = 1.0 / (1.0 + (-a).exp());
                a * sig
            })
            .collect();
        let out = Tensor::from_vec(data, &v.shape)?;
        Ok(self.push(out, Op::Silu { x }))
    }

    pub fn rms_norm(&mut self, x: Var, weight: Var, eps: Float) -> Result<Var> {
        let xv = &self.nodes[x.0].value;
        let wv = &self.nodes[weight.0].value;
        let last = *xv.shape.last().unwrap();
        let outer = xv.numel() / last;
        let mut out = vec![0.0; xv.numel()];
        for i in 0..outer {
            let start = i * last;
            let row = &xv.data[start..start + last];
            let mut sq = 0.0f32;
            for &a in row {
                sq += a * a;
            }
            let inv = 1.0 / (sq / last as f32 + eps).sqrt();
            for j in 0..last {
                out[start + j] = row[j] * inv * wv.data[j];
            }
        }
        let t = Tensor::from_vec(out, &xv.shape)?;
        Ok(self.push(t, Op::RmsNorm { x, weight, eps }))
    }

    pub fn embed(&mut self, weight: Var, indices: &[u32]) -> Result<Var> {
        let w = &self.nodes[weight.0].value;
        let d = w.shape[1];
        let mut data = Vec::with_capacity(indices.len() * d);
        for &id in indices {
            let start = id as usize * d;
            data.extend_from_slice(&w.data[start..start + d]);
        }
        let out = Tensor::from_vec(data, &[indices.len(), d])?;
        Ok(self.push(
            out,
            Op::Embed {
                weight,
                indices: indices.to_vec(),
            },
        ))
    }

    pub fn reshape(&mut self, x: Var, shape: &[usize]) -> Result<Var> {
        let t = self.nodes[x.0].value.reshape(shape)?;
        Ok(self.push(t, Op::Reshape { x, shape: shape.to_vec() }))
    }

    /// Scalar CE loss Var (value shape [1]).
    pub fn softmax_ce(&mut self, logits: Var, targets: &[u32]) -> Result<Var> {
        let (loss, _d) = crate::ops::softmax_cross_entropy(&self.nodes[logits.0].value, targets)?;
        let out = Tensor::from_vec(vec![loss], &[1])?;
        Ok(self.push(
            out,
            Op::SoftmaxCe {
                logits,
                targets: targets.to_vec(),
            },
        ))
    }

    fn acc_grad(&mut self, v: Var, g: &Tensor) -> Result<()> {
        let node = &mut self.nodes[v.0];
        match &mut node.grad {
            Some(existing) => {
                for (e, ng) in existing.data.iter_mut().zip(g.data.iter()) {
                    *e += *ng;
                }
            }
            None => {
                node.grad = Some(g.clone());
            }
        }
        Ok(())
    }

    /// Reverse-mode autodiff from a scalar loss variable.
    pub fn backward(&mut self, loss: Var) -> Result<()> {
        // seed
        let ones = Tensor::ones(&[1]);
        self.nodes[loss.0].grad = Some(ones);

        for i in (0..self.nodes.len()).rev() {
            let grad = match &self.nodes[i].grad {
                Some(g) => g.clone(),
                None => continue,
            };
            let op = self.nodes[i].op.clone();
            match op {
                Op::Leaf { .. } => {}
                Op::Matmul { a, b } => {
                    // dA = G @ B^T, dB = A^T @ G
                    let va = &self.nodes[a.0].value;
                    let vb = &self.nodes[b.0].value;
                    let (m, k) = (va.shape[0], va.shape[1]);
                    let n = vb.shape[1];
                    // G: [m,n]
                    let mut da = vec![0.0; m * k];
                    for i in 0..m {
                        for p in 0..k {
                            let mut s = 0.0f32;
                            for j in 0..n {
                                s += grad.data[i * n + j] * vb.data[p * n + j];
                            }
                            da[i * k + p] = s;
                        }
                    }
                    let mut db = vec![0.0; k * n];
                    for p in 0..k {
                        for j in 0..n {
                            let mut s = 0.0f32;
                            for i in 0..m {
                                s += va.data[i * k + p] * grad.data[i * n + j];
                            }
                            db[p * n + j] = s;
                        }
                    }
                    self.acc_grad(a, &Tensor::from_vec(da, &[m, k])?)?;
                    self.acc_grad(b, &Tensor::from_vec(db, &[k, n])?)?;
                }
                Op::Add { a, b } => {
                    self.acc_grad(a, &grad)?;
                    self.acc_grad(b, &grad)?;
                }
                Op::Mul { a, b } => {
                    let va = &self.nodes[a.0].value;
                    let vb = &self.nodes[b.0].value;
                    let da: Vec<Float> = grad
                        .data
                        .iter()
                        .zip(vb.data.iter())
                        .map(|(g, y)| g * y)
                        .collect();
                    let db: Vec<Float> = grad
                        .data
                        .iter()
                        .zip(va.data.iter())
                        .map(|(g, x)| g * x)
                        .collect();
                    self.acc_grad(a, &Tensor::from_vec(da, &va.shape)?)?;
                    self.acc_grad(b, &Tensor::from_vec(db, &vb.shape)?)?;
                }
                Op::Scale { x, factor } => {
                    let mut g = grad.clone();
                    g.scale_inplace(factor);
                    self.acc_grad(x, &g)?;
                }
                Op::Silu { x } => {
                    // d/dx [x*sigmoid(x)] = sig + x*sig*(1-sig)
                    let v = &self.nodes[x.0].value;
                    let mut gx = vec![0.0; v.numel()];
                    for i in 0..v.numel() {
                        let a = v.data[i];
                        let sig = 1.0 / (1.0 + (-a).exp());
                        gx[i] = grad.data[i] * (sig + a * sig * (1.0 - sig));
                    }
                    self.acc_grad(x, &Tensor::from_vec(gx, &v.shape)?)?;
                }
                Op::RmsNorm { x, weight, eps } => {
                    let xv = &self.nodes[x.0].value;
                    let wv = &self.nodes[weight.0].value;
                    let last = *xv.shape.last().unwrap();
                    let outer = xv.numel() / last;
                    let mut gx = vec![0.0; xv.numel()];
                    let mut gw = vec![0.0; last];
                    for i in 0..outer {
                        let start = i * last;
                        let row = &xv.data[start..start + last];
                        let grow = &grad.data[start..start + last];
                        let mut sq = 0.0f32;
                        for &a in row {
                            sq += a * a;
                        }
                        let rms = (sq / last as f32 + eps).sqrt();
                        let inv = 1.0 / rms;
                        // Simplified grad w.r.t x (diagonal approx + weight)
                        for j in 0..last {
                            gw[j] += grow[j] * row[j] * inv;
                            gx[start + j] += grow[j] * inv * wv.data[j];
                        }
                    }
                    self.acc_grad(x, &Tensor::from_vec(gx, &xv.shape)?)?;
                    self.acc_grad(weight, &Tensor::from_vec(gw, &wv.shape)?)?;
                }
                Op::Embed { weight, indices } => {
                    let w = &self.nodes[weight.0].value;
                    let d = w.shape[1];
                    let mut gw = vec![0.0; w.numel()];
                    for (t, &id) in indices.iter().enumerate() {
                        let src = t * d;
                        let dst = id as usize * d;
                        for j in 0..d {
                            gw[dst + j] += grad.data[src + j];
                        }
                    }
                    self.acc_grad(weight, &Tensor::from_vec(gw, &w.shape)?)?;
                }
                Op::SoftmaxCe { logits, targets } => {
                    let (_loss, d_logits) =
                        crate::ops::softmax_cross_entropy(&self.nodes[logits.0].value, &targets)?;
                    // loss scalar grad is 1
                    self.acc_grad(logits, &d_logits)?;
                }
                Op::Reshape { x, shape: _ } => {
                    let mut g = grad.clone();
                    g.shape = self.nodes[x.0].value.shape.clone();
                    self.acc_grad(x, &g)?;
                }
            }
        }
        Ok(())
    }

    /// Collect gradients for all parameter leaves.
    pub fn param_grads(&self) -> GradMap {
        let mut map = GradMap::new();
        for (i, node) in self.nodes.iter().enumerate() {
            if let Op::Leaf { is_param: true } = node.op {
                if let Some(g) = &node.grad {
                    map.insert(Var(i), g.clone());
                }
            }
        }
        map
    }
}

impl Default for Tape {
    fn default() -> Self {
        Self::new()
    }
}
