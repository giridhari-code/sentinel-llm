//! Train a tiny Sentinel and write real weights to models/sentinel-small.ckpt

use sentinel_core::config::ModelConfig;
use sentinel_train::{TrainConfig, Trainer};
use std::path::PathBuf;

fn main() {
    let mut cfg = ModelConfig::sentinel_small();
    cfg.d_model = 64;
    cfg.n_heads = 4;
    cfg.d_head = 16;
    cfg.d_ff = 128;
    cfg.n_layers = 2;
    cfg.n_loops = 1;
    cfg.n_experts = 2;
    cfg.top_k = 1;
    cfg.vocab_size = 256;
    cfg.max_seq_len = 32;

    let train_cfg = TrainConfig {
        lr: 3e-4,
        max_seq_len: 32,
        grad_clip: 1.0,
        warmup_steps: 5,
        total_steps: 30,
        ..TrainConfig::default()
    };

    let mut trainer = Trainer::new(cfg, train_cfg);
    // synthetic token stream
    let tokens: Vec<u32> = (0..32 * 40).map(|i| (i % 256) as u32).collect();
    let steps = 30usize;
    let seq = 32usize;
    println!("Training {} steps seq={} ...", steps, seq);
    for s in 0..steps {
        let start = (s * 7) % (tokens.len().saturating_sub(seq));
        let window = &tokens[start..start + seq];
        match trainer.train_step(window) {
            Ok(st) => {
                if s % 5 == 0 || s + 1 == steps {
                    println!(
                        "step {:3} loss={:.4} ce={:.4} aux={:.4}",
                        st.step, st.loss, st.ce_loss, st.aux_loss
                    );
                }
            }
            Err(e) => {
                eprintln!("step {} failed: {}", s, e);
                std::process::exit(1);
            }
        }
    }

    let out = PathBuf::from("models/sentinel-small.ckpt");
    if let Some(parent) = out.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    trainer.save(&out).expect("save checkpoint");
    let meta = std::fs::metadata(&out).expect("stat ckpt");
    println!("Saved real trained weights → {} ({} bytes)", out.display(), meta.len());
}
