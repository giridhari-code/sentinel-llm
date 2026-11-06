//! Sentinel Inference CLI
//!
//! Inference: Prefill → Decode (KV cache) + temperature / top-p sampling
//! Training:  causal LM steps via sentinel-train

use clap::{Parser, Subcommand};
use log::info;
use sentinel_core::config::ModelConfig;
use sentinel_kernel::KERNEL_VERSION;
use sentinel_tokenizer::{load_tokenizer, Tokenizer};
use sentinel_train::{TrainConfig, Trainer};
use sentinel_transformer::{SamplingConfig, SentinelModel};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "sentinel", about = "Sentinel — MoE Looped Transformer LLM (Tesla Kernel)")]
struct Cli {
    #[arg(long, global = true)]
    tokenizer: Option<PathBuf>,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Info {
        #[arg(long, default_value = "small")]
        size: String,
    },
    Prefill {
        #[arg(long, default_value = "Hello Sentinel")]
        prompt: String,
    },
    /// Prefill + Decode with sampling
    Generate {
        #[arg(long, default_value = "Hello")]
        prompt: String,
        #[arg(long, default_value = "16")]
        max_tokens: usize,
        /// Softmax temperature (0 = greedy)
        #[arg(long, default_value = "0.8")]
        temperature: f32,
        /// Nucleus top-p
        #[arg(long, default_value = "0.9")]
        top_p: f32,
        /// Top-k (0 = off)
        #[arg(long, default_value = "50")]
        top_k: usize,
    },
    Smoke {
        #[arg(long, default_value = "Hello world")]
        prompt: String,
        #[arg(long, default_value = "4")]
        steps: usize,
    },
    /// Run a few causal-LM training steps on synthetic tokens
    Train {
        #[arg(long, default_value = "20")]
        steps: usize,
        #[arg(long, default_value = "64")]
        seq_len: usize,
        #[arg(long, default_value = "3e-4")]
        lr: f32,
        /// Optional token-id file (whitespace separated u32)
        #[arg(long)]
        data: Option<PathBuf>,
        /// Save checkpoint path after training
        #[arg(long)]
        save: Option<PathBuf>,
        /// Load checkpoint before training
        #[arg(long)]
        load: Option<PathBuf>,
    },
    Tok {
        text: String,
        #[arg(long)]
        bos: bool,
        #[arg(long)]
        eos: bool,
    },
    Detok {
        ids: Vec<u32>,
    },
}

fn main() -> anyhow::Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let cli = Cli::parse();
    let tokenizer = load_tokenizer(cli.tokenizer.as_deref())?;

    match cli.command {
        Commands::Info { size } => {
            let config = match size.as_str() {
                "medium" => ModelConfig::sentinel_medium(),
                _ => ModelConfig::sentinel_small(),
            };
            println!("SENTINEL — Orbit Model");
            println!("Architecture : MoE + Looped Transformer");
            println!("Inference    : Prefill → Decode (KV cache)");
            println!("Sampling     : temperature / top-p / top-k");
            println!("Training     : Full autograd + AdamW + ckpt");
            println!("Kernel       : {}", KERNEL_VERSION);
            println!("Model        : {}", config.name);
            println!(
                "d_model={} n_layers={} n_loops={} n_experts={} top_k={}",
                config.d_model, config.n_layers, config.n_loops, config.n_experts, config.top_k
            );
            println!(
                "vocab_size={} tok_vocab={}",
                config.vocab_size,
                tokenizer.vocab_size()
            );
            println!(
                "Est. params  : ~{:.2}M",
                config.estimate_params() as f64 / 1e6
            );
        }
        Commands::Prefill { prompt } => {
            info!("Building model...");
            let model = SentinelModel::new(ModelConfig::sentinel_small());
            let prompt_ids = tokenizer.encode_with_special(&prompt, true, false)?;
            info!("Prefill prompt: {:?} → {} tokens", prompt, prompt_ids.len());

            let step = model.prefill(&prompt_ids)?;
            info!(
                "Prefill done. cache_len={} logits_shape={:?}",
                step.cache.len, step.logits.shape
            );

            let next = SentinelModel::sample_greedy(&step.logits);
            println!("Prompt tokens : {:?}", prompt_ids);
            println!("Cache length  : {}", step.cache.len);
            println!("Next token ID : {}", next);
        }
        Commands::Generate {
            prompt,
            max_tokens,
            temperature,
            top_p,
            top_k,
        } => {
            info!("Building model...");
            let model = SentinelModel::new(ModelConfig::sentinel_small());
            let prompt_ids = tokenizer.encode_with_special(&prompt, true, false)?;
            let sampling = SamplingConfig {
                temperature,
                top_p,
                top_k,
                ban: Vec::new(),
            };
            info!(
                "Generate | temp={} top_p={} top_k={} max_new={}",
                temperature, top_p, top_k, max_tokens
            );

            let out_ids = model.generate(&prompt_ids, max_tokens, &sampling)?;
            let new_ids = &out_ids[prompt_ids.len()..];
            let decoded = tokenizer.decode(&out_ids)?;

            println!("Prompt tokens : {:?}", prompt_ids);
            println!("Generated IDs : {:?}", new_ids);
            println!("Full decode   : {:?}", decoded);
        }
        Commands::Smoke { prompt, steps } => {
            info!("Smoke test: Prefill → {} Decode steps", steps);
            let model = SentinelModel::new(ModelConfig::sentinel_small());
            let prompt_ids = tokenizer.encode_with_special(&prompt, true, false)?;

            let mut step = model.prefill(&prompt_ids)?;
            info!(
                "Prefill OK | cache_len={} logits={:?}",
                step.cache.len, step.logits.shape
            );

            let sampling = SamplingConfig::creative();
            let mut tokens = prompt_ids.clone();
            for i in 0..steps {
                let next = sentinel_transformer::sample_token(&step.logits, &sampling);
                tokens.push(next);
                step = model.decode(next, &step.cache)?;
                info!(
                    "Decode step {} → token {} | cache_len={}",
                    i + 1,
                    next,
                    step.cache.len
                );
            }
            info!("Smoke test passed ✓");
            println!("Final tokens: {:?}", tokens);
        }
        Commands::Train {
            steps,
            seq_len,
            lr,
            data,
            save,
            load,
        } => {
            info!(
                "Training | steps={} seq_len={} lr={} full_autograd=true",
                steps, seq_len, lr
            );
            let train_cfg = TrainConfig {
                lr,
                max_seq_len: seq_len,
                train_embed: true,
                full_autograd: true,
                ..TrainConfig::default()
            };

            let mut trainer = if let Some(ref path) = load {
                info!("Loading checkpoint {}", path.display());
                Trainer::from_checkpoint(path, train_cfg)?
            } else {
                let mut model_cfg = ModelConfig::sentinel_small();
                model_cfg.d_model = 128;
                model_cfg.n_heads = 4;
                model_cfg.d_head = 32;
                model_cfg.d_ff = 256;
                model_cfg.n_layers = 2;
                model_cfg.n_loops = 1;
                model_cfg.n_experts = 2;
                model_cfg.vocab_size = 256;
                model_cfg.max_seq_len = seq_len;
                Trainer::new(model_cfg, train_cfg)
            };

            let stats = if let Some(ref path) = data {
                info!("Dataset file {}", path.display());
                let ds = sentinel_train::TokenDataset::from_id_file(path, seq_len)?;
                trainer.train_on_dataset(&ds, steps)?
            } else {
                let stream: Vec<u32> = (0..seq_len * steps * 2)
                    .map(|i| (i % 256) as u32)
                    .collect();
                trainer.train_on_tokens(&stream, steps, seq_len)?
            };

            for s in &stats {
                info!("step {} | loss={:.4} ce={:.4} aux={:.4} | tokens={}", s.step, s.loss, s.ce_loss, s.aux_loss, s.tokens);
            }
            if let Some(last) = stats.last() {
                println!(
                    "Train done | steps={} final_loss={:.4}",
                    last.step, last.loss
                );
            }
            if let Some(ref path) = save {
                trainer.save(path)?;
                println!("Checkpoint saved → {}", path.display());
            }
        }
        Commands::Tok { text, bos, eos } => {
            let ids = tokenizer.encode_with_special(&text, bos, eos)?;
            println!("Text   : {:?}", text);
            println!("Tokens : {:?}", ids);
            println!("Length : {}", ids.len());
        }
        Commands::Detok { ids } => {
            let text = tokenizer.decode(&ids)?;
            println!("IDs    : {:?}", ids);
            println!("Text   : {:?}", text);
        }
    }

    Ok(())
}
