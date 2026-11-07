//! **Bhutu** — Orbit inference runtime CLI (Prefill / Decode / Eval / Train).
//!
//! Inference: Prefill → Decode (KV cache) + temperature / top-p sampling
//! Training:  causal LM steps via sentinel-train

use clap::{Parser, Subcommand};
use log::info;
use sentinel_core::config::ModelConfig;
use sentinel_core::{product_tag, BHUTU, DOBU, ORBIT, SENTINEL, TESLA};
use sentinel_kernel::{cuda_feature_enabled, Device, KERNEL_VERSION};
use sentinel_tokenizer::{load_tokenizer, Tokenizer};
use sentinel_train::{evaluate_dataset, evaluate_tokens, load_checkpoint, TrainConfig, Trainer};
use sentinel_transformer::{SamplingConfig, SentinelModel};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "sentinel", about = "Orbit · Sentinel LLM · Tesla kernel · DoBu data · Bhutu runtime")]
struct Cli {
    #[arg(long, global = true)]
    tokenizer: Option<PathBuf>,

    /// Compute device: cpu | cuda | cuda:0
    #[arg(long, global = true, default_value = "cpu")]
    device: String,

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
        #[arg(long)]
        load: Option<PathBuf>,
    },
    /// Prefill + Decode with sampling
    Generate {
        #[arg(long, default_value = "Hello")]
        prompt: String,
        #[arg(long, default_value = "16")]
        max_tokens: usize,
        #[arg(long, default_value = "0.8")]
        temperature: f32,
        #[arg(long, default_value = "0.9")]
        top_p: f32,
        #[arg(long, default_value = "50")]
        top_k: usize,
        /// Load Orbit checkpoint (SENTCKP1)
        #[arg(long)]
        load: Option<PathBuf>,
    },
    Smoke {
        #[arg(long, default_value = "Hello world")]
        prompt: String,
        #[arg(long, default_value = "4")]
        steps: usize,
        #[arg(long)]
        load: Option<PathBuf>,
    },
    /// Bhutu HTTP API (company MVP serve)
    Serve {
        #[arg(long, default_value = "127.0.0.1:8787")]
        bind: String,
        #[arg(long)]
        load: Option<PathBuf>,
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
    /// Evaluate loss / perplexity on token ids or synthetic data
    Eval {
        #[arg(long, default_value = "64")]
        seq_len: usize,
        /// Max windows to score (0 = all)
        #[arg(long, default_value = "0")]
        max_windows: usize,
        /// Token-id file (whitespace-separated u32)
        #[arg(long)]
        data: Option<PathBuf>,
        /// Load checkpoint
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

fn load_model(load: Option<&PathBuf>) -> anyhow::Result<SentinelModel> {
    if let Some(path) = load {
        info!("Loading checkpoint {}", path.display());
        Ok(load_checkpoint(path)?)
    } else {
        Ok(SentinelModel::new(ModelConfig::sentinel_small()))
    }
}

fn main() -> anyhow::Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let cli = Cli::parse();
    let tokenizer = load_tokenizer(cli.tokenizer.as_deref())?;
    let device = Device::parse(&cli.device).map_err(|e| anyhow::anyhow!(e))?;
    if device.is_cuda() {
        match sentinel_kernel::create_context(device) {
            Ok(ctx) => log::info!("GPU context: {} ({})", ctx.device_name, ctx.backend),
            Err(e) => log::warn!("GPU requested but unavailable: {}", e),
        }
    }

    match cli.command {
        Commands::Info { size } => {
            let config = match size.as_str() {
                "medium" => ModelConfig::sentinel_medium(),
                _ => ModelConfig::sentinel_small(),
            };
            println!("{}", product_tag());
            println!("Platform     : {}", ORBIT);
            println!("Model        : {} (MoE + Looped Transformer)", SENTINEL);
            println!("Kernel       : {} ({})", TESLA, KERNEL_VERSION);
            println!("Data         : {} (tokenizer / datasets)", DOBU);
            println!("Runtime      : {} (prefill → decode)", BHUTU);
            println!("Inference    : Prefill → Decode (KV cache)");
            println!("Sampling     : temperature / top-p / top-k");
            println!("Training     : Full autograd + AdamW + ckpt");
            println!("Eval         : CE loss + perplexity");
            println!("Device       : {} (cuda_feature={})", device, cuda_feature_enabled());
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
        Commands::Prefill { prompt, load } => {
            info!("Building model...");
            let model = load_model(load.as_ref())?;
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
            load,
        } => {
            info!("Building model...");
            let model = load_model(load.as_ref())?;
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
        Commands::Smoke { prompt, steps, load } => {
            info!("Smoke test: Prefill → {} Decode steps", steps);
            let model = load_model(load.as_ref())?;
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

        Commands::Eval {
            seq_len,
            max_windows,
            data,
            load,
        } => {
            info!("Eval / perplexity | seq_len={}", seq_len);
            let model = if let Some(ref path) = load {
                info!("Loading checkpoint {}", path.display());
                load_checkpoint(path)?
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
                SentinelModel::new(model_cfg)
            };

            let metrics = if let Some(ref path) = data {
                let ds = sentinel_train::TokenDataset::from_id_file(path, seq_len)?;
                if max_windows > 0 {
                    let take = max_windows.saturating_mul(seq_len).max(seq_len);
                    let end = ds.tokens.len().min(take);
                    evaluate_tokens(&model, &ds.tokens[..end], seq_len, None)?
                } else {
                    evaluate_dataset(&model, &ds)?
                }
            } else {
                let stream: Vec<u32> = (0..seq_len * 32).map(|i| (i % 256) as u32).collect();
                let stream = if max_windows > 0 {
                    let take = max_windows.saturating_mul(seq_len).max(seq_len);
                    stream[..stream.len().min(take)].to_vec()
                } else {
                    stream
                };
                evaluate_tokens(&model, &stream, seq_len, None)?
            };

            println!("Eval results");
            println!("  windows : {}", metrics.num_windows);
            println!("  tokens  : {}", metrics.num_tokens);
            println!("  loss    : {:.6}", metrics.loss);
            println!("  ppl     : {:.4}", metrics.perplexity);
        }


        Commands::Serve { bind, load } => {
            info!("Bhutu serve on http://{}", bind);
            let model = load_model(load.as_ref())?;
            bhutu_serve(&bind, model, tokenizer)?;
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


/// Minimal HTTP server for company MVP (no external web framework).
fn bhutu_serve(bind: &str, model: SentinelModel, tokenizer: Tokenizer) -> anyhow::Result<()> {
    use std::io::{Read, Write};
    use std::net::TcpListener;

    let listener = TcpListener::bind(bind)?;
    println!("Bhutu listening on http://{}", bind);
    println!("  GET  /health");
    println!("  GET  /v1/info");
    println!("  POST /v1/generate  body: plain prompt text");

    for stream in listener.incoming() {
        let mut stream = match stream {
            Ok(s) => s,
            Err(_) => continue,
        };
        let mut buf = [0u8; 8192];
        let n = stream.read(&mut buf).unwrap_or(0);
        if n == 0 {
            continue;
        }
        let req = String::from_utf8_lossy(&buf[..n]);
        let first = req.lines().next().unwrap_or("");
        let (status, content_type, body) = if first.starts_with("GET /health") {
            (200, "text/plain", "ok".to_string())
        } else if first.starts_with("GET /v1/info") {
            let json = format!(
                r#"{{"platform":"{}","model":"{}","kernel":"{}","runtime":"{}"}}"#,
                ORBIT, SENTINEL, TESLA, BHUTU
            );
            (200, "application/json", json)
        } else if first.starts_with("POST /v1/generate") {
            // body after blank line
            let prompt = req
                .split("\r\n\r\n")
                .nth(1)
                .or_else(|| req.split("\n\n").nth(1))
                .unwrap_or("Hello")
                .trim();
            let prompt = if prompt.is_empty() { "Hello" } else { prompt };
            match tokenizer.encode_with_special(prompt, true, false) {
                Ok(ids) => {
                    let sampling = SamplingConfig {
                        temperature: 0.8,
                        top_p: 0.9,
                        top_k: 50,
                        ban: Vec::new(),
                    };
                    match model.generate(&ids, 16, &sampling) {
                        Ok(out) => {
                            let decoded = tokenizer.decode(&out).unwrap_or_default();
                            let json = format!(
                                r#"{{"prompt":{},"tokens":{},"text":{}}}"#,
                                serde_json::to_string(prompt).unwrap_or_else(|_| "\"\"".into()),
                                serde_json::to_string(&out).unwrap_or_else(|_| "[]".into()),
                                serde_json::to_string(&decoded).unwrap_or_else(|_| "\"\"".into()),
                            );
                            (200, "application/json", json)
                        }
                        Err(e) => (500, "text/plain", format!("generate error: {}", e)),
                    }
                }
                Err(e) => (400, "text/plain", format!("tokenize error: {}", e)),
            }
        } else {
            (404, "text/plain", "not found".to_string())
        };

        let resp = format!(
            "HTTP/1.1 {} {}\r\nContent-Type: {}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            status,
            if status == 200 { "OK" } else if status == 404 { "Not Found" } else { "Error" },
            content_type,
            body.len(),
            body
        );
        let _ = stream.write_all(resp.as_bytes());
    }
    Ok(())
}
