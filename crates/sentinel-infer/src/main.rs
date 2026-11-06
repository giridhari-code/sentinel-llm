//! Sentinel Inference CLI
//!
//! Inference path: Prefill → Decode (no separate "encode" stage for the model).

use clap::{Parser, Subcommand};
use log::info;
use sentinel_core::config::ModelConfig;
use sentinel_transformer::SentinelModel;
use sentinel_kernel::KERNEL_VERSION;
use sentinel_tokenizer::{load_tokenizer, Tokenizer};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "sentinel", about = "Sentinel — MoE Looped Transformer LLM (Tesla Kernel)")]
struct Cli {
    /// Optional path to HuggingFace tokenizer.json
    #[arg(long, global = true)]
    tokenizer: Option<PathBuf>,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Print model info and architecture
    Info {
        #[arg(long, default_value = "small")]
        size: String,
    },
    /// Prefill phase — process full prompt, return last-position logits shape
    Prefill {
        /// Prompt text
        #[arg(long, default_value = "Hello Sentinel")]
        prompt: String,
    },
    /// Full generation = Prefill + Decode loop
    Generate {
        /// Prompt text
        #[arg(long, default_value = "Hello")]
        prompt: String,
        #[arg(long, default_value = "16")]
        max_tokens: usize,
    },
    /// Smoke test (prefill + a few decode steps)
    Smoke {
        #[arg(long, default_value = "Hello world")]
        prompt: String,
        #[arg(long, default_value = "4")]
        steps: usize,
    },
    /// Tokenizer helper: text → token IDs (not model encode)
    Tok {
        text: String,
        #[arg(long)]
        bos: bool,
        #[arg(long)]
        eos: bool,
    },
    /// Tokenizer helper: token IDs → text
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
            println!("╔══════════════════════════════════════════════════╗");
            println!("║          SENTINEL — Orbit Model                  ║");
            println!("╠══════════════════════════════════════════════════╣");
            println!("║  Architecture : MoE + Looped Transformer         ║");
            println!("║  Inference    : Prefill → Decode                 ║");
            println!("║  Kernel       : {} (custom, no 3rd party)   ║", KERNEL_VERSION);
            println!("║  Model name   : {}                          ║", config.name);
            println!("╠══════════════════════════════════════════════════╣");
            println!("║  d_model      : {:>6}                           ║", config.d_model);
            println!("║  n_heads      : {:>6}                           ║", config.n_heads);
            println!("║  n_layers     : {:>6}  (unique)                 ║", config.n_layers);
            println!("║  n_loops      : {:>6}  (effective depth)        ║", config.n_loops);
            println!("║  n_experts    : {:>6}  (top_k={})                ║", config.n_experts, config.top_k);
            println!("║  d_ff         : {:>6}                           ║", config.d_ff);
            println!("║  max_seq_len  : {:>6}                           ║", config.max_seq_len);
            println!("║  vocab_size   : {:>6}  (model config)           ║", config.vocab_size);
            println!("║  tok vocab    : {:>6}  (tokenizer)              ║", tokenizer.vocab_size());
            println!("╠══════════════════════════════════════════════════╣");
            println!("║  Est. params  : ~{:.2}M                          ║", config.estimate_params() as f64 / 1e6);
            println!("╚══════════════════════════════════════════════════╝");
        }
        Commands::Prefill { prompt } => {
            info!("Building model...");
            let config = ModelConfig::sentinel_small();
            let model = SentinelModel::new(config);

            let prompt_ids = tokenizer.encode_with_special(&prompt, true, false)?;
            info!("Prefill prompt: {:?} → {} tokens", prompt, prompt_ids.len());

            let step = model.prefill(&prompt_ids)?;
            info!("Prefill done. cache_len={} logits_shape={:?}", step.cache.len, step.logits.shape);

            let next = SentinelModel::sample_greedy(&step.logits);
            println!("Prompt tokens : {:?}", prompt_ids);
            println!("Cache length  : {}", step.cache.len);
            println!("Next token ID : {}", next);
            if let Ok(t) = tokenizer.decode(&[next]) {
                println!("Next token    : {:?}", t);
            }
        }
        Commands::Generate { prompt, max_tokens } => {
            info!("Building model (random weights)...");
            let config = ModelConfig::sentinel_small();
            let model = SentinelModel::new(config);

            let prompt_ids = tokenizer.encode_with_special(&prompt, true, false)?;
            info!("Prefill + Decode | prompt={:?} max_new={}", prompt, max_tokens);

            let out_ids = model.generate(&prompt_ids, max_tokens)?;
            let new_ids = &out_ids[prompt_ids.len()..];
            let decoded = tokenizer.decode(&out_ids)?;

            println!("Prompt tokens : {:?}", prompt_ids);
            println!("Generated IDs : {:?}", new_ids);
            println!("Full decode   : {:?}", decoded);
        }
        Commands::Smoke { prompt, steps } => {
            info!("Smoke test: Prefill → {} Decode steps", steps);
            let config = ModelConfig::sentinel_small();
            let model = SentinelModel::new(config);

            let prompt_ids = tokenizer.encode_with_special(&prompt, true, false)?;
            info!("Prefill...");
            let mut step = model.prefill(&prompt_ids)?;
            info!("Prefill OK | cache_len={} logits={:?}", step.cache.len, step.logits.shape);

            let mut tokens = prompt_ids.clone();
            for i in 0..steps {
                let next = SentinelModel::sample_greedy(&step.logits);
                tokens.push(next);
                step = model.decode(next, &step.cache)?;
                info!("Decode step {} → token {}", i + 1, next);
            }
            info!("Smoke test passed ✓");
            println!("Final tokens: {:?}", tokens);
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
