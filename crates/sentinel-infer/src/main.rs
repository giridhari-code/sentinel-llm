//! Sentinel Inference CLI

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
    /// Run a quick forward pass smoke test
    Smoke {
        #[arg(long, default_value = "16")]
        seq_len: usize,
    },
    /// Encode text → token IDs
    Encode {
        /// Text to encode
        text: String,
        /// Add BOS token
        #[arg(long)]
        bos: bool,
        /// Add EOS token
        #[arg(long)]
        eos: bool,
    },
    /// Decode token IDs → text
    Decode {
        /// Space-separated token IDs
        ids: Vec<u32>,
    },
    /// Generate tokens from a text prompt (random weights — architecture demo)
    Generate {
        /// Prompt text
        #[arg(long, default_value = "Hello")]
        prompt: String,
        #[arg(long, default_value = "16")]
        max_tokens: usize,
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
        Commands::Smoke { seq_len } => {
            info!("Building Sentinel-Small for smoke test...");
            let config = ModelConfig::sentinel_small();
            let model = SentinelModel::new(config);
            let tokens: Vec<u32> = (0..seq_len as u32).collect();
            info!("Running forward pass (seq_len={})...", seq_len);
            let logits = model.forward(&tokens)?;
            info!("Logits shape: {:?}", logits.shape);
            info!("Smoke test passed ✓");
        }
        Commands::Encode { text, bos, eos } => {
            let ids = tokenizer.encode_with_special(&text, bos, eos)?;
            println!("Text   : {:?}", text);
            println!("Tokens : {:?}", ids);
            println!("Length : {}", ids.len());
            if let Ok(decoded) = tokenizer.decode(&ids) {
                println!("Decode : {:?}", decoded);
            }
        }
        Commands::Decode { ids } => {
            let text = tokenizer.decode(&ids)?;
            println!("IDs    : {:?}", ids);
            println!("Text   : {:?}", text);
        }
        Commands::Generate { prompt, max_tokens } => {
            info!("Building model (random weights)...");
            let config = ModelConfig::sentinel_small();
            let model = SentinelModel::new(config);

            let prompt_ids = tokenizer.encode_with_special(&prompt, true, false)?;
            info!("Prompt: {:?} → {:?}", prompt, prompt_ids);

            let out_ids = model.generate(&prompt_ids, max_tokens)?;
            let new_ids = &out_ids[prompt_ids.len()..];
            let decoded = tokenizer.decode(&out_ids)?;

            println!("Prompt tokens : {:?}", prompt_ids);
            println!("Generated IDs : {:?}", new_ids);
            println!("Full decode   : {:?}", decoded);
        }
    }

    Ok(())
}
