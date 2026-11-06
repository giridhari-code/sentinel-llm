//! Sentinel Inference CLI

use clap::{Parser, Subcommand};
use log::info;
use sentinel_core::config::ModelConfig;
use sentinel_transformer::SentinelModel;
use sentinel_kernel::KERNEL_VERSION;

#[derive(Parser)]
#[command(name = "sentinel", about = "Sentinel — MoE Looped Transformer LLM (Tesla Kernel)")]
struct Cli {
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
    /// Generate tokens (random init — just for architecture demo)
    Generate {
        #[arg(long, default_value = "8")]
        max_tokens: usize,
    },
}

fn main() -> anyhow::Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let cli = Cli::parse();

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
            println!("║  vocab_size   : {:>6}                           ║", config.vocab_size);
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
        Commands::Generate { max_tokens } => {
            info!("Building model (random weights)...");
            let config = ModelConfig::sentinel_small();
            let model = SentinelModel::new(config);
            let prompt = vec![1u32, 2, 3, 4];
            info!("Generating {} tokens from prompt {:?}...", max_tokens, prompt);
            let out = model.generate(&prompt, max_tokens)?;
            println!("Generated token IDs: {:?}", out);
        }
    }

    Ok(())
}
