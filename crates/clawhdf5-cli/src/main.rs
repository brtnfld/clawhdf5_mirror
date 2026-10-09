use std::path::PathBuf;

use clap::{Parser, Subcommand};
use clawhdf5_agent::{AgentMemory, HDF5Memory, MemoryConfig, MemoryEntry};

/// ClawhDF5 — HDF5-backed cognitive memory for AI agents
#[derive(Parser)]
#[command(name = "clawhdf5", version, about)]
struct Cli {
    /// Path to the .h5 memory file
    #[arg(short, long, env = "CLAWHDF5_PATH")]
    path: PathBuf,

    /// Fail closed if the file carries no integrity protection: reject a
    /// missing or stripped `_merkle_root` and a legacy unauthenticated
    /// version-1 WAL sidecar instead of loading them unverified.
    #[arg(long, global = true, env = "CLAWHDF5_STRICT")]
    strict: bool,

    #[command(subcommand)]
    command: Commands,
}

/// Open the memory file under the chosen integrity policy.
fn open_memory(path: &std::path::Path, strict: bool) -> Result<HDF5Memory, clawhdf5_agent::MemoryError> {
    if strict {
        HDF5Memory::open_strict(path)
    } else {
        HDF5Memory::open(path)
    }
}

#[derive(Subcommand)]
enum Commands {
    /// Create a new memory file
    Create {
        /// Agent identifier
        #[arg(long, default_value = "default")]
        agent_id: String,
        /// Embedding dimension
        #[arg(long, default_value_t = 384)]
        dim: usize,
        /// Enable write-ahead log
        #[arg(long)]
        wal: bool,
    },
    /// Save a memory entry (reads JSON from stdin or --json)
    Save {
        /// JSON: {"chunk":"...","embedding":[...],"source_channel":"...","timestamp":0.0,"session_id":"...","tags":""}
        #[arg(long)]
        json: Option<String>,
    },
    /// Search memory by embedding vector
    Search {
        /// Query embedding as JSON array of f32
        #[arg(long)]
        embedding: String,
        /// Optional text query for hybrid BM25+vector search
        #[arg(long, default_value = "")]
        query: String,
        /// Number of results
        #[arg(short = 'k', long, default_value_t = 5)]
        top_k: usize,
        /// Vector similarity weight (0.0-1.0)
        #[arg(long, default_value_t = 0.7)]
        vector_weight: f32,
        /// BM25 keyword weight (0.0-1.0)
        #[arg(long, default_value_t = 0.3)]
        keyword_weight: f32,
    },
    /// Get a specific memory chunk by index
    Recall {
        /// Chunk index
        index: usize,
    },
    /// Show memory stats (count, active, config)
    Stats,
    /// Flush WAL to main HDF5 file
    FlushWal,
    /// Generate AGENTS.md from memory contents
    AgentsMd {
        /// Write to file instead of stdout
        #[arg(long)]
        output: Option<PathBuf>,
    },
    /// Export all entries as JSON lines to stdout
    Export,
    /// Snapshot (copy) memory file to destination
    Snapshot {
        /// Destination path
        dest: PathBuf,
    },
}

fn main() {
    let cli = Cli::parse();

    if let Err(e) = run(cli) {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn run(cli: Cli) -> Result<(), Box<dyn std::error::Error>> {
    match cli.command {
        Commands::Create { agent_id, dim, wal } => {
            let mut config = MemoryConfig::new(cli.path.clone(), &agent_id, dim);
            config.wal_enabled = wal;
            let mem = HDF5Memory::create(config)?;
            let j = serde_json::json!({
                "status": "created",
                "path": cli.path.display().to_string(),
                "agent_id": agent_id,
                "embedding_dim": dim,
                "wal_enabled": wal,
                "count": mem.count(),
            });
            println!("{}", serde_json::to_string_pretty(&j)?);
        }

        Commands::Save { json } => {
            let input = match json {
                Some(s) => s,
                None => {
                    use std::io::Read;
                    let mut buf = String::new();
                    std::io::stdin().read_to_string(&mut buf)?;
                    buf
                }
            };
            let entry: MemoryEntry = serde_json::from_str(&input)?;
            let mut mem = open_memory(&cli.path, cli.strict)?;
            let idx = mem.save(entry)?;
            let j = serde_json::json!({ "status": "saved", "index": idx, "count": mem.count() });
            println!("{}", serde_json::to_string(&j)?);
        }

        Commands::Search {
            embedding,
            query,
            top_k,
            vector_weight,
            keyword_weight,
        } => {
            let emb: Vec<f32> = serde_json::from_str(&embedding)?;
            let mut mem = open_memory(&cli.path, cli.strict)?;
            let results = mem.hybrid_search(&emb, &query, vector_weight, keyword_weight, top_k);
            let j: Vec<serde_json::Value> = results
                .iter()
                .map(|r| {
                    serde_json::json!({
                        "index": r.index,
                        "score": r.score,
                        "chunk": &r.chunk,
                        "timestamp": r.timestamp,
                        "source_channel": &r.source_channel,
                    })
                })
                .collect();
            println!("{}", serde_json::to_string_pretty(&j)?);
        }

        Commands::Recall { index } => {
            let mem = open_memory(&cli.path, cli.strict)?;
            match mem.get_chunk(index) {
                Some(content) => {
                    let j = serde_json::json!({ "index": index, "chunk": content });
                    println!("{}", serde_json::to_string_pretty(&j)?);
                }
                None => {
                    eprintln!("no entry at index {index}");
                    std::process::exit(1);
                }
            }
        }

        Commands::Stats => {
            let mem = open_memory(&cli.path, cli.strict)?;
            let cfg = mem.config();
            let j = serde_json::json!({
                "path": cli.path.display().to_string(),
                "agent_id": cfg.agent_id,
                "embedding_dim": cfg.embedding_dim,
                "count": mem.count(),
                "active": mem.count_active(),
                "wal_enabled": cfg.wal_enabled,
                "wal_pending": mem.wal_pending_count(),
            });
            println!("{}", serde_json::to_string_pretty(&j)?);
        }

        Commands::FlushWal => {
            let mut mem = open_memory(&cli.path, cli.strict)?;
            let before = mem.wal_pending_count();
            mem.flush_wal()?;
            let j = serde_json::json!({
                "status": "flushed",
                "entries_flushed": before,
                "wal_pending": mem.wal_pending_count(),
            });
            println!("{}", serde_json::to_string(&j)?);
        }

        Commands::AgentsMd { output } => {
            let mem = open_memory(&cli.path, cli.strict)?;
            let md = mem.generate_agents_md();
            match output {
                Some(p) => {
                    std::fs::write(&p, &md)?;
                    eprintln!("wrote {}", p.display());
                }
                None => print!("{md}"),
            }
        }

        Commands::Export => {
            let mem = open_memory(&cli.path, cli.strict)?;
            for i in 0..mem.count() {
                if let Some(chunk) = mem.get_chunk(i) {
                    let j = serde_json::json!({ "index": i, "chunk": chunk });
                    println!("{}", serde_json::to_string(&j)?);
                }
            }
        }

        Commands::Snapshot { dest } => {
            let _result = clawhdf5_agent::storage::snapshot_file(&cli.path, &dest)?;
            let j = serde_json::json!({
                "status": "snapshot_created",
                "source": cli.path.display().to_string(),
                "dest": dest.display().to_string(),
            });
            println!("{}", serde_json::to_string(&j)?);
        }
    }

    Ok(())
}
