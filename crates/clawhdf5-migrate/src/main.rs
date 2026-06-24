mod hdf5_reader;
mod hdf5_writer;
mod sqlite_reader;
mod validate;

use clap::Parser;

use sqlite_reader::SchemaConfig;

/// Migrate ZeroClaw agent memory from SQLite to HDF5 format.
#[derive(Parser, Debug)]
#[command(name = "clawhdf5-migrate", version, about)]
struct Cli {
    /// Source SQLite database path
    #[arg(long)]
    sqlite: String,

    /// Destination HDF5 file path
    #[arg(long)]
    hdf5: String,

    /// Agent ID for metadata
    #[arg(long, default_value = "migrated")]
    agent_id: String,

    /// Embedder name for metadata
    #[arg(long, default_value = "unknown")]
    embedder: String,

    /// Embedding dimension (auto-detect from first row if not specified)
    #[arg(long)]
    embedding_dim: Option<usize>,

    /// Skip deleted/tombstoned entries
    #[arg(long)]
    skip_deleted: bool,

    /// Enable deflate compression on embeddings
    #[arg(long)]
    compression: bool,

    /// Compression level 1-9
    #[arg(long, default_value_t = 4)]
    compression_level: u32,

    /// Store embeddings as float16 (halves storage)
    #[arg(long)]
    float16: bool,

    /// Validate without writing
    #[arg(long)]
    dry_run: bool,

    /// Content-check every migrated row (default: a representative sample)
    #[arg(long)]
    validate_full: bool,

    /// Append only rows newer than the existing output (by chunk id), merging
    /// into the file at --hdf5 if it exists
    #[arg(long)]
    incremental: bool,

    /// Override the SQLite table name for memory chunks
    #[arg(long)]
    chunks_table: Option<String>,

    /// Override the SQLite table name for sessions
    #[arg(long)]
    sessions_table: Option<String>,

    /// Override the SQLite table name for entities
    #[arg(long)]
    entities_table: Option<String>,

    /// Override the SQLite table name for relations
    #[arg(long)]
    relations_table: Option<String>,

    /// Print progress
    #[arg(long)]
    verbose: bool,
}

/// Build the schema config from CLI table-name overrides (defaults otherwise).
fn schema_from_cli(cli: &Cli) -> SchemaConfig {
    let mut c = SchemaConfig::default();
    if let Some(t) = &cli.chunks_table {
        c.chunks.table = t.clone();
    }
    if let Some(t) = &cli.sessions_table {
        c.sessions.table = t.clone();
    }
    if let Some(t) = &cli.entities_table {
        c.entities.table = t.clone();
    }
    if let Some(t) = &cli.relations_table {
        c.relations.table = t.clone();
    }
    c
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    let schema = schema_from_cli(&cli);

    // Dry run: a fast count-only pass that does not buffer the database.
    if cli.dry_run {
        let counts = sqlite_reader::read_counts(&cli.sqlite, cli.skip_deleted, &schema)?;
        eprintln!("Dry run — no output file written.");
        eprintln!(
            "Would migrate: {} chunks, {} sessions, {} entities, {} relations",
            counts.chunks, counts.sessions, counts.entities, counts.relations
        );
        return Ok(());
    }

    if cli.verbose {
        eprintln!("Reading SQLite database: {}", cli.sqlite);
    }

    // Incremental: merge new rows into the existing output (if present).
    let incremental_base = if cli.incremental && std::path::Path::new(&cli.hdf5).exists() {
        Some(hdf5_reader::read_hdf5(&cli.hdf5)?)
    } else {
        None
    };
    let min_chunk_id = incremental_base
        .as_ref()
        .map(|d| d.chunks.iter().map(|c| c.id).max().unwrap_or(0))
        .unwrap_or(0);
    let dim_hint = cli
        .embedding_dim
        .or_else(|| incremental_base.as_ref().map(|d| d.embedding_dim));

    let source = if min_chunk_id > 0 {
        sqlite_reader::read_sqlite_filtered(
            &cli.sqlite,
            cli.skip_deleted,
            dim_hint,
            &schema,
            min_chunk_id,
        )?
    } else {
        sqlite_reader::read_sqlite(&cli.sqlite, cli.skip_deleted, dim_hint, &schema)?
    };

    // Build the dataset to write: either the source alone, or the existing
    // output plus the newly-read rows (metadata groups refreshed from source).
    let data = match incremental_base {
        Some(mut base) => {
            let added = source.chunks.len();
            base.chunks.extend(source.chunks);
            base.sessions = source.sessions;
            base.entities = source.entities;
            base.relations = source.relations;
            base.embedding_dim = source.embedding_dim.max(base.embedding_dim);
            if cli.verbose {
                eprintln!("Incremental: appended {added} new chunks (id > {min_chunk_id})");
            }
            base
        }
        None => source,
    };

    if cli.verbose {
        eprintln!(
            "Migrating {} chunks, {} sessions, {} entities, {} relations (dim={})",
            data.chunks.len(),
            data.sessions.len(),
            data.entities.len(),
            data.relations.len(),
            data.embedding_dim
        );
        eprintln!("Writing HDF5 file: {}", cli.hdf5);
    }

    let opts = hdf5_writer::WriteOptions {
        agent_id: cli.agent_id,
        embedder: cli.embedder,
        compression: cli.compression,
        compression_level: cli.compression_level.clamp(1, 9),
        float16: cli.float16,
    };

    hdf5_writer::write_hdf5(&cli.hdf5, &data, &opts)?;

    if cli.verbose {
        eprintln!("Validating output (content check)...");
    }

    let summary = validate::validate_hdf5(&cli.hdf5, &data, cli.validate_full, cli.float16)?;

    eprintln!(
        "Migration complete: {} chunks, {} sessions, {} entities, {} relations (dim={}); {} rows content-verified",
        summary.chunks,
        summary.sessions,
        summary.entities,
        summary.relations,
        summary.embedding_dim,
        summary.rows_checked,
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rusqlite::Connection;
    use tempfile::TempDir;

    /// Create a test SQLite database with the ZeroClaw schema.
    fn create_test_db(dir: &TempDir) -> String {
        let db_path = dir.path().join("test.db");
        let path_str = db_path.to_str().unwrap().to_string();
        let conn = Connection::open(&path_str).unwrap();
        conn.execute_batch(
            "CREATE TABLE memory_chunks (
                id INTEGER PRIMARY KEY,
                chunk TEXT NOT NULL,
                embedding BLOB NOT NULL,
                source_channel TEXT DEFAULT 'api',
                timestamp REAL NOT NULL,
                session_id TEXT,
                tags TEXT DEFAULT '',
                deleted INTEGER DEFAULT 0
            );
            CREATE TABLE sessions (
                id TEXT PRIMARY KEY,
                start_idx INTEGER,
                end_idx INTEGER,
                channel TEXT,
                timestamp REAL,
                summary TEXT
            );
            CREATE TABLE entities (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                type TEXT NOT NULL,
                embedding_idx INTEGER DEFAULT -1
            );
            CREATE TABLE relations (
                src INTEGER NOT NULL,
                tgt INTEGER NOT NULL,
                relation TEXT NOT NULL,
                weight REAL DEFAULT 1.0,
                timestamp REAL,
                FOREIGN KEY (src) REFERENCES entities(id),
                FOREIGN KEY (tgt) REFERENCES entities(id)
            );",
        )
        .unwrap();
        path_str
    }

    /// Insert a memory chunk with a known embedding.
    fn insert_chunk(conn: &Connection, id: i64, text: &str, embedding: &[f32], deleted: i32) {
        let blob: Vec<u8> = embedding.iter().flat_map(|v| v.to_le_bytes()).collect();
        conn.execute(
            "INSERT INTO memory_chunks (id, chunk, embedding, source_channel, timestamp, session_id, tags, deleted)
             VALUES (?1, ?2, ?3, 'api', 1700000000.0, 'sess-1', 'tag1,tag2', ?4)",
            rusqlite::params![id, text, blob, deleted],
        )
        .unwrap();
    }

    fn insert_session(conn: &Connection, id: &str, start: i64, end: i64) {
        conn.execute(
            "INSERT INTO sessions (id, start_idx, end_idx, channel, timestamp, summary)
             VALUES (?1, ?2, ?3, 'discord', 1700000000.0, 'test summary')",
            rusqlite::params![id, start, end],
        )
        .unwrap();
    }

    fn insert_entity(conn: &Connection, id: i64, name: &str, etype: &str) {
        conn.execute(
            "INSERT INTO entities (id, name, type, embedding_idx) VALUES (?1, ?2, ?3, -1)",
            rusqlite::params![id, name, etype],
        )
        .unwrap();
    }

    fn insert_relation(conn: &Connection, src: i64, tgt: i64, rel: &str) {
        conn.execute(
            "INSERT INTO relations (src, tgt, relation, weight, timestamp)
             VALUES (?1, ?2, ?3, 1.0, 1700000000.0)",
            rusqlite::params![src, tgt, rel],
        )
        .unwrap();
    }

    fn make_embedding(dim: usize, seed: f32) -> Vec<f32> {
        (0..dim).map(|i| seed + i as f32 * 0.1).collect()
    }

    // ---------- Test 1: Basic end-to-end migration ----------
    #[test]
    fn test_basic_migration() {
        let dir = TempDir::new().unwrap();
        let db_path = create_test_db(&dir);
        let h5_path = dir.path().join("out.h5");

        let conn = Connection::open(&db_path).unwrap();
        insert_chunk(&conn, 1, "hello world", &make_embedding(8, 1.0), 0);
        insert_chunk(&conn, 2, "goodbye world", &make_embedding(8, 2.0), 0);
        insert_session(&conn, "s1", 0, 1);
        insert_entity(&conn, 1, "Alice", "person");
        insert_relation(&conn, 1, 1, "self");
        drop(conn);

        let data =
            sqlite_reader::read_sqlite(&db_path, false, None, &SchemaConfig::default()).unwrap();
        let opts = hdf5_writer::WriteOptions {
            agent_id: "test-agent".into(),
            embedder: "test-embed".into(),
            compression: false,
            compression_level: 4,
            float16: false,
        };
        hdf5_writer::write_hdf5(h5_path.to_str().unwrap(), &data, &opts).unwrap();

        let summary =
            validate::validate_hdf5(h5_path.to_str().unwrap(), &data, false, false).unwrap();
        assert_eq!(summary.chunks, 2);
        assert_eq!(summary.sessions, 1);
        assert_eq!(summary.entities, 1);
        assert_eq!(summary.relations, 1);
        assert_eq!(summary.embedding_dim, 8);
    }

    // ---------- Test 2: Skip deleted rows ----------
    #[test]
    fn test_skip_deleted() {
        let dir = TempDir::new().unwrap();
        let db_path = create_test_db(&dir);
        let h5_path = dir.path().join("out.h5");

        let conn = Connection::open(&db_path).unwrap();
        insert_chunk(&conn, 1, "active", &make_embedding(4, 1.0), 0);
        insert_chunk(&conn, 2, "deleted", &make_embedding(4, 2.0), 1);
        insert_chunk(&conn, 3, "also active", &make_embedding(4, 3.0), 0);
        drop(conn);

        let data =
            sqlite_reader::read_sqlite(&db_path, true, None, &SchemaConfig::default()).unwrap();
        assert_eq!(data.chunks.len(), 2);

        let opts = hdf5_writer::WriteOptions {
            agent_id: "t".into(),
            embedder: "t".into(),
            compression: false,
            compression_level: 4,
            float16: false,
        };
        hdf5_writer::write_hdf5(h5_path.to_str().unwrap(), &data, &opts).unwrap();

        let summary =
            validate::validate_hdf5(h5_path.to_str().unwrap(), &data, false, false).unwrap();
        assert_eq!(summary.chunks, 2);
    }

    // ---------- Test 3: Include deleted rows ----------
    #[test]
    fn test_include_deleted() {
        let dir = TempDir::new().unwrap();
        let db_path = create_test_db(&dir);

        let conn = Connection::open(&db_path).unwrap();
        insert_chunk(&conn, 1, "active", &make_embedding(4, 1.0), 0);
        insert_chunk(&conn, 2, "deleted", &make_embedding(4, 2.0), 1);
        drop(conn);

        let data =
            sqlite_reader::read_sqlite(&db_path, false, None, &SchemaConfig::default()).unwrap();
        assert_eq!(data.chunks.len(), 2);
    }

    // ---------- Test 4: Auto-detect embedding dimension ----------
    #[test]
    fn test_auto_detect_dim() {
        let dir = TempDir::new().unwrap();
        let db_path = create_test_db(&dir);

        let conn = Connection::open(&db_path).unwrap();
        insert_chunk(&conn, 1, "test", &make_embedding(16, 0.5), 0);
        drop(conn);

        let data =
            sqlite_reader::read_sqlite(&db_path, false, None, &SchemaConfig::default()).unwrap();
        assert_eq!(data.embedding_dim, 16);
    }

    // ---------- Test 5: Manual embedding dimension ----------
    #[test]
    fn test_manual_dim() {
        let dir = TempDir::new().unwrap();
        let db_path = create_test_db(&dir);

        let conn = Connection::open(&db_path).unwrap();
        insert_chunk(&conn, 1, "test", &make_embedding(16, 0.5), 0);
        drop(conn);

        let data =
            sqlite_reader::read_sqlite(&db_path, false, Some(8), &SchemaConfig::default()).unwrap();
        assert_eq!(data.embedding_dim, 8);
        // Embedding truncated to dim 8
        assert_eq!(data.chunks[0].embedding.len(), 8);
    }

    // ---------- Test 6: Float16 conversion ----------
    #[test]
    fn test_float16_conversion() {
        let dir = TempDir::new().unwrap();
        let db_path = create_test_db(&dir);
        let h5_path = dir.path().join("out.h5");

        let conn = Connection::open(&db_path).unwrap();
        let emb = vec![1.0f32, 2.5, -0.5, 3.125];
        insert_chunk(&conn, 1, "test", &emb, 0);
        drop(conn);

        let data =
            sqlite_reader::read_sqlite(&db_path, false, None, &SchemaConfig::default()).unwrap();
        let opts = hdf5_writer::WriteOptions {
            agent_id: "t".into(),
            embedder: "t".into(),
            compression: false,
            compression_level: 4,
            float16: true,
        };
        hdf5_writer::write_hdf5(h5_path.to_str().unwrap(), &data, &opts).unwrap();

        // Content-validate with the float16 tolerance enabled.
        let summary =
            validate::validate_hdf5(h5_path.to_str().unwrap(), &data, true, true).unwrap();
        assert_eq!(summary.chunks, 1);

        // Verify float16 values are within tolerance
        for &v in &emb {
            let f16 = half::f16::from_f32(v);
            let roundtrip = f16.to_f32();
            assert!(
                (v - roundtrip).abs() < 0.01,
                "f16 roundtrip too lossy for {v}"
            );
        }
    }

    // ---------- Test 7: Compression produces valid file ----------
    #[test]
    fn test_compression() {
        let dir = TempDir::new().unwrap();
        let db_path = create_test_db(&dir);
        let h5_compressed = dir.path().join("compressed.h5");
        let h5_uncompressed = dir.path().join("uncompressed.h5");

        let conn = Connection::open(&db_path).unwrap();
        // Insert enough data so compression can be effective
        for i in 0..100 {
            insert_chunk(&conn, i, &format!("chunk {i}"), &make_embedding(32, 0.0), 0);
        }
        drop(conn);

        let data =
            sqlite_reader::read_sqlite(&db_path, false, None, &SchemaConfig::default()).unwrap();

        let opts_compressed = hdf5_writer::WriteOptions {
            agent_id: "t".into(),
            embedder: "t".into(),
            compression: true,
            compression_level: 6,
            float16: false,
        };
        hdf5_writer::write_hdf5(h5_compressed.to_str().unwrap(), &data, &opts_compressed).unwrap();

        let opts_plain = hdf5_writer::WriteOptions {
            agent_id: "t".into(),
            embedder: "t".into(),
            compression: false,
            compression_level: 4,
            float16: false,
        };
        hdf5_writer::write_hdf5(h5_uncompressed.to_str().unwrap(), &data, &opts_plain).unwrap();

        let sz_c = std::fs::metadata(&h5_compressed).unwrap().len();
        let sz_u = std::fs::metadata(&h5_uncompressed).unwrap().len();
        assert!(
            sz_c < sz_u,
            "Compressed ({sz_c}) should be smaller than uncompressed ({sz_u})"
        );
    }

    // ---------- Test 8: Dry run doesn't create file ----------
    #[test]
    fn test_dry_run() {
        let dir = TempDir::new().unwrap();
        let db_path = create_test_db(&dir);
        let h5_path = dir.path().join("should_not_exist.h5");

        let conn = Connection::open(&db_path).unwrap();
        insert_chunk(&conn, 1, "test", &make_embedding(4, 1.0), 0);
        drop(conn);

        // Simulate dry-run: read data but don't write
        let data =
            sqlite_reader::read_sqlite(&db_path, false, None, &SchemaConfig::default()).unwrap();
        assert_eq!(data.chunks.len(), 1);
        assert!(!h5_path.exists());
    }

    // ---------- Test 9: Empty database migration ----------
    #[test]
    fn test_empty_db() {
        let dir = TempDir::new().unwrap();
        let db_path = create_test_db(&dir);
        let h5_path = dir.path().join("out.h5");

        let data =
            sqlite_reader::read_sqlite(&db_path, false, None, &SchemaConfig::default()).unwrap();
        assert_eq!(data.chunks.len(), 0);
        assert_eq!(data.sessions.len(), 0);
        assert_eq!(data.entities.len(), 0);
        assert_eq!(data.relations.len(), 0);

        let opts = hdf5_writer::WriteOptions {
            agent_id: "t".into(),
            embedder: "t".into(),
            compression: false,
            compression_level: 4,
            float16: false,
        };
        hdf5_writer::write_hdf5(h5_path.to_str().unwrap(), &data, &opts).unwrap();

        let summary =
            validate::validate_hdf5(h5_path.to_str().unwrap(), &data, false, false).unwrap();
        assert_eq!(summary.chunks, 0);
    }

    // ---------- Test 10: Large migration (1000 entries) ----------
    #[test]
    fn test_large_migration() {
        let dir = TempDir::new().unwrap();
        let db_path = create_test_db(&dir);
        let h5_path = dir.path().join("out.h5");

        let conn = Connection::open(&db_path).unwrap();
        for i in 0..1000 {
            insert_chunk(
                &conn,
                i,
                &format!("chunk number {i} with some text"),
                &make_embedding(64, i as f32 * 0.01),
                0,
            );
        }
        drop(conn);

        let data =
            sqlite_reader::read_sqlite(&db_path, false, None, &SchemaConfig::default()).unwrap();
        assert_eq!(data.chunks.len(), 1000);

        let opts = hdf5_writer::WriteOptions {
            agent_id: "t".into(),
            embedder: "t".into(),
            compression: false,
            compression_level: 4,
            float16: false,
        };
        hdf5_writer::write_hdf5(h5_path.to_str().unwrap(), &data, &opts).unwrap();

        let summary =
            validate::validate_hdf5(h5_path.to_str().unwrap(), &data, false, false).unwrap();
        assert_eq!(summary.chunks, 1000);
    }

    // ---------- Test 11: Session migration ----------
    #[test]
    fn test_session_migration() {
        let dir = TempDir::new().unwrap();
        let db_path = create_test_db(&dir);
        let h5_path = dir.path().join("out.h5");

        let conn = Connection::open(&db_path).unwrap();
        insert_session(&conn, "session-alpha", 0, 10);
        insert_session(&conn, "session-beta", 11, 20);
        insert_session(&conn, "session-gamma", 21, 30);
        drop(conn);

        let data =
            sqlite_reader::read_sqlite(&db_path, false, None, &SchemaConfig::default()).unwrap();
        assert_eq!(data.sessions.len(), 3);

        let opts = hdf5_writer::WriteOptions {
            agent_id: "t".into(),
            embedder: "t".into(),
            compression: false,
            compression_level: 4,
            float16: false,
        };
        hdf5_writer::write_hdf5(h5_path.to_str().unwrap(), &data, &opts).unwrap();

        let summary =
            validate::validate_hdf5(h5_path.to_str().unwrap(), &data, false, false).unwrap();
        assert_eq!(summary.sessions, 3);
    }

    // ---------- Test 12: Knowledge graph (entities + relations) ----------
    #[test]
    fn test_knowledge_graph_migration() {
        let dir = TempDir::new().unwrap();
        let db_path = create_test_db(&dir);
        let h5_path = dir.path().join("out.h5");

        let conn = Connection::open(&db_path).unwrap();
        insert_entity(&conn, 1, "Alice", "person");
        insert_entity(&conn, 2, "Bob", "person");
        insert_entity(&conn, 3, "Rust", "language");
        insert_relation(&conn, 1, 2, "knows");
        insert_relation(&conn, 1, 3, "uses");
        insert_relation(&conn, 2, 3, "uses");
        drop(conn);

        let data =
            sqlite_reader::read_sqlite(&db_path, false, None, &SchemaConfig::default()).unwrap();
        assert_eq!(data.entities.len(), 3);
        assert_eq!(data.relations.len(), 3);

        let opts = hdf5_writer::WriteOptions {
            agent_id: "t".into(),
            embedder: "t".into(),
            compression: false,
            compression_level: 4,
            float16: false,
        };
        hdf5_writer::write_hdf5(h5_path.to_str().unwrap(), &data, &opts).unwrap();

        let summary =
            validate::validate_hdf5(h5_path.to_str().unwrap(), &data, false, false).unwrap();
        assert_eq!(summary.entities, 3);
        assert_eq!(summary.relations, 3);
    }

    // ---------- Test 13: Validation catches chunk count mismatch ----------
    #[test]
    fn test_validation_catches_count_mismatch() {
        let dir = TempDir::new().unwrap();
        let db_path = create_test_db(&dir);
        let h5_path = dir.path().join("out.h5");

        let conn = Connection::open(&db_path).unwrap();
        insert_chunk(&conn, 1, "test", &make_embedding(4, 1.0), 0);
        drop(conn);

        let data =
            sqlite_reader::read_sqlite(&db_path, false, None, &SchemaConfig::default()).unwrap();
        let opts = hdf5_writer::WriteOptions {
            agent_id: "t".into(),
            embedder: "t".into(),
            compression: false,
            compression_level: 4,
            float16: false,
        };
        hdf5_writer::write_hdf5(h5_path.to_str().unwrap(), &data, &opts).unwrap();

        // Validating against a source with an extra (unwritten) chunk must fail.
        let mut bigger =
            sqlite_reader::read_sqlite(&db_path, false, None, &SchemaConfig::default()).unwrap();
        let mut extra = bigger.chunks[0].clone();
        extra.id = 999;
        bigger.chunks.push(extra);
        let result = validate::validate_hdf5(h5_path.to_str().unwrap(), &bigger, false, false);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("count mismatch"));
    }

    // ---------- Test 14: Metadata attributes are stored ----------
    #[test]
    fn test_metadata_attributes() {
        let dir = TempDir::new().unwrap();
        let db_path = create_test_db(&dir);
        let h5_path = dir.path().join("out.h5");

        let conn = Connection::open(&db_path).unwrap();
        insert_chunk(&conn, 1, "test", &make_embedding(8, 1.0), 0);
        drop(conn);

        let data =
            sqlite_reader::read_sqlite(&db_path, false, None, &SchemaConfig::default()).unwrap();
        let opts = hdf5_writer::WriteOptions {
            agent_id: "my-agent-42".into(),
            embedder: "openai-ada".into(),
            compression: false,
            compression_level: 4,
            float16: false,
        };
        hdf5_writer::write_hdf5(h5_path.to_str().unwrap(), &data, &opts).unwrap();

        let file = clawhdf5::File::open(h5_path.to_str().unwrap()).unwrap();
        let root = file.root();
        let attrs = root.attrs().unwrap();

        match attrs.get("agent_id") {
            Some(clawhdf5_format::type_builders::AttrValue::String(s)) => {
                assert_eq!(s, "my-agent-42");
            }
            other => panic!("Expected String agent_id, got {other:?}"),
        }
        match attrs.get("embedder") {
            Some(clawhdf5_format::type_builders::AttrValue::String(s)) => {
                assert_eq!(s, "openai-ada");
            }
            other => panic!("Expected String embedder, got {other:?}"),
        }
        match attrs.get("embedding_dim") {
            Some(clawhdf5_format::type_builders::AttrValue::I64(d)) => {
                assert_eq!(*d, 8);
            }
            other => panic!("Expected I64 embedding_dim, got {other:?}"),
        }
    }

    // ---------- Test 15: Embedding values roundtrip correctly ----------
    #[test]
    fn test_embedding_roundtrip() {
        let dir = TempDir::new().unwrap();
        let db_path = create_test_db(&dir);
        let h5_path = dir.path().join("out.h5");

        let conn = Connection::open(&db_path).unwrap();
        let emb = vec![0.1, 0.2, 0.3, 0.4];
        insert_chunk(&conn, 1, "test", &emb, 0);
        drop(conn);

        let data =
            sqlite_reader::read_sqlite(&db_path, false, None, &SchemaConfig::default()).unwrap();
        let opts = hdf5_writer::WriteOptions {
            agent_id: "t".into(),
            embedder: "t".into(),
            compression: false,
            compression_level: 4,
            float16: false,
        };
        hdf5_writer::write_hdf5(h5_path.to_str().unwrap(), &data, &opts).unwrap();

        let file = clawhdf5::File::open(h5_path.to_str().unwrap()).unwrap();
        let chunks_group = file.group("chunks").unwrap();
        let emb_ds = chunks_group.dataset("embeddings").unwrap();
        let read_back = emb_ds.read_f32().unwrap();

        assert_eq!(read_back.len(), 4);
        for (a, b) in emb.iter().zip(read_back.iter()) {
            assert!((a - b).abs() < 1e-6, "Embedding mismatch: {a} vs {b}");
        }
    }

    // ---------- Test 16: Full combined migration ----------
    #[test]
    fn test_full_combined_migration() {
        let dir = TempDir::new().unwrap();
        let db_path = create_test_db(&dir);
        let h5_path = dir.path().join("out.h5");

        let conn = Connection::open(&db_path).unwrap();
        for i in 0..5 {
            insert_chunk(
                &conn,
                i,
                &format!("chunk {i}"),
                &make_embedding(16, i as f32),
                if i == 3 { 1 } else { 0 },
            );
        }
        insert_session(&conn, "s1", 0, 2);
        insert_session(&conn, "s2", 3, 4);
        insert_entity(&conn, 1, "Alice", "person");
        insert_entity(&conn, 2, "Bob", "person");
        insert_relation(&conn, 1, 2, "knows");
        drop(conn);

        // Skip deleted
        let data =
            sqlite_reader::read_sqlite(&db_path, true, None, &SchemaConfig::default()).unwrap();
        assert_eq!(data.chunks.len(), 4); // chunk 3 is deleted

        let opts = hdf5_writer::WriteOptions {
            agent_id: "combined-test".into(),
            embedder: "test-embedder".into(),
            compression: true,
            compression_level: 4,
            float16: false,
        };
        hdf5_writer::write_hdf5(h5_path.to_str().unwrap(), &data, &opts).unwrap();

        let summary =
            validate::validate_hdf5(h5_path.to_str().unwrap(), &data, false, false).unwrap();
        assert_eq!(summary.chunks, 4);
        assert_eq!(summary.sessions, 2);
        assert_eq!(summary.entities, 2);
        assert_eq!(summary.relations, 1);
        assert_eq!(summary.embedding_dim, 16);
    }

    // ---------- Test 17: Validation catches session mismatch ----------
    #[test]
    fn test_validation_session_mismatch() {
        let dir = TempDir::new().unwrap();
        let db_path = create_test_db(&dir);
        let h5_path = dir.path().join("out.h5");

        let conn = Connection::open(&db_path).unwrap();
        insert_session(&conn, "s1", 0, 10);
        drop(conn);

        let data =
            sqlite_reader::read_sqlite(&db_path, false, None, &SchemaConfig::default()).unwrap();
        let opts = hdf5_writer::WriteOptions {
            agent_id: "t".into(),
            embedder: "t".into(),
            compression: false,
            compression_level: 4,
            float16: false,
        };
        hdf5_writer::write_hdf5(h5_path.to_str().unwrap(), &data, &opts).unwrap();

        // Validating against a source whose session content differs must fail.
        let mut tampered =
            sqlite_reader::read_sqlite(&db_path, false, None, &SchemaConfig::default()).unwrap();
        tampered.sessions[0].summary = "DIFFERENT".into();
        let result = validate::validate_hdf5(h5_path.to_str().unwrap(), &tampered, false, false);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("session"));
    }

    // ---------- Real content validation catches corrupt embeddings ----------
    #[test]
    fn test_content_validation_catches_embedding_corruption() {
        let dir = TempDir::new().unwrap();
        let db_path = create_test_db(&dir);
        let h5_path = dir.path().join("out.h5");

        let conn = Connection::open(&db_path).unwrap();
        insert_chunk(&conn, 1, "hello", &make_embedding(8, 1.0), 0);
        drop(conn);

        let data =
            sqlite_reader::read_sqlite(&db_path, false, None, &SchemaConfig::default()).unwrap();
        let opts = hdf5_writer::WriteOptions {
            agent_id: "t".into(),
            embedder: "t".into(),
            compression: false,
            compression_level: 4,
            float16: false,
        };
        hdf5_writer::write_hdf5(h5_path.to_str().unwrap(), &data, &opts).unwrap();

        // A source whose embedding differs (but counts match) must fail validation.
        let mut tampered =
            sqlite_reader::read_sqlite(&db_path, false, None, &SchemaConfig::default()).unwrap();
        tampered.chunks[0].embedding[3] += 9.0;
        let result = validate::validate_hdf5(h5_path.to_str().unwrap(), &tampered, true, false);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("embedding"));
    }

    // ---------- Configurable schema: custom table names ----------
    #[test]
    fn test_configurable_table_names() {
        let dir = TempDir::new().unwrap();
        let db_path = dir.path().join("custom.db");
        let path_str = db_path.to_str().unwrap().to_string();
        let conn = Connection::open(&path_str).unwrap();
        // Chunks live in a differently-named table; the others use defaults.
        conn.execute_batch(
            "CREATE TABLE my_chunks (
                id INTEGER PRIMARY KEY, chunk TEXT, embedding BLOB,
                source_channel TEXT, timestamp REAL, session_id TEXT, tags TEXT, deleted INTEGER
            );
            CREATE TABLE sessions (id TEXT, start_idx INTEGER, end_idx INTEGER, channel TEXT, timestamp REAL, summary TEXT);
            CREATE TABLE entities (id INTEGER, name TEXT, type TEXT, embedding_idx INTEGER);
            CREATE TABLE relations (src INTEGER, tgt INTEGER, relation TEXT, weight REAL, timestamp REAL);",
        )
        .unwrap();
        let blob: Vec<u8> = make_embedding(4, 1.0)
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        conn.execute(
            "INSERT INTO my_chunks VALUES (1, 'hi', ?1, 'api', 1.0, 's', '', 0)",
            rusqlite::params![blob],
        )
        .unwrap();
        drop(conn);

        let mut schema = SchemaConfig::default();
        schema.chunks.table = "my_chunks".into();
        let data = sqlite_reader::read_sqlite(&path_str, false, None, &schema).unwrap();
        assert_eq!(data.chunks.len(), 1);
        assert_eq!(data.chunks[0].chunk, "hi");
        assert_eq!(data.embedding_dim, 4);

        // Counts pass should also honor the custom table name.
        let counts = sqlite_reader::read_counts(&path_str, false, &schema).unwrap();
        assert_eq!(counts.chunks, 1);
    }

    // ---------- Incremental migration appends only new rows ----------
    #[test]
    fn test_incremental_migration() {
        let dir = TempDir::new().unwrap();
        let db_path = create_test_db(&dir);
        let h5_path = dir.path().join("out.h5");
        let cfg = SchemaConfig::default();
        let opts = hdf5_writer::WriteOptions {
            agent_id: "t".into(),
            embedder: "t".into(),
            compression: false,
            compression_level: 4,
            float16: false,
        };

        // First migration: 2 chunks.
        let conn = Connection::open(&db_path).unwrap();
        insert_chunk(&conn, 1, "one", &make_embedding(4, 1.0), 0);
        insert_chunk(&conn, 2, "two", &make_embedding(4, 2.0), 0);
        drop(conn);
        let data = sqlite_reader::read_sqlite(&db_path, false, None, &cfg).unwrap();
        hdf5_writer::write_hdf5(h5_path.to_str().unwrap(), &data, &opts).unwrap();

        // Add two more rows, then migrate incrementally.
        let conn = Connection::open(&db_path).unwrap();
        insert_chunk(&conn, 3, "three", &make_embedding(4, 3.0), 0);
        insert_chunk(&conn, 4, "four", &make_embedding(4, 4.0), 0);
        drop(conn);

        let base = hdf5_reader::read_hdf5(h5_path.to_str().unwrap()).unwrap();
        let max_id = base.chunks.iter().map(|c| c.id).max().unwrap_or(0);
        assert_eq!(max_id, 2);
        let new =
            sqlite_reader::read_sqlite_filtered(&db_path, false, Some(4), &cfg, max_id).unwrap();
        assert_eq!(new.chunks.len(), 2); // only id 3 and 4

        let mut merged = base;
        merged.chunks.extend(new.chunks);
        hdf5_writer::write_hdf5(h5_path.to_str().unwrap(), &merged, &opts).unwrap();

        let final_data = hdf5_reader::read_hdf5(h5_path.to_str().unwrap()).unwrap();
        assert_eq!(final_data.chunks.len(), 4);
        let texts: Vec<&str> = final_data.chunks.iter().map(|c| c.chunk.as_str()).collect();
        assert_eq!(texts, vec!["one", "two", "three", "four"]);
    }
}
