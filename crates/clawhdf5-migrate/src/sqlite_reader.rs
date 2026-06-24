use rusqlite::{Connection, Result as SqlResult};

/// A memory chunk read from SQLite.
#[derive(Debug, Clone)]
pub struct MemoryChunk {
    pub id: i64,
    pub chunk: String,
    pub embedding: Vec<f32>,
    pub source_channel: String,
    pub timestamp: f64,
    pub session_id: String,
    pub tags: String,
    pub deleted: i32,
}

/// A session read from SQLite.
#[derive(Debug, Clone)]
pub struct Session {
    pub id: String,
    pub start_idx: i64,
    pub end_idx: i64,
    pub channel: String,
    pub timestamp: f64,
    pub summary: String,
}

/// An entity read from SQLite.
#[derive(Debug, Clone)]
pub struct Entity {
    pub id: i64,
    pub name: String,
    pub entity_type: String,
    pub embedding_idx: i64,
}

/// A relation read from SQLite.
#[derive(Debug, Clone)]
pub struct Relation {
    pub src: i64,
    pub tgt: i64,
    pub relation: String,
    pub weight: f64,
    pub timestamp: f64,
}

/// All data read from a ZeroClaw SQLite database.
#[derive(Debug)]
pub struct SqliteData {
    pub chunks: Vec<MemoryChunk>,
    pub sessions: Vec<Session>,
    pub entities: Vec<Entity>,
    pub relations: Vec<Relation>,
    pub embedding_dim: usize,
}

/// A table name plus the ordered column names the reader maps by position.
#[derive(Debug, Clone)]
pub struct TableSchema {
    pub table: String,
    pub columns: Vec<&'static str>,
}

/// Configurable mapping from a SQLite layout to the migration's data model.
///
/// Defaults to the ZeroClaw schema; the CLI can override the table names so the
/// tool can migrate databases whose tables are named differently. Column names
/// (and order) are part of the config too, so a library caller can remap them.
#[derive(Debug, Clone)]
pub struct SchemaConfig {
    pub chunks: TableSchema,
    pub sessions: TableSchema,
    pub entities: TableSchema,
    pub relations: TableSchema,
}

impl Default for SchemaConfig {
    fn default() -> Self {
        SchemaConfig {
            chunks: TableSchema {
                table: "memory_chunks".into(),
                columns: vec![
                    "id",
                    "chunk",
                    "embedding",
                    "source_channel",
                    "timestamp",
                    "session_id",
                    "tags",
                    "deleted",
                ],
            },
            sessions: TableSchema {
                table: "sessions".into(),
                columns: vec![
                    "id",
                    "start_idx",
                    "end_idx",
                    "channel",
                    "timestamp",
                    "summary",
                ],
            },
            entities: TableSchema {
                table: "entities".into(),
                columns: vec!["id", "name", "type", "embedding_idx"],
            },
            relations: TableSchema {
                table: "relations".into(),
                columns: vec!["src", "tgt", "relation", "weight", "timestamp"],
            },
        }
    }
}

impl TableSchema {
    fn select(&self, where_clause: &str) -> String {
        format!(
            "SELECT {} FROM {}{}",
            self.columns.join(", "),
            self.table,
            where_clause
        )
    }
}

/// Row counts for each table — a fast pass that does not load row contents.
/// Used for `--dry-run` and progress without buffering the whole database.
#[derive(Debug, Default, Clone, Copy)]
pub struct RowCounts {
    pub chunks: u64,
    pub sessions: u64,
    pub entities: u64,
    pub relations: u64,
}

fn count_rows(conn: &Connection, table: &str, where_clause: &str) -> SqlResult<u64> {
    conn.query_row(
        &format!("SELECT COUNT(*) FROM {table}{where_clause}"),
        [],
        |r| r.get(0),
    )
}

/// Count rows in each table without reading their contents.
pub fn read_counts(
    path: &str,
    skip_deleted: bool,
    config: &SchemaConfig,
) -> Result<RowCounts, Box<dyn std::error::Error>> {
    let conn = Connection::open(path)?;
    let deleted_col = config.chunks.columns.get(7).copied().unwrap_or("deleted");
    let chunk_where = if skip_deleted {
        format!(" WHERE {deleted_col} = 0")
    } else {
        String::new()
    };
    Ok(RowCounts {
        chunks: count_rows(&conn, &config.chunks.table, &chunk_where)?,
        sessions: count_rows(&conn, &config.sessions.table, "")?,
        entities: count_rows(&conn, &config.entities.table, "")?,
        relations: count_rows(&conn, &config.relations.table, "")?,
    })
}

/// Auto-detect embedding dimension from the first chunk's BLOB size.
fn detect_embedding_dim(conn: &Connection, config: &SchemaConfig) -> SqlResult<Option<usize>> {
    let emb_col = config.chunks.columns.get(2).copied().unwrap_or("embedding");
    let mut stmt = conn.prepare(&format!(
        "SELECT {emb_col} FROM {} LIMIT 1",
        config.chunks.table
    ))?;
    let mut rows = stmt.query([])?;
    if let Some(row) = rows.next()? {
        let blob: Vec<u8> = row.get(0)?;
        Ok(Some(blob.len() / 4)) // f32 = 4 bytes
    } else {
        Ok(None)
    }
}

/// Parse a raw byte BLOB into a Vec<f32>.
fn blob_to_f32(blob: &[u8]) -> Vec<f32> {
    blob.chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

/// Read all data from a ZeroClaw SQLite database.
///
/// If `skip_deleted` is true, rows with `deleted=1` are excluded from chunks.
/// If `embedding_dim` is `None`, auto-detect from the first row.
pub fn read_sqlite(
    path: &str,
    skip_deleted: bool,
    embedding_dim: Option<usize>,
    config: &SchemaConfig,
) -> Result<SqliteData, Box<dyn std::error::Error>> {
    read_sqlite_filtered(path, skip_deleted, embedding_dim, config, 0)
}

/// Like [`read_sqlite`] but only reads chunks whose id is greater than
/// `min_chunk_id` (0 = all). Used for incremental migration.
pub fn read_sqlite_filtered(
    path: &str,
    skip_deleted: bool,
    embedding_dim: Option<usize>,
    config: &SchemaConfig,
    min_chunk_id: i64,
) -> Result<SqliteData, Box<dyn std::error::Error>> {
    let conn = Connection::open(path)?;

    let dim = match embedding_dim {
        Some(d) => d,
        None => detect_embedding_dim(&conn, config)?.unwrap_or(0),
    };

    let chunks = read_chunks(&conn, skip_deleted, dim, config, min_chunk_id)?;
    let sessions = read_sessions(&conn, config)?;
    let entities = read_entities(&conn, config)?;
    let relations = read_relations(&conn, config)?;

    Ok(SqliteData {
        chunks,
        sessions,
        entities,
        relations,
        embedding_dim: dim,
    })
}

fn read_chunks(
    conn: &Connection,
    skip_deleted: bool,
    expected_dim: usize,
    config: &SchemaConfig,
    min_chunk_id: i64,
) -> SqlResult<Vec<MemoryChunk>> {
    let id_col = config.chunks.columns.first().copied().unwrap_or("id");
    let deleted_col = config.chunks.columns.get(7).copied().unwrap_or("deleted");
    let mut conds = Vec::new();
    if skip_deleted {
        conds.push(format!("{deleted_col} = 0"));
    }
    if min_chunk_id > 0 {
        conds.push(format!("{id_col} > {min_chunk_id}"));
    }
    let where_clause = if conds.is_empty() {
        String::new()
    } else {
        format!(" WHERE {}", conds.join(" AND "))
    };
    let sql = config.chunks.select(&where_clause);

    let mut stmt = conn.prepare(&sql)?;
    let rows = stmt.query_map([], |row| {
        let blob: Vec<u8> = row.get(2)?;
        let mut embedding = blob_to_f32(&blob);

        // Validate/truncate to expected dimension
        if expected_dim > 0 {
            embedding.truncate(expected_dim);
        }

        Ok(MemoryChunk {
            id: row.get(0)?,
            chunk: row.get(1)?,
            embedding,
            source_channel: row.get::<_, Option<String>>(3)?.unwrap_or_default(),
            timestamp: row.get(4)?,
            session_id: row.get::<_, Option<String>>(5)?.unwrap_or_default(),
            tags: row.get::<_, Option<String>>(6)?.unwrap_or_default(),
            deleted: row.get(7)?,
        })
    })?;

    rows.collect()
}

fn read_sessions(conn: &Connection, config: &SchemaConfig) -> SqlResult<Vec<Session>> {
    let mut stmt = conn.prepare(&config.sessions.select(""))?;
    let rows = stmt.query_map([], |row| {
        Ok(Session {
            id: row.get(0)?,
            start_idx: row.get::<_, Option<i64>>(1)?.unwrap_or(0),
            end_idx: row.get::<_, Option<i64>>(2)?.unwrap_or(0),
            channel: row.get::<_, Option<String>>(3)?.unwrap_or_default(),
            timestamp: row.get::<_, Option<f64>>(4)?.unwrap_or(0.0),
            summary: row.get::<_, Option<String>>(5)?.unwrap_or_default(),
        })
    })?;
    rows.collect()
}

fn read_entities(conn: &Connection, config: &SchemaConfig) -> SqlResult<Vec<Entity>> {
    let mut stmt = conn.prepare(&config.entities.select(""))?;
    let rows = stmt.query_map([], |row| {
        Ok(Entity {
            id: row.get(0)?,
            name: row.get(1)?,
            entity_type: row.get(2)?,
            embedding_idx: row.get::<_, Option<i64>>(3)?.unwrap_or(-1),
        })
    })?;
    rows.collect()
}

fn read_relations(conn: &Connection, config: &SchemaConfig) -> SqlResult<Vec<Relation>> {
    let mut stmt = conn.prepare(&config.relations.select(""))?;
    let rows = stmt.query_map([], |row| {
        Ok(Relation {
            src: row.get(0)?,
            tgt: row.get(1)?,
            relation: row.get(2)?,
            weight: row.get::<_, Option<f64>>(3)?.unwrap_or(1.0),
            timestamp: row.get::<_, Option<f64>>(4)?.unwrap_or(0.0),
        })
    })?;
    rows.collect()
}
