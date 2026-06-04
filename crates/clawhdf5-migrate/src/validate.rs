use crate::hdf5_reader::read_hdf5;
use crate::sqlite_reader::SqliteData;

type BoxErr = Box<dyn std::error::Error>;

/// Summary of a migration validation.
#[derive(Debug)]
pub struct ValidationSummary {
    pub chunks: u64,
    pub sessions: u64,
    pub entities: u64,
    pub relations: u64,
    pub embedding_dim: u64,
    /// Number of rows whose full content was compared against the source.
    pub rows_checked: u64,
}

/// Validate a migrated HDF5 file against the source data.
///
/// Reads the written file back and compares actual content — chunk text,
/// embeddings, and every session/entity/relation field — to the source, not
/// just the row counts. When `full` is false a representative sample of chunk
/// rows is content-checked (counts and all other groups are always checked in
/// full); when `full` is true every chunk row is compared too. `float16` widens
/// the embedding tolerance to allow for half-precision quantization.
pub fn validate_hdf5(
    path: &str,
    source: &SqliteData,
    full: bool,
    float16: bool,
) -> Result<ValidationSummary, BoxErr> {
    let got = read_hdf5(path)?;

    // ---- Counts ----
    check_count("chunk", got.chunks.len(), source.chunks.len())?;
    check_count("session", got.sessions.len(), source.sessions.len())?;
    check_count("entity", got.entities.len(), source.entities.len())?;
    check_count("relation", got.relations.len(), source.relations.len())?;
    if got.embedding_dim != source.embedding_dim {
        return Err(format!(
            "embedding_dim mismatch: HDF5 has {}, source has {}",
            got.embedding_dim, source.embedding_dim
        )
        .into());
    }

    // ---- Chunk content (sampled or full) ----
    let (emb_abs, emb_rel) = if float16 { (1e-2, 1e-2) } else { (1e-4, 0.0) };
    let mut rows_checked = 0u64;
    for i in sample_indices(source.chunks.len(), full) {
        let (s, g) = (&source.chunks[i], &got.chunks[i]);
        if s.id != g.id {
            return Err(field_err("chunk", i, "id", s.id, g.id));
        }
        if s.chunk != g.chunk {
            return Err(format!(
                "chunk[{i}].text mismatch: source {:?}, HDF5 {:?}",
                truncate(&s.chunk),
                truncate(&g.chunk)
            )
            .into());
        }
        if s.session_id != g.session_id || s.source_channel != g.source_channel || s.tags != g.tags
        {
            return Err(format!("chunk[{i}] string field mismatch").into());
        }
        if s.deleted != g.deleted {
            return Err(field_err("chunk", i, "deleted", s.deleted, g.deleted));
        }
        if s.embedding.len() != g.embedding.len() {
            return Err(format!(
                "chunk[{i}] embedding length mismatch: {} vs {}",
                s.embedding.len(),
                g.embedding.len()
            )
            .into());
        }
        for (k, (&a, &b)) in s.embedding.iter().zip(g.embedding.iter()).enumerate() {
            if (a - b).abs() > emb_abs + emb_rel * a.abs() {
                return Err(format!(
                    "chunk[{i}].embedding[{k}] mismatch: source {a}, HDF5 {b}"
                )
                .into());
            }
        }
        rows_checked += 1;
    }

    // ---- Other groups (always full — they are small) ----
    for (i, (s, g)) in source.sessions.iter().zip(got.sessions.iter()).enumerate() {
        if s.id != g.id
            || s.start_idx != g.start_idx
            || s.end_idx != g.end_idx
            || s.channel != g.channel
            || s.summary != g.summary
        {
            return Err(format!("session[{i}] mismatch").into());
        }
        rows_checked += 1;
    }
    for (i, (s, g)) in source.entities.iter().zip(got.entities.iter()).enumerate() {
        if s.id != g.id
            || s.name != g.name
            || s.entity_type != g.entity_type
            || s.embedding_idx != g.embedding_idx
        {
            return Err(format!("entity[{i}] mismatch").into());
        }
        rows_checked += 1;
    }
    for (i, (s, g)) in source.relations.iter().zip(got.relations.iter()).enumerate() {
        if s.src != g.src || s.tgt != g.tgt || s.relation != g.relation {
            return Err(format!("relation[{i}] mismatch").into());
        }
        rows_checked += 1;
    }

    Ok(ValidationSummary {
        chunks: got.chunks.len() as u64,
        sessions: got.sessions.len() as u64,
        entities: got.entities.len() as u64,
        relations: got.relations.len() as u64,
        embedding_dim: got.embedding_dim as u64,
        rows_checked,
    })
}

fn check_count(kind: &str, got: usize, expected: usize) -> Result<(), BoxErr> {
    if got != expected {
        return Err(format!("{kind} count mismatch: HDF5 has {got}, source has {expected}").into());
    }
    Ok(())
}

fn field_err<T: std::fmt::Display>(kind: &str, i: usize, field: &str, s: T, g: T) -> BoxErr {
    format!("{kind}[{i}].{field} mismatch: source {s}, HDF5 {g}").into()
}

fn truncate(s: &str) -> String {
    if s.len() <= 40 {
        s.to_string()
    } else {
        format!("{}…", &s[..40])
    }
}

/// Indices of chunk rows to content-check. Full = all; otherwise a spread of
/// representative rows (first/last and evenly-spaced interior samples).
fn sample_indices(n: usize, full: bool) -> Vec<usize> {
    if n == 0 {
        return Vec::new();
    }
    if full || n <= 16 {
        return (0..n).collect();
    }
    let mut idx: Vec<usize> = (0..16).map(|k| k * (n - 1) / 15).collect();
    idx.dedup();
    idx
}
