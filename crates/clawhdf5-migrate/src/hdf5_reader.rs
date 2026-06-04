//! Read a migration HDF5 file back into the in-memory data model.
//!
//! Used to verify migrated content (real validation) and to merge new rows into
//! an existing output (incremental migration). Mirrors the layout produced by
//! [`crate::hdf5_writer`].

use clawhdf5::reader::{File, Group};
use clawhdf5_format::type_builders::AttrValue;

use crate::sqlite_reader::{Entity, MemoryChunk, Relation, Session, SqliteData};

type BoxErr = Box<dyn std::error::Error>;

fn read_strings(group: &Group<'_>, name: &str) -> Result<Vec<String>, BoxErr> {
    Ok(group.dataset(name)?.read_string()?)
}

fn read_i64s(group: &Group<'_>, name: &str) -> Result<Vec<i64>, BoxErr> {
    Ok(group.dataset(name)?.read_i64()?)
}

fn read_f64s(group: &Group<'_>, name: &str) -> Result<Vec<f64>, BoxErr> {
    Ok(group.dataset(name)?.read_f64()?)
}

/// Read the embeddings dataset as a flat `Vec<f32>` of `n * dim` values,
/// handling both f32 and (lossy) f16 storage.
fn read_embeddings_flat(group: &Group<'_>) -> Result<Vec<f32>, BoxErr> {
    Ok(group.dataset("embeddings")?.read_f32()?)
}

/// Read a migration HDF5 file into a [`SqliteData`].
pub fn read_hdf5(path: &str) -> Result<SqliteData, BoxErr> {
    let file = File::open(path)?;

    let embedding_dim = match file.root().attrs()?.get("embedding_dim") {
        Some(AttrValue::I64(d)) => *d as usize,
        _ => 0,
    };

    let chunks = read_chunks(&file, embedding_dim)?;
    let sessions = read_sessions(&file)?;
    let entities = read_entities(&file)?;
    let relations = read_relations(&file)?;

    Ok(SqliteData {
        chunks,
        sessions,
        entities,
        relations,
        embedding_dim,
    })
}

fn read_chunks(file: &File, dim: usize) -> Result<Vec<MemoryChunk>, BoxErr> {
    let g = file.group("chunks")?;
    let count = group_count(&g)?;
    if count == 0 {
        return Ok(Vec::new());
    }
    let ids = read_i64s(&g, "id")?;
    let texts = read_strings(&g, "text")?;
    let channels = read_strings(&g, "source_channel")?;
    let timestamps = read_f64s(&g, "timestamp")?;
    let session_ids = read_strings(&g, "session_id")?;
    let tags = read_strings(&g, "tags")?;
    let deleted = g.dataset("deleted")?.read_i32()?;
    let emb_flat = read_embeddings_flat(&g)?;
    let dim = dim.max(1);

    let mut chunks = Vec::with_capacity(ids.len());
    for (i, &id) in ids.iter().enumerate() {
        let embedding = emb_flat
            .get(i * dim..(i + 1) * dim)
            .map(|s| s.to_vec())
            .unwrap_or_default();
        chunks.push(MemoryChunk {
            id,
            chunk: texts.get(i).cloned().unwrap_or_default(),
            embedding,
            source_channel: channels.get(i).cloned().unwrap_or_default(),
            timestamp: timestamps.get(i).copied().unwrap_or(0.0),
            session_id: session_ids.get(i).cloned().unwrap_or_default(),
            tags: tags.get(i).cloned().unwrap_or_default(),
            deleted: deleted.get(i).copied().unwrap_or(0),
        });
    }
    Ok(chunks)
}

fn read_sessions(file: &File) -> Result<Vec<Session>, BoxErr> {
    let g = file.group("sessions")?;
    if group_count(&g)? == 0 {
        return Ok(Vec::new());
    }
    let ids = read_strings(&g, "id")?;
    let starts = read_i64s(&g, "start_idx")?;
    let ends = read_i64s(&g, "end_idx")?;
    let channels = read_strings(&g, "channel")?;
    let timestamps = read_f64s(&g, "timestamp")?;
    let summaries = read_strings(&g, "summary")?;
    Ok((0..ids.len())
        .map(|i| Session {
            id: ids[i].clone(),
            start_idx: starts.get(i).copied().unwrap_or(0),
            end_idx: ends.get(i).copied().unwrap_or(0),
            channel: channels.get(i).cloned().unwrap_or_default(),
            timestamp: timestamps.get(i).copied().unwrap_or(0.0),
            summary: summaries.get(i).cloned().unwrap_or_default(),
        })
        .collect())
}

fn read_entities(file: &File) -> Result<Vec<Entity>, BoxErr> {
    let g = file.group("entities")?;
    if group_count(&g)? == 0 {
        return Ok(Vec::new());
    }
    let ids = read_i64s(&g, "id")?;
    let names = read_strings(&g, "name")?;
    let types = read_strings(&g, "type")?;
    let emb_idxs = read_i64s(&g, "embedding_idx")?;
    Ok((0..ids.len())
        .map(|i| Entity {
            id: ids[i],
            name: names.get(i).cloned().unwrap_or_default(),
            entity_type: types.get(i).cloned().unwrap_or_default(),
            embedding_idx: emb_idxs.get(i).copied().unwrap_or(-1),
        })
        .collect())
}

fn read_relations(file: &File) -> Result<Vec<Relation>, BoxErr> {
    let g = file.group("relations")?;
    if group_count(&g)? == 0 {
        return Ok(Vec::new());
    }
    let srcs = read_i64s(&g, "src")?;
    let tgts = read_i64s(&g, "tgt")?;
    let rels = read_strings(&g, "relation")?;
    let weights = read_f64s(&g, "weight")?;
    let timestamps = read_f64s(&g, "timestamp")?;
    Ok((0..srcs.len())
        .map(|i| Relation {
            src: srcs[i],
            tgt: tgts.get(i).copied().unwrap_or(0),
            relation: rels.get(i).cloned().unwrap_or_default(),
            weight: weights.get(i).copied().unwrap_or(1.0),
            timestamp: timestamps.get(i).copied().unwrap_or(0.0),
        })
        .collect())
}

fn group_count(group: &Group<'_>) -> Result<u64, BoxErr> {
    match group.attrs()?.get("count") {
        Some(AttrValue::I64(n)) => Ok(*n as u64),
        _ => Ok(0),
    }
}
