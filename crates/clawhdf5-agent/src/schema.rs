//! HDF5 schema creation and validation.
//!
//! Handles building the HDF5 file structure from in-memory caches and
//! reading/validating existing files.

use clawhdf5::AttrValue;
use clawhdf5::FillTime;
use clawhdf5_format::datatype::{CharacterSet, Datatype, StringPadding};

use crate::MemoryConfig;
use crate::MemoryError;
use crate::cache::MemoryCache;
use crate::knowledge::KnowledgeCache;
use crate::session::SessionCache;

pub const SCHEMA_VERSION: &str = "1.0";
pub const ZEROCLAW_VERSION: &str = "0.8.0";

/// Build a complete HDF5 file from the in-memory state.
pub fn build_hdf5_file(
    config: &MemoryConfig,
    cache: &MemoryCache,
    sessions: &SessionCache,
    knowledge: &KnowledgeCache,
) -> Result<Vec<u8>, MemoryError> {
    let mut builder = clawhdf5::FileBuilder::new();

    // /meta group with schema attributes
    let mut meta = builder.create_group("meta");
    meta.set_attr("schema_version", AttrValue::String(SCHEMA_VERSION.into()));
    meta.set_attr("created_at", AttrValue::String(config.created_at.clone()));
    meta.set_attr("agent_id", AttrValue::String(config.agent_id.clone()));
    meta.set_attr("embedder", AttrValue::String(config.embedder.clone()));
    meta.set_attr("embedding_dim", AttrValue::I64(config.embedding_dim as i64));
    meta.set_attr("chunk_size", AttrValue::I64(config.chunk_size as i64));
    meta.set_attr("overlap", AttrValue::I64(config.overlap as i64));
    meta.set_attr(
        "edgehdf5_version",
        AttrValue::String(ZEROCLAW_VERSION.into()),
    );
    // P2.4 Finding 1: commit a Merkle root over ALL persisted content (memory,
    // sessions, knowledge graph) so read-side tampering is caught end-to-end.
    // Empty stores carry no root (nothing to protect); see `integrity` module.
    #[cfg(feature = "integrity")]
    if let Some(root_hex) = crate::integrity::content_root_hex(cache, sessions, knowledge) {
        meta.set_attr(
            crate::integrity::MERKLE_ROOT_ATTR,
            AttrValue::String(root_hex),
        );
    }
    // Need at least one dataset in the group for it to be a proper group
    meta.create_dataset("_marker").with_u8_data(&[1]).compact();
    let finished_meta = meta.finish();
    builder.add_group(finished_meta);

    // /memory group
    build_memory_group(&mut builder, config, cache)?;

    // /sessions group
    build_sessions_group(&mut builder, sessions)?;

    // /knowledge_graph group
    build_knowledge_group(&mut builder, knowledge)?;

    builder
        .finish()
        .map_err(|e| MemoryError::Hdf5(e.to_string()))
}

fn build_memory_group(
    builder: &mut clawhdf5::FileBuilder,
    config: &MemoryConfig,
    cache: &MemoryCache,
) -> Result<(), MemoryError> {
    let mut group = builder.create_group("memory");

    // chunks: fixed-length string array
    write_string_dataset(&mut group, "chunks", &cache.chunks);

    // embeddings: f32 [N x D]
    let n = cache.embeddings.len() as u64;
    let d = cache.embedding_dim as u64;
    let flat = cache.flat_embeddings();
    {
        let ds = group
            .create_dataset("embeddings")
            .with_f32_data(&flat)
            .with_shape(&[n, d]);

        // Chunk size tuning: target ~256KB per chunk for optimal I/O
        if n > 0 && d > 0 {
            let target_chunk_bytes: u64 = 256 * 1024;
            let rows_per_chunk = (target_chunk_bytes / (d * 4)).max(1).min(n);
            ds.with_chunks(&[rows_per_chunk, d]);

            // Compression: Zstd for embeddings — faster than deflate at same ratio.
            // Shuffle is applied automatically (auto-shuffle pre-filter).
            if config.compression {
                let level = if config.compression_level > 0 {
                    config.compression_level.min(22)
                } else {
                    3 // Zstd level 3: fast + good ratio for f32 embeddings
                };
                ds.with_zstd(level);
            }
        }

        // Skip fill-value initialization — embeddings are fully written
        ds.fill_time(FillTime::Never);
        // Page-aligned for sequential scans
        ds.align(4096);
    }

    // source_channel: fixed-length string array
    write_string_dataset(&mut group, "source_channel", &cache.source_channels);

    // timestamps: f64 array
    group
        .create_dataset("timestamps")
        .with_f64_data(&cache.timestamps)
        .fill_time(FillTime::Never);

    // session_ids: fixed-length string array (auto-compressed when large)
    write_string_dataset(&mut group, "session_ids", &cache.session_ids);

    // tags: fixed-length string array (auto-compressed when large)
    write_string_dataset(&mut group, "tags", &cache.tags);

    // tombstones: u8 array — use compact if small
    {
        let ds = group
            .create_dataset("tombstones")
            .with_u8_data(&cache.tombstones);
        if cache.tombstones.len() <= 65536 {
            ds.compact();
        }
        ds.fill_time(FillTime::Never);
    }

    // norms: f32 array (pre-computed L2 norms)
    group
        .create_dataset("norms")
        .with_f32_data(&cache.norms)
        .fill_time(FillTime::Never);

    // activation_weights: f32 array (Hebbian activation weights)
    group
        .create_dataset("activation_weights")
        .with_f32_data(&cache.activation_weights)
        .fill_time(FillTime::Never);

    let finished = group.finish();
    builder.add_group(finished);
    Ok(())
}

fn build_sessions_group(
    builder: &mut clawhdf5::FileBuilder,
    sessions: &SessionCache,
) -> Result<(), MemoryError> {
    let mut group = builder.create_group("sessions");

    let ids: Vec<String> = sessions.entries.iter().map(|e| e.id.clone()).collect();
    write_string_dataset(&mut group, "ids", &ids);

    let start_idxs: Vec<i64> = sessions
        .entries
        .iter()
        .map(|e| e.start_idx as i64)
        .collect();
    group
        .create_dataset("start_idxs")
        .with_i64_data(&start_idxs);

    let end_idxs: Vec<i64> = sessions.entries.iter().map(|e| e.end_idx as i64).collect();
    group.create_dataset("end_idxs").with_i64_data(&end_idxs);

    let channels: Vec<String> = sessions.entries.iter().map(|e| e.channel.clone()).collect();
    write_string_dataset(&mut group, "channels", &channels);

    let timestamps: Vec<f64> = sessions.entries.iter().map(|e| e.ts).collect();
    group
        .create_dataset("timestamps")
        .with_f64_data(&timestamps);

    write_string_dataset(&mut group, "summaries", &sessions.summaries);

    let finished = group.finish();
    builder.add_group(finished);
    Ok(())
}

fn build_knowledge_group(
    builder: &mut clawhdf5::FileBuilder,
    knowledge: &KnowledgeCache,
) -> Result<(), MemoryError> {
    let mut group = builder.create_group("knowledge_graph");

    // Entities
    let entity_ids: Vec<i64> = knowledge.entities.iter().map(|e| e.id as i64).collect();
    group
        .create_dataset("entity_ids")
        .with_i64_data(&entity_ids);

    let entity_names: Vec<String> = knowledge.entities.iter().map(|e| e.name.clone()).collect();
    write_string_dataset(&mut group, "entity_names", &entity_names);

    let entity_types: Vec<String> = knowledge
        .entities
        .iter()
        .map(|e| e.entity_type.clone())
        .collect();
    write_string_dataset(&mut group, "entity_types", &entity_types);

    let emb_idxs: Vec<i64> = knowledge.entities.iter().map(|e| e.embedding_idx).collect();
    group
        .create_dataset("entity_emb_idxs")
        .with_i64_data(&emb_idxs);

    // Relations
    let rel_srcs: Vec<i64> = knowledge.relations.iter().map(|r| r.src as i64).collect();
    group
        .create_dataset("relation_srcs")
        .with_i64_data(&rel_srcs);

    let rel_tgts: Vec<i64> = knowledge.relations.iter().map(|r| r.tgt as i64).collect();
    group
        .create_dataset("relation_tgts")
        .with_i64_data(&rel_tgts);

    let rel_types: Vec<String> = knowledge
        .relations
        .iter()
        .map(|r| r.relation.clone())
        .collect();
    write_string_dataset(&mut group, "relation_types", &rel_types);

    let rel_weights: Vec<f32> = knowledge.relations.iter().map(|r| r.weight).collect();
    group
        .create_dataset("relation_weights")
        .with_f32_data(&rel_weights);

    let rel_ts: Vec<f64> = knowledge.relations.iter().map(|r| r.ts).collect();
    group.create_dataset("relation_ts").with_f64_data(&rel_ts);

    // Aliases
    if !knowledge.alias_strings.is_empty() {
        write_string_dataset(&mut group, "alias_strings", &knowledge.alias_strings);
        group
            .create_dataset("alias_entity_ids")
            .with_i64_data(&knowledge.alias_entity_ids);
    }

    let finished = group.finish();
    builder.add_group(finished);
    Ok(())
}

/// Write a string array as a fixed-length string dataset.
///
/// Uses `Datatype::String` with NullPad encoding. Each string is padded
/// to the length of the longest string in the array.
///
/// When `compress` is true, uses chunked storage with deflate(6) —
/// NullPad strings have high redundancy and compress very well.
/// Payload size (bytes) at or above which a fixed-length string dataset is
/// stored chunked + deflate-compressed. Below this, the chunk B-tree/heap
/// overhead outweighs the savings, so the data is left contiguous.
const STRING_COMPRESS_THRESHOLD: usize = 4096;

fn write_string_dataset(
    group: &mut clawhdf5_format::type_builders::GroupBuilder,
    name: &str,
    strings: &[String],
) {
    if strings.is_empty() {
        // Empty dataset: use 1-byte string type with no data
        let dtype = Datatype::String {
            size: 1,
            padding: StringPadding::NullPad,
            charset: CharacterSet::Utf8,
        };
        group
            .create_dataset(name)
            .with_compound_data(dtype, vec![], 0);
        return;
    }

    let max_len = strings.iter().map(|s| s.len()).max().unwrap_or(0).max(1);
    let mut raw = Vec::with_capacity(strings.len() * max_len);
    for s in strings {
        let mut bytes = s.as_bytes().to_vec();
        bytes.resize(max_len, 0);
        raw.extend_from_slice(&bytes);
    }
    let raw_len = raw.len();

    let dtype = Datatype::String {
        size: max_len as u32,
        padding: StringPadding::NullPad,
        charset: CharacterSet::Utf8,
    };
    let ds = group
        .create_dataset(name)
        .with_compound_data(dtype, raw, strings.len() as u64);

    // Fixed-length NullPad strings have high redundancy (padding + repeated
    // content), so deflate pays off once the payload is large enough to absorb
    // the chunking overhead. Fixed-length string datasets are chunkable like
    // any other fixed-size datatype.
    if strings.len() > 1 && raw_len >= STRING_COMPRESS_THRESHOLD {
        // Target ~64KB chunks for string data.
        let elem_size = max_len as u64;
        let target_chunk = 64 * 1024;
        let rows_per_chunk = (target_chunk / elem_size).max(1).min(strings.len() as u64);
        ds.with_chunks(&[rows_per_chunk]);
        ds.with_deflate(6);
    }
}

/// Validate an HDF5 file has the correct schema and load all data.
///
/// Non-strict: a file with no `_merkle_root` attribute (older writers, or the
/// `integrity` feature disabled) loads unverified for backward compatibility.
/// Use [`validate_and_load_strict`] to reject such files.
pub fn validate_and_load(
    file: &clawhdf5::File,
) -> Result<(MemoryConfig, MemoryCache, SessionCache, KnowledgeCache), MemoryError> {
    validate_and_load_impl(file, false)
}

/// Like [`validate_and_load`], but fail-closed when the file carries no
/// content Merkle root (P2.4 Finding 1: closes the strip-to-downgrade attack
/// where an adversary deletes `_merkle_root` to force an unverified load).
pub fn validate_and_load_strict(
    file: &clawhdf5::File,
) -> Result<(MemoryConfig, MemoryCache, SessionCache, KnowledgeCache), MemoryError> {
    validate_and_load_impl(file, true)
}

fn validate_and_load_impl(
    file: &clawhdf5::File,
    strict: bool,
) -> Result<(MemoryConfig, MemoryCache, SessionCache, KnowledgeCache), MemoryError> {
    // `strict` only affects the integrity path; keep the binding used even when
    // the feature is compiled out.
    #[cfg(not(feature = "integrity"))]
    let _ = strict;
    // Read /meta group attributes
    let meta = file
        .group("meta")
        .map_err(|e| MemoryError::Schema(format!("missing /meta group: {e}")))?;
    let attrs = meta
        .attrs()
        .map_err(|e| MemoryError::Schema(format!("cannot read /meta attrs: {e}")))?;

    let schema_version = match attrs.get("schema_version") {
        Some(AttrValue::String(s)) => s.clone(),
        _ => return Err(MemoryError::Schema("missing schema_version attr".into())),
    };
    if schema_version != SCHEMA_VERSION {
        return Err(MemoryError::Schema(format!(
            "schema version mismatch: expected {SCHEMA_VERSION}, got {schema_version}"
        )));
    }

    let created_at = extract_string_attr(&attrs, "created_at")?;
    let agent_id = extract_string_attr(&attrs, "agent_id")?;
    let embedder = extract_string_attr(&attrs, "embedder")?;
    let embedding_dim = extract_i64_attr(&attrs, "embedding_dim")? as usize;
    let chunk_size = extract_i64_attr(&attrs, "chunk_size")? as usize;
    let overlap = extract_i64_attr(&attrs, "overlap")? as usize;

    let config = MemoryConfig {
        path: std::path::PathBuf::new(), // will be set by caller
        agent_id,
        embedder,
        embedding_dim,
        chunk_size,
        overlap,
        float16: false,
        compression: false,
        compression_level: 0,
        compact_threshold: 0.3,
        hebbian_boost: 0.15,
        decay_factor: 0.98,
        created_at,
        wal_enabled: true,
        wal_max_entries: 500,
    };

    // Load /memory group
    let memory_cache = load_memory_group(file, embedding_dim)?;

    // Load /sessions group
    let session_cache = load_sessions_group(file)?;

    // Load /knowledge_graph group
    let knowledge_cache = load_knowledge_group(file)?;

    // P2.4 Finding 1: verify the content Merkle root over ALL loaded groups
    // (fail-closed on mismatch; `strict` also rejects a missing root). Done
    // after every group is loaded so the root covers the full persisted state.
    #[cfg(feature = "integrity")]
    {
        let stored = match attrs.get(crate::integrity::MERKLE_ROOT_ATTR) {
            Some(AttrValue::String(s)) => Some(s.as_str()),
            _ => None,
        };
        crate::integrity::verify_content(
            &memory_cache,
            &session_cache,
            &knowledge_cache,
            stored,
            strict,
        )?;
    }

    Ok((config, memory_cache, session_cache, knowledge_cache))
}

fn load_memory_group(
    file: &clawhdf5::File,
    embedding_dim: usize,
) -> Result<MemoryCache, MemoryError> {
    let group = file
        .group("memory")
        .map_err(|e| MemoryError::Schema(format!("missing /memory group: {e}")))?;

    let chunks = read_string_dataset_from_group(&group, "chunks")?;
    let n = chunks.len();

    let mut cache = MemoryCache::new(embedding_dim);

    if n == 0 {
        return Ok(cache);
    }

    let flat_embeddings = read_f32_dataset(&group, "embeddings")?;
    let source_channels = read_string_dataset_from_group(&group, "source_channel")?;
    let timestamps = read_f64_dataset(&group, "timestamps")?;
    let session_ids = read_string_dataset_from_group(&group, "session_ids")?;
    let tags = read_string_dataset_from_group(&group, "tags")?;
    let tombstones = read_u8_dataset(&group, "tombstones")?;

    // Read norms if present, otherwise compute from embeddings
    let norms = match read_f32_dataset(&group, "norms") {
        Ok(v) if v.len() == n => v,
        _ => {
            // Compute norms from flat embeddings
            flat_embeddings
                .chunks(embedding_dim)
                .map(|chunk| {
                    let sq_sum: f32 = chunk.iter().map(|x| x * x).sum();
                    sq_sum.sqrt()
                })
                .collect()
        }
    };

    // Unflatten embeddings
    let embeddings: Vec<Vec<f32>> = flat_embeddings
        .chunks(embedding_dim)
        .map(|c| c.to_vec())
        .collect();

    // Read activation_weights if present, default to vec![1.0; N] for backward compat
    let activation_weights = match read_f32_dataset(&group, "activation_weights") {
        Ok(w) if w.len() == n => w,
        _ => vec![1.0; n],
    };

    cache.chunks = chunks;
    cache.embeddings = embeddings;
    cache.source_channels = source_channels;
    cache.timestamps = timestamps;
    cache.session_ids = session_ids;
    cache.tags = tags;
    cache.tombstones = tombstones;
    cache.norms = norms;
    cache.activation_weights = activation_weights;

    Ok(cache)
}

fn load_sessions_group(file: &clawhdf5::File) -> Result<SessionCache, MemoryError> {
    let group = file
        .group("sessions")
        .map_err(|e| MemoryError::Schema(format!("missing /sessions group: {e}")))?;

    let ids = read_string_dataset_from_group(&group, "ids")?;
    if ids.is_empty() {
        return Ok(SessionCache::new());
    }

    let start_idxs = read_i64_dataset(&group, "start_idxs")?;
    let end_idxs = read_i64_dataset(&group, "end_idxs")?;
    let channels = read_string_dataset_from_group(&group, "channels")?;
    let timestamps = read_f64_dataset(&group, "timestamps")?;
    let summaries = read_string_dataset_from_group(&group, "summaries")?;

    let mut cache = SessionCache::new();
    for i in 0..ids.len() {
        cache.entries.push(crate::session::SessionEntry {
            id: ids[i].clone(),
            start_idx: start_idxs[i] as u64,
            end_idx: end_idxs[i] as u64,
            channel: channels[i].clone(),
            ts: timestamps[i],
        });
        cache.summaries.push(summaries[i].clone());
    }

    Ok(cache)
}

fn load_knowledge_group(file: &clawhdf5::File) -> Result<KnowledgeCache, MemoryError> {
    let group = file
        .group("knowledge_graph")
        .map_err(|e| MemoryError::Schema(format!("missing /knowledge_graph group: {e}")))?;

    let entity_ids = read_i64_dataset(&group, "entity_ids")?;
    let next_id = entity_ids.iter().max().map(|&m| m as u64 + 1).unwrap_or(0);
    let mut cache = KnowledgeCache::new_with_next_id(next_id);

    if !entity_ids.is_empty() {
        let entity_names = read_string_dataset_from_group(&group, "entity_names")?;
        let entity_types = read_string_dataset_from_group(&group, "entity_types")?;
        let emb_idxs = read_i64_dataset(&group, "entity_emb_idxs")?;

        for i in 0..entity_ids.len() {
            cache.entities.push(crate::knowledge::Entity {
                id: entity_ids[i] as u64,
                name: entity_names[i].clone(),
                entity_type: entity_types[i].clone(),
                embedding_idx: emb_idxs[i],
                ..Default::default()
            });
        }
    }

    let rel_srcs = read_i64_dataset(&group, "relation_srcs")?;
    if !rel_srcs.is_empty() {
        let rel_tgts = read_i64_dataset(&group, "relation_tgts")?;
        let rel_types = read_string_dataset_from_group(&group, "relation_types")?;
        let rel_weights = read_f32_dataset(&group, "relation_weights")?;
        let rel_ts = read_f64_dataset(&group, "relation_ts")?;

        for i in 0..rel_srcs.len() {
            cache.relations.push(crate::knowledge::Relation {
                src: rel_srcs[i] as u64,
                tgt: rel_tgts[i] as u64,
                relation: rel_types[i].clone(),
                weight: rel_weights[i],
                ts: rel_ts[i],
                ..Default::default()
            });
        }
    }

    // Load aliases (default to empty for backward compat)
    if let Ok(alias_strings) = read_string_dataset_from_group(&group, "alias_strings")
        && let Ok(alias_entity_ids) = read_i64_dataset(&group, "alias_entity_ids")
    {
        cache.alias_strings = alias_strings;
        cache.alias_entity_ids = alias_entity_ids;
    }

    Ok(cache)
}

// --- Helper functions ---

fn extract_string_attr(
    attrs: &std::collections::HashMap<String, AttrValue>,
    name: &str,
) -> Result<String, MemoryError> {
    match attrs.get(name) {
        Some(AttrValue::String(s)) => Ok(s.clone()),
        _ => Err(MemoryError::Schema(format!("missing attr: {name}"))),
    }
}

fn extract_i64_attr(
    attrs: &std::collections::HashMap<String, AttrValue>,
    name: &str,
) -> Result<i64, MemoryError> {
    match attrs.get(name) {
        Some(AttrValue::I64(v)) => Ok(*v),
        _ => Err(MemoryError::Schema(format!("missing attr: {name}"))),
    }
}

fn read_string_dataset_from_group(
    group: &clawhdf5::Group<'_>,
    name: &str,
) -> Result<Vec<String>, MemoryError> {
    let ds = group
        .dataset(name)
        .map_err(|e| MemoryError::Hdf5(format!("cannot read {name}: {e}")))?;
    let shape = ds
        .shape()
        .map_err(|e| MemoryError::Hdf5(format!("cannot read shape of {name}: {e}")))?;
    if shape.first() == Some(&0) || shape.is_empty() {
        return Ok(Vec::new());
    }
    ds.read_string()
        .map_err(|e| MemoryError::Hdf5(format!("cannot read strings from {name}: {e}")))
}

fn read_f32_dataset(group: &clawhdf5::Group<'_>, name: &str) -> Result<Vec<f32>, MemoryError> {
    let ds = group
        .dataset(name)
        .map_err(|e| MemoryError::Hdf5(format!("cannot read {name}: {e}")))?;
    let shape = ds
        .shape()
        .map_err(|e| MemoryError::Hdf5(format!("cannot read shape of {name}: {e}")))?;
    if shape.first() == Some(&0) {
        return Ok(Vec::new());
    }
    ds.read_f32()
        .map_err(|e| MemoryError::Hdf5(format!("cannot read f32 from {name}: {e}")))
}

fn read_f64_dataset(group: &clawhdf5::Group<'_>, name: &str) -> Result<Vec<f64>, MemoryError> {
    let ds = group
        .dataset(name)
        .map_err(|e| MemoryError::Hdf5(format!("cannot read {name}: {e}")))?;
    let shape = ds
        .shape()
        .map_err(|e| MemoryError::Hdf5(format!("cannot read shape of {name}: {e}")))?;
    if shape.first() == Some(&0) {
        return Ok(Vec::new());
    }
    ds.read_f64()
        .map_err(|e| MemoryError::Hdf5(format!("cannot read f64 from {name}: {e}")))
}

fn read_i64_dataset(group: &clawhdf5::Group<'_>, name: &str) -> Result<Vec<i64>, MemoryError> {
    let ds = group
        .dataset(name)
        .map_err(|e| MemoryError::Hdf5(format!("cannot read {name}: {e}")))?;
    let shape = ds
        .shape()
        .map_err(|e| MemoryError::Hdf5(format!("cannot read shape of {name}: {e}")))?;
    if shape.first() == Some(&0) {
        return Ok(Vec::new());
    }
    ds.read_i64()
        .map_err(|e| MemoryError::Hdf5(format!("cannot read i64 from {name}: {e}")))
}

fn read_u8_dataset(group: &clawhdf5::Group<'_>, name: &str) -> Result<Vec<u8>, MemoryError> {
    let ds = group
        .dataset(name)
        .map_err(|e| MemoryError::Hdf5(format!("cannot read {name}: {e}")))?;
    let shape = ds
        .shape()
        .map_err(|e| MemoryError::Hdf5(format!("cannot read shape of {name}: {e}")))?;
    if shape.first() == Some(&0) {
        return Ok(Vec::new());
    }
    // Read raw bytes - for u8 data we need the raw representation
    let data = ds
        .read_i32()
        .map_err(|e| MemoryError::Hdf5(format!("cannot read u8 from {name}: {e}")))?;
    Ok(data.into_iter().map(|v| v as u8).collect())
}
