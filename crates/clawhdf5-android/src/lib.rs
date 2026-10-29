//! Android JNI bridge for clawhdf5-agent HDF5 backend.
//!
//! Exposes `extern "C"` functions for use via JNI from Kotlin.
//! Each HDF5Memory instance is managed via an opaque handle (pointer).
//!
//! Thread safety: the caller (Kotlin side) must synchronize access
//! to a single handle. Multiple handles are independent.

use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::path::PathBuf;
use std::ptr;

use clawhdf5_agent::{AgentMemory, HDF5Memory, MemoryConfig, MemoryEntry};

// ---------------------------------------------------------------------------
// Handle management
// ---------------------------------------------------------------------------

/// Opaque handle to an HDF5Memory instance.
type Handle = *mut HDF5Memory;

/// Run an exported function's body, turning a panic into that function's
/// error value.
///
/// A Rust panic that reaches an `extern "C"` boundary aborts the process, so
/// any panic inside the agent took the whole host app down with it. Two such
/// panics were reachable from this ABI before the 2026-10 red-team round (a
/// wrong-dimension save followed by any search, and a wrong-dimension query);
/// both are fixed at their source, and this guard keeps the next one from
/// crashing the app. `AssertUnwindSafe` is sound here: a handle whose call
/// panicked may hold partially updated state, the same as after any error,
/// and the caller owns the decision to keep using or close it.
fn ffi_guard<T>(on_panic: T, f: impl FnOnce() -> T) -> T {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).unwrap_or(on_panic)
}

/// Create a new HDF5 memory file.
///
/// Returns a handle on success, null on failure.
///
/// # Safety
///
/// `path` and `agent_id` must be valid, null-terminated C strings.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn edgehdf5_create(
    path: *const c_char,
    agent_id: *const c_char,
    embedding_dim: u32,
) -> Handle {
    ffi_guard(ptr::null_mut(), || {
        // SAFETY: JNI caller guarantees the pointer argument is a valid null-terminated C string.
        let path = match unsafe { cstr_to_string(path) } {
            Some(s) => s,
            None => return ptr::null_mut(),
        };
        // SAFETY: JNI caller guarantees the pointer argument is a valid null-terminated C string.
        let agent_id = match unsafe { cstr_to_string(agent_id) } {
            Some(s) => s,
            None => return ptr::null_mut(),
        };

        let config = MemoryConfig::new(PathBuf::from(path), &agent_id, embedding_dim as usize);
        match HDF5Memory::create(config) {
            Ok(mem) => Box::into_raw(Box::new(mem)),
            Err(_) => ptr::null_mut(),
        }
    })
}

/// Open an existing HDF5 memory file.
///
/// Returns a handle on success, null on failure.
///
/// # Safety
///
/// `path` must be a valid, null-terminated C string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn edgehdf5_open(path: *const c_char) -> Handle {
    ffi_guard(ptr::null_mut(), || {
        // SAFETY: JNI caller guarantees the pointer argument is a valid null-terminated C string.
        let path = match unsafe { cstr_to_string(path) } {
            Some(s) => s,
            None => return ptr::null_mut(),
        };

        match HDF5Memory::open(std::path::Path::new(&path)) {
            Ok(mem) => Box::into_raw(Box::new(mem)),
            Err(_) => ptr::null_mut(),
        }
    })
}

/// Open an existing HDF5 memory file, failing closed when it carries no
/// content Merkle root or its WAL sidecar is unauthenticated (see
/// `HDF5Memory::open_strict`). Returns a handle on success, null on failure.
///
/// Every other public way to open a store can request this guarantee; the
/// C ABI could not until now (2026-10 red-team finding).
///
/// # Safety
///
/// `path` must be a valid, null-terminated C string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn edgehdf5_open_strict(path: *const c_char) -> Handle {
    ffi_guard(ptr::null_mut(), || {
        // SAFETY: JNI caller guarantees the pointer argument is a valid null-terminated C string.
        let path = match unsafe { cstr_to_string(path) } {
            Some(s) => s,
            None => return ptr::null_mut(),
        };

        match HDF5Memory::open_strict(std::path::Path::new(&path)) {
            Ok(mem) => Box::into_raw(Box::new(mem)),
            Err(_) => ptr::null_mut(),
        }
    })
}

/// Close and free an HDF5Memory handle.
///
/// # Safety
///
/// `handle` must be a handle previously returned by [`edgehdf5_create`] or
/// [`edgehdf5_open`], and must not be used after this call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn edgehdf5_close(handle: Handle) {
    ffi_guard((), || {
        if !handle.is_null() {
            // SAFETY: handle was created by Box::into_raw in edgehdf5_create; this is the final use.
            unsafe { drop(Box::from_raw(handle)) };
        }
    })
}

// ---------------------------------------------------------------------------
// Memory operations
// ---------------------------------------------------------------------------

/// Save a memory entry. Returns the entry index, or -1 on failure.
///
/// # Safety
///
/// - `handle` must be a valid, non-null handle.
/// - All `*const c_char` arguments must be valid, null-terminated C strings.
/// - `embedding_ptr` must point to at least `embedding_len` contiguous `f32` values.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn edgehdf5_save(
    handle: Handle,
    chunk: *const c_char,
    embedding_ptr: *const f32,
    embedding_len: u32,
    source_channel: *const c_char,
    timestamp: f64,
    session_id: *const c_char,
    tags: *const c_char,
) -> i64 {
    ffi_guard(-1, || {
        // SAFETY: handle is a valid non-null Handle from edgehdf5_create; caller ensures exclusive access.
        let mem = match unsafe { handle.as_mut() } {
            Some(m) => m,
            None => return -1,
        };

        // SAFETY: JNI caller guarantees the pointer argument is a valid null-terminated C string.
        let chunk = match unsafe { cstr_to_string(chunk) } {
            Some(s) => s,
            None => return -1,
        };
        // SAFETY: JNI caller guarantees the pointer argument is a valid null-terminated C string.
        let source_channel = match unsafe { cstr_to_string(source_channel) } {
            Some(s) => s,
            None => return -1,
        };
        // SAFETY: JNI caller guarantees the pointer argument is a valid null-terminated C string.
        let session_id = match unsafe { cstr_to_string(session_id) } {
            Some(s) => s,
            None => return -1,
        };
        // SAFETY: JNI caller guarantees the pointer argument is a valid null-terminated C string.
        let tags = match unsafe { cstr_to_string(tags) } {
            Some(s) => s,
            None => return -1,
        };

        // `slice::from_raw_parts` requires a non-null pointer even for length 0;
        // null with length 0 is the usual C spelling of "no embedding" (a
        // text-only memory), and null with a length is a caller error.
        let embedding = if embedding_ptr.is_null() {
            if embedding_len != 0 {
                return -1;
            }
            Vec::new()
        } else {
            // SAFETY: non-null (checked above); the caller guarantees embedding_ptr
            // points to embedding_len valid f32 values.
            unsafe { std::slice::from_raw_parts(embedding_ptr, embedding_len as usize) }.to_vec()
        };

        let entry = MemoryEntry {
            chunk,
            embedding,
            source_channel,
            timestamp,
            session_id,
            tags,
        };

        match mem.save(entry) {
            Ok(idx) => idx as i64,
            Err(_) => -1,
        }
    })
}

/// Get the number of active (non-deleted) entries.
///
/// # Safety
///
/// `handle` must be a valid handle or null (returns 0 if null).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn edgehdf5_count_active(handle: Handle) -> u64 {
    ffi_guard(0, || {
        // SAFETY: handle is a valid non-null Handle from edgehdf5_create.
        match unsafe { handle.as_ref() } {
            Some(mem) => mem.count_active() as u64,
            None => 0,
        }
    })
}

/// Get the total number of entries (including tombstoned).
///
/// # Safety
///
/// `handle` must be a valid handle or null (returns 0 if null).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn edgehdf5_count(handle: Handle) -> u64 {
    ffi_guard(0, || {
        // SAFETY: handle is a valid non-null Handle from edgehdf5_create.
        match unsafe { handle.as_ref() } {
            Some(mem) => mem.count() as u64,
            None => 0,
        }
    })
}

/// Delete a memory entry by index. Returns 0 on success, -1 on failure.
///
/// # Safety
///
/// `handle` must be a valid, non-null handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn edgehdf5_delete(handle: Handle, index: u64) -> i32 {
    ffi_guard(-1, || {
        // SAFETY: handle is a valid non-null Handle from edgehdf5_create; caller ensures exclusive access.
        let mem = match unsafe { handle.as_mut() } {
            Some(m) => m,
            None => return -1,
        };

        match mem.delete(index as usize) {
            Ok(()) => 0,
            Err(_) => -1,
        }
    })
}

// ---------------------------------------------------------------------------
// Hybrid search
// ---------------------------------------------------------------------------

/// Result buffer for hybrid search. Caller allocates arrays.
///
/// Performs hybrid search and writes up to `max_results` entries into the
/// provided output arrays. Returns the number of results written.
///
/// # Safety
///
/// - `handle` must be a valid, non-null handle.
/// - `query_text` must be a valid, null-terminated C string.
/// - `query_embedding_ptr` must point to at least `query_embedding_len` `f32` values.
/// - `out_indices` and `out_scores` must point to arrays of at least `max_results` elements.
/// - `out_chunks` must be null or point to an array of at least `max_results` pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn edgehdf5_hybrid_search(
    handle: Handle,
    query_embedding_ptr: *const f32,
    query_embedding_len: u32,
    query_text: *const c_char,
    vector_weight: f32,
    keyword_weight: f32,
    max_results: u32,
    out_indices: *mut u64,
    out_scores: *mut f32,
    out_chunks: *mut *mut c_char,
) -> u32 {
    ffi_guard(0, || {
        // SAFETY: handle is a valid non-null Handle from edgehdf5_create; caller ensures exclusive access.
        let mem = match unsafe { handle.as_mut() } {
            Some(m) => m,
            None => return 0,
        };
        // SAFETY: JNI caller guarantees the pointer argument is a valid null-terminated C string.
        let query_text = match unsafe { cstr_to_string(query_text) } {
            Some(s) => s,
            None => return 0,
        };
        // Same null convention as `edgehdf5_save`: null with length 0 means "no
        // vector", so the search is BM25-only.
        let query_embedding: &[f32] = if query_embedding_ptr.is_null() {
            if query_embedding_len != 0 {
                return 0;
            }
            &[]
        } else {
            // SAFETY: non-null (checked above); the caller guarantees
            // query_embedding_ptr points to query_embedding_len valid f32 values.
            unsafe { std::slice::from_raw_parts(query_embedding_ptr, query_embedding_len as usize) }
        };
        // The result arrays are written unconditionally below.
        if max_results > 0 && (out_indices.is_null() || out_scores.is_null()) {
            return 0;
        }

        let results = mem.hybrid_search(
            query_embedding,
            &query_text,
            vector_weight,
            keyword_weight,
            max_results as usize,
        );

        let count = results.len().min(max_results as usize);
        for (i, result) in results.iter().take(count).enumerate() {
            // SAFETY: The CString result pointers are valid Rust-owned allocations from CString::into_raw.
            unsafe {
                *out_indices.add(i) = result.index as u64;
                *out_scores.add(i) = result.score;
                if !out_chunks.is_null() {
                    match CString::new(result.chunk.as_str()) {
                        Ok(cs) => *out_chunks.add(i) = cs.into_raw(),
                        Err(_) => *out_chunks.add(i) = ptr::null_mut(),
                    }
                }
            }
        }

        count as u32
    })
}

/// Free a chunk string returned by hybrid search.
///
/// # Safety
///
/// `s` must be a pointer previously returned by [`edgehdf5_hybrid_search`]
/// via `out_chunks`, or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn edgehdf5_free_string(s: *mut c_char) {
    ffi_guard((), || {
        if !s.is_null() {
            // SAFETY: s was created by CString::into_raw in this module; this is the final use.
            unsafe { drop(CString::from_raw(s)) };
        }
    })
}

// ---------------------------------------------------------------------------
// Session management
// ---------------------------------------------------------------------------

/// Add a session entry. Returns 0 on success, -1 on failure.
///
/// # Safety
///
/// - `handle` must be a valid, non-null handle.
/// - All `*const c_char` arguments must be valid, null-terminated C strings.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn edgehdf5_add_session(
    handle: Handle,
    id: *const c_char,
    start_idx: u64,
    end_idx: u64,
    channel: *const c_char,
    summary: *const c_char,
) -> i32 {
    ffi_guard(-1, || {
        // SAFETY: handle is a valid non-null Handle from edgehdf5_create; caller ensures exclusive access.
        let mem = match unsafe { handle.as_mut() } {
            Some(m) => m,
            None => return -1,
        };
        // SAFETY: JNI caller guarantees the pointer argument is a valid null-terminated C string.
        let id = match unsafe { cstr_to_string(id) } {
            Some(s) => s,
            None => return -1,
        };
        // SAFETY: JNI caller guarantees the pointer argument is a valid null-terminated C string.
        let channel = match unsafe { cstr_to_string(channel) } {
            Some(s) => s,
            None => return -1,
        };
        // SAFETY: JNI caller guarantees the pointer argument is a valid null-terminated C string.
        let summary = match unsafe { cstr_to_string(summary) } {
            Some(s) => s,
            None => return -1,
        };

        match mem.add_session(
            &id,
            start_idx as usize,
            end_idx as usize,
            &channel,
            &summary,
        ) {
            Ok(()) => 0,
            Err(_) => -1,
        }
    })
}

/// Get a session summary by ID. Returns a C string (caller must free with
/// `edgehdf5_free_string`), or null if not found.
///
/// # Safety
///
/// - `handle` must be a valid handle or null.
/// - `session_id` must be a valid, null-terminated C string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn edgehdf5_get_session_summary(
    handle: Handle,
    session_id: *const c_char,
) -> *mut c_char {
    ffi_guard(ptr::null_mut(), || {
        // SAFETY: handle is a valid non-null Handle from edgehdf5_create.
        let mem = match unsafe { handle.as_ref() } {
            Some(m) => m,
            None => return ptr::null_mut(),
        };
        // SAFETY: JNI caller guarantees the pointer argument is a valid null-terminated C string.
        let session_id = match unsafe { cstr_to_string(session_id) } {
            Some(s) => s,
            None => return ptr::null_mut(),
        };

        match mem.get_session_summary(&session_id) {
            Ok(Some(summary)) => match CString::new(summary) {
                Ok(cs) => cs.into_raw(),
                Err(_) => ptr::null_mut(),
            },
            _ => ptr::null_mut(),
        }
    })
}

// ---------------------------------------------------------------------------
// Knowledge graph
// ---------------------------------------------------------------------------

/// Add a knowledge graph entity. Returns entity ID, or -1 on failure.
///
/// # Safety
///
/// - `handle` must be a valid, non-null handle.
/// - `name` and `entity_type` must be valid, null-terminated C strings.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn edgehdf5_add_entity(
    handle: Handle,
    name: *const c_char,
    entity_type: *const c_char,
    embedding_idx: i64,
) -> i64 {
    ffi_guard(-1, || {
        // SAFETY: handle is a valid non-null Handle from edgehdf5_create; caller ensures exclusive access.
        let mem = match unsafe { handle.as_mut() } {
            Some(m) => m,
            None => return -1,
        };
        // SAFETY: JNI caller guarantees the pointer argument is a valid null-terminated C string.
        let name = match unsafe { cstr_to_string(name) } {
            Some(s) => s,
            None => return -1,
        };
        // SAFETY: JNI caller guarantees the pointer argument is a valid null-terminated C string.
        let entity_type = match unsafe { cstr_to_string(entity_type) } {
            Some(s) => s,
            None => return -1,
        };

        match mem.add_entity(&name, &entity_type, embedding_idx) {
            Ok(id) => id as i64,
            Err(_) => -1,
        }
    })
}

/// Add a knowledge graph relation. Returns 0 on success, -1 on failure.
///
/// # Safety
///
/// - `handle` must be a valid, non-null handle.
/// - `relation` must be a valid, null-terminated C string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn edgehdf5_add_relation(
    handle: Handle,
    src: u64,
    tgt: u64,
    relation: *const c_char,
    weight: f32,
) -> i32 {
    ffi_guard(-1, || {
        // SAFETY: handle is a valid non-null Handle from edgehdf5_create; caller ensures exclusive access.
        let mem = match unsafe { handle.as_mut() } {
            Some(m) => m,
            None => return -1,
        };
        // SAFETY: JNI caller guarantees the pointer argument is a valid null-terminated C string.
        let relation = match unsafe { cstr_to_string(relation) } {
            Some(s) => s,
            None => return -1,
        };

        match mem.add_relation(src, tgt, &relation, weight) {
            Ok(()) => 0,
            Err(_) => -1,
        }
    })
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Convert a C string pointer to an owned Rust String.
///
/// # Safety
/// The pointer must be a valid null-terminated C string.
unsafe fn cstr_to_string(ptr: *const c_char) -> Option<String> {
    if ptr.is_null() {
        return None;
    }
    // SAFETY: ptr is non-null (checked above) and is a valid null-terminated C string per caller.
    unsafe { CStr::from_ptr(ptr) }
        .to_str()
        .ok()
        .map(String::from)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cstr(s: &str) -> CString {
        CString::new(s).unwrap()
    }

    fn new_store(dir: &std::path::Path, name: &str) -> (CString, Handle) {
        let path = cstr(dir.join(name).to_str().unwrap());
        let agent = cstr("agent");
        // SAFETY: valid C strings; the dimension matches the 4-float vectors below.
        let h = unsafe { edgehdf5_create(path.as_ptr(), agent.as_ptr(), 4) };
        assert!(!h.is_null());
        (path, h)
    }

    fn save(h: Handle, text: &str, emb: Option<&[f32]>) -> i64 {
        let (chunk, chan, sess, tags) = (cstr(text), cstr("c"), cstr("s"), cstr(""));
        let (ptr, len) = match emb {
            Some(e) => (e.as_ptr(), e.len() as u32),
            None => (std::ptr::null(), 0),
        };
        // SAFETY: valid handle, valid C strings, ptr/len describe `emb` or are null/0.
        unsafe {
            edgehdf5_save(h, chunk.as_ptr(), ptr, len, chan.as_ptr(), 1.0, sess.as_ptr(), tags.as_ptr())
        }
    }

    /// 2026-10 red-team regressions: null pointers are handled rather than
    /// passed to `slice::from_raw_parts` (undefined behavior even at length
    /// 0), and a wrong-dimension save or query returns an error instead of
    /// panicking into the C ABI (which aborts the host process).
    #[test]
    fn null_pointers_and_wrong_dimensions_are_errors_not_ub_or_aborts() {
        let dir = tempfile::tempdir().unwrap();
        let (_p, h) = new_store(dir.path(), "m.h5");
        assert!(save(h, "text only memory", None) >= 0, "null + 0 is a text-only memory");
        assert_eq!(save(h, "with vector", Some(&[0.5, 0.5, 0.5, 0.5])), 1);
        // Null pointer with a non-zero length.
        let (chunk, chan, sess, tags) = (cstr("x"), cstr("c"), cstr("s"), cstr(""));
        // SAFETY: valid handle and C strings; the null embedding is the case under test.
        let r = unsafe {
            edgehdf5_save(h, chunk.as_ptr(), std::ptr::null(), 4, chan.as_ptr(), 1.0, sess.as_ptr(), tags.as_ptr())
        };
        assert_eq!(r, -1);
        // Wrong dimension: rejected, not stored.
        assert_eq!(save(h, "bad", Some(&[1.0, 2.0])), -1);

        let query = cstr("memory");
        let mut idx = [0u64; 4];
        let mut scores = [0f32; 4];
        // SAFETY: valid handle; output arrays hold 4 entries; null chunk output is allowed.
        let n = unsafe {
            edgehdf5_hybrid_search(
                h, std::ptr::null(), 0, query.as_ptr(), 0.7, 0.3, 4,
                idx.as_mut_ptr(), scores.as_mut_ptr(), std::ptr::null_mut(),
            )
        };
        assert!(n >= 1, "BM25-only search with a null query vector");
        let short = [1.0f32, 0.0];
        // SAFETY: as above, with a 2-element query against a 4-dimensional store.
        let n = unsafe {
            edgehdf5_hybrid_search(
                h, short.as_ptr(), 2, query.as_ptr(), 0.7, 0.3, 4,
                idx.as_mut_ptr(), scores.as_mut_ptr(), std::ptr::null_mut(),
            )
        };
        assert!(n >= 1, "a wrong-dimension query scores no vector component but still returns BM25 hits");
        // Null output arrays.
        // SAFETY: valid handle; the null outputs are the case under test.
        let n = unsafe {
            edgehdf5_hybrid_search(
                h, std::ptr::null(), 0, query.as_ptr(), 0.7, 0.3, 4,
                std::ptr::null_mut(), std::ptr::null_mut(), std::ptr::null_mut(),
            )
        };
        assert_eq!(n, 0);
        // SAFETY: h came from edgehdf5_create and is not used afterwards.
        unsafe { edgehdf5_close(h) };
    }

    /// The bridge now builds the agent with `integrity`, so stores carry a
    /// content root and `edgehdf5_open_strict` can verify them; an empty
    /// store (written with no root) is refused by strict and accepted by the
    /// default open.
    #[test]
    fn open_strict_verifies_protected_stores_and_refuses_unprotected_ones() {
        let dir = tempfile::tempdir().unwrap();
        let (empty_path, h) = new_store(dir.path(), "empty.h5");
        // SAFETY: h came from edgehdf5_create.
        unsafe { edgehdf5_close(h) };
        // SAFETY: valid C string.
        let strict = unsafe { edgehdf5_open_strict(empty_path.as_ptr()) };
        assert!(strict.is_null(), "no content root: strict open must refuse");
        // SAFETY: valid C string.
        let lax = unsafe { edgehdf5_open(empty_path.as_ptr()) };
        assert!(!lax.is_null());
        // SAFETY: lax came from edgehdf5_open.
        unsafe { edgehdf5_close(lax) };

        let mut cfg = clawhdf5_agent::MemoryConfig::new(dir.path().join("full.h5"), "agent", 4);
        cfg.wal_enabled = false; // write through, so the root is on disk
        let full_path = cstr(cfg.path.to_str().unwrap());
        let mut mem = HDF5Memory::create(cfg).unwrap();
        mem.save(MemoryEntry {
            chunk: "protected".into(),
            embedding: vec![0.5; 4],
            source_channel: "c".into(),
            timestamp: 1.0,
            session_id: "s".into(),
            tags: String::new(),
        })
        .unwrap();
        drop(mem);
        // SAFETY: valid C string.
        let strict = unsafe { edgehdf5_open_strict(full_path.as_ptr()) };
        assert!(!strict.is_null(), "a store written by this build carries a root and verifies");
        // SAFETY: strict came from edgehdf5_open_strict.
        unsafe { edgehdf5_close(strict) };
    }

    #[test]
    fn ffi_guard_turns_a_panic_into_the_error_value() {
        assert_eq!(ffi_guard(-1i64, || 7), 7);
        assert_eq!(ffi_guard(-1i64, || panic!("boom")), -1);
        assert!(ffi_guard(ptr::null_mut::<HDF5Memory>(), || panic!("boom")).is_null());
    }
}
