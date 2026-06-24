//! GPU search backend for accelerated vector similarity.
//!
//! When the `gpu` feature is enabled and GPU hardware is available, uses
//! `clawhdf5_gpu::GpuAccelerator` for real GPU-accelerated cosine and L2
//! searches. Falls back gracefully to CPU SIMD search otherwise.

/// GPU search backend that manages vector data on the GPU.
///
/// When the `gpu` feature is enabled, this backend wraps a real
/// `clawhdf5_gpu::GpuAccelerator` for hardware-accelerated search.
/// Falls back to CPU SIMD search when GPU is unavailable.
pub struct GpuSearchBackend {
    /// Real GPU accelerator (when gpu feature is enabled and hardware available).
    #[cfg(feature = "gpu")]
    accelerator: Option<clawhdf5_gpu::GpuAccelerator>,
    /// Vector dimension.
    dim: usize,
    /// Minimum collection size to justify GPU overhead.
    threshold: usize,
    /// Number of vectors currently uploaded.
    num_vectors: usize,
}

impl GpuSearchBackend {
    /// Attempt to initialize GPU backend.
    ///
    /// Returns a backend with GPU active only if hardware is detected,
    /// the `gpu` feature is enabled, and the collection size exceeds the threshold.
    pub fn try_init(vectors: &[Vec<f32>], norms: &[f32], dim: usize, threshold: usize) -> Self {
        #[cfg(feature = "gpu")]
        {
            if vectors.len() >= threshold {
                match clawhdf5_gpu::GpuAccelerator::new() {
                    Ok(mut accel) => {
                        let flat: Vec<f32> =
                            vectors.iter().flat_map(|v| v.iter().copied()).collect();
                        if accel.upload_vectors(&flat, dim).is_ok()
                            && accel.upload_norms(norms).is_ok()
                        {
                            return Self {
                                accelerator: Some(accel),
                                dim,
                                threshold,
                                num_vectors: vectors.len(),
                            };
                        }
                    }
                    Err(e) => {
                        log_gpu_fallback(&e.to_string());
                    }
                }
            }

            Self {
                accelerator: None,
                dim,
                threshold,
                num_vectors: vectors.len(),
            }
        }

        #[cfg(not(feature = "gpu"))]
        {
            let _ = (vectors, norms);
            Self {
                dim,
                threshold,
                num_vectors: 0,
            }
        }
    }

    /// Check if GPU acceleration is active.
    pub fn is_available(&self) -> bool {
        #[cfg(feature = "gpu")]
        {
            self.accelerator.is_some()
        }
        #[cfg(not(feature = "gpu"))]
        {
            false
        }
    }

    /// Get the dimension this backend was initialized with.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Get the threshold for GPU activation.
    pub fn threshold(&self) -> usize {
        self.threshold
    }

    /// Re-upload vectors after mutation (save/compact).
    pub fn re_upload(&mut self, vectors: &[Vec<f32>], norms: &[f32]) {
        self.num_vectors = vectors.len();

        #[cfg(feature = "gpu")]
        {
            // If we have an accelerator and still above threshold, re-upload
            if let Some(ref mut accel) = self.accelerator {
                if vectors.len() >= self.threshold {
                    let flat: Vec<f32> = vectors.iter().flat_map(|v| v.iter().copied()).collect();
                    if accel.upload_vectors(&flat, self.dim).is_err()
                        || accel.upload_norms(norms).is_err()
                    {
                        self.accelerator = None;
                    }
                } else {
                    // Below threshold, deactivate GPU
                    self.accelerator = None;
                }
                return;
            }

            // If we don't have an accelerator but now above threshold, try init
            if vectors.len() >= self.threshold
                && let Ok(mut accel) = clawhdf5_gpu::GpuAccelerator::new()
            {
                let flat: Vec<f32> = vectors.iter().flat_map(|v| v.iter().copied()).collect();
                if accel.upload_vectors(&flat, self.dim).is_ok()
                    && accel.upload_norms(norms).is_ok()
                {
                    self.accelerator = Some(accel);
                }
            }
        }

        #[cfg(not(feature = "gpu"))]
        {
            let _ = (vectors, norms);
        }
    }

    /// Search using GPU-accelerated cosine similarity.
    ///
    /// If GPU is not available, falls back to CPU SIMD prenorm search.
    pub fn search_cosine(
        &self,
        query: &[f32],
        vectors: &[Vec<f32>],
        norms: &[f32],
        tombstones: &[u8],
        k: usize,
    ) -> Vec<(usize, f32)> {
        #[cfg(feature = "gpu")]
        {
            if let Some(ref accel) = self.accelerator {
                match accel.cosine_search(query, k.min(self.num_vectors.max(1))) {
                    Ok(mut results) => {
                        // Filter out tombstoned entries
                        results.retain(|(i, _)| *i < tombstones.len() && tombstones[*i] == 0);
                        results.truncate(k);
                        return results;
                    }
                    Err(_) => {
                        // Fall through to CPU
                    }
                }
            }
        }

        cpu_fallback_cosine(query, vectors, norms, tombstones, k)
    }

    /// Search using GPU-accelerated L2 distance.
    pub fn search_l2(
        &self,
        query: &[f32],
        vectors: &[Vec<f32>],
        tombstones: &[u8],
        k: usize,
    ) -> Vec<(usize, f32)> {
        #[cfg(feature = "gpu")]
        {
            if let Some(ref accel) = self.accelerator {
                match accel.l2_search(query, k.min(self.num_vectors.max(1))) {
                    Ok(mut results) => {
                        results.retain(|(i, _)| *i < tombstones.len() && tombstones[*i] == 0);
                        results.truncate(k);
                        return results;
                    }
                    Err(_) => {
                        // Fall through to CPU
                    }
                }
            }
        }

        cpu_fallback_l2(query, vectors, tombstones, k)
    }

    /// Get the device info string (for metrics/logging).
    pub fn device_info(&self) -> String {
        #[cfg(feature = "gpu")]
        {
            if let Some(ref accel) = self.accelerator {
                return accel.device_info().to_string();
            }
        }
        "none".to_string()
    }
}

/// Check if GPU hardware is available at all.
pub fn detect_gpu() -> bool {
    #[cfg(feature = "gpu")]
    {
        clawhdf5_gpu::GpuAccelerator::is_available()
    }
    #[cfg(not(feature = "gpu"))]
    {
        false
    }
}

#[cfg(feature = "gpu")]
fn log_gpu_fallback(reason: &str) {
    // Logging for GPU init failure; callers can check is_available()
    eprintln!("[clawhdf5-agent] GPU init failed, falling back to CPU: {reason}");
}

/// CPU fallback for cosine search when GPU is not available.
fn cpu_fallback_cosine(
    query: &[f32],
    vectors: &[Vec<f32>],
    norms: &[f32],
    tombstones: &[u8],
    k: usize,
) -> Vec<(usize, f32)> {
    let query_norm = clawhdf5_accel::vector_norm(query);
    if query_norm == 0.0 {
        return Vec::new();
    }

    let mut results: Vec<(usize, f32)> = Vec::with_capacity(vectors.len());

    for (i, vec) in vectors.iter().enumerate() {
        if i < tombstones.len() && tombstones[i] != 0 {
            continue;
        }
        let vec_norm = if i < norms.len() {
            norms[i]
        } else {
            clawhdf5_accel::vector_norm(vec)
        };
        let score = crate::cosine_similarity_prenorm(query, query_norm, vec, vec_norm);
        results.push((i, score));
    }

    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    results.truncate(k);
    results
}

/// CPU fallback for L2 distance search.
fn cpu_fallback_l2(
    query: &[f32],
    vectors: &[Vec<f32>],
    tombstones: &[u8],
    k: usize,
) -> Vec<(usize, f32)> {
    let mut results: Vec<(usize, f32)> = Vec::with_capacity(vectors.len());

    for (i, vec) in vectors.iter().enumerate() {
        if i < tombstones.len() && tombstones[i] != 0 {
            continue;
        }
        let dist = clawhdf5_accel::l2_distance(query, vec);
        results.push((i, dist));
    }

    // Sort ascending (smallest distance first)
    results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    results.truncate(k);
    results
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_vectors(n: usize, dim: usize, seed: u32) -> Vec<Vec<f32>> {
        let mut s = seed;
        let mut next = || -> f32 {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            ((s >> 16) as f32) / 65536.0 - 0.5
        };
        (0..n).map(|_| (0..dim).map(|_| next()).collect()).collect()
    }

    #[test]
    fn gpu_detect_default_status() {
        // Without GPU feature or hardware, detection depends on compilation
        let detected = detect_gpu();
        // Just verify it returns a bool without panicking
        let _ = detected;
    }

    #[test]
    fn gpu_backend_fallback_when_unavailable() {
        let vectors = make_vectors(100, 32, 42);
        let norms: Vec<f32> = vectors
            .iter()
            .map(|v| clawhdf5_accel::vector_norm(v))
            .collect();
        let backend = GpuSearchBackend::try_init(&vectors, &norms, 32, 50);

        // On most CI/test environments GPU won't be available
        let tombstones = vec![0u8; 100];
        let query = vectors[0].clone();
        let results = backend.search_cosine(&query, &vectors, &norms, &tombstones, 10);

        assert!(!results.is_empty());
        assert!(results.len() <= 10);
        // First result should be the query vector itself (index 0)
        assert_eq!(results[0].0, 0);
        assert!((results[0].1 - 1.0).abs() < 1e-5);
    }

    #[test]
    fn gpu_backend_below_threshold() {
        let vectors = make_vectors(10, 32, 42);
        let norms: Vec<f32> = vectors
            .iter()
            .map(|v| clawhdf5_accel::vector_norm(v))
            .collect();
        let backend = GpuSearchBackend::try_init(&vectors, &norms, 32, 100);

        assert!(!backend.is_available());
        assert_eq!(backend.dim(), 32);
        assert_eq!(backend.threshold(), 100);
    }

    #[test]
    fn gpu_cosine_cpu_fallback_matches() {
        let vectors = make_vectors(200, 64, 42);
        let norms: Vec<f32> = vectors
            .iter()
            .map(|v| clawhdf5_accel::vector_norm(v))
            .collect();
        let tombstones = vec![0u8; 200];
        let query = vectors[5].clone();

        let fallback = cpu_fallback_cosine(&query, &vectors, &norms, &tombstones, 10);
        let backend = GpuSearchBackend::try_init(&vectors, &norms, 64, 50);
        let backend_results = backend.search_cosine(&query, &vectors, &norms, &tombstones, 10);

        assert_eq!(fallback.len(), backend_results.len());
        for (f, b) in fallback.iter().zip(&backend_results) {
            assert_eq!(f.0, b.0);
            assert!((f.1 - b.1).abs() < 1e-6);
        }
    }

    #[test]
    fn gpu_l2_search_returns_nearest() {
        let vectors = vec![
            vec![0.0, 0.0, 0.0],
            vec![1.0, 0.0, 0.0],
            vec![10.0, 10.0, 10.0],
        ];
        let tombstones = vec![0u8; 3];
        let query = vec![0.1, 0.0, 0.0];

        let results = cpu_fallback_l2(&query, &vectors, &tombstones, 3);
        // Closest should be vector 0 (distance ~0.01), then vector 1 (distance ~0.81)
        assert_eq!(results[0].0, 0);
        assert_eq!(results[1].0, 1);
        assert_eq!(results[2].0, 2);
    }

    #[test]
    fn gpu_search_respects_tombstones() {
        let vectors = make_vectors(50, 16, 42);
        let norms: Vec<f32> = vectors
            .iter()
            .map(|v| clawhdf5_accel::vector_norm(v))
            .collect();
        let mut tombstones = vec![0u8; 50];
        tombstones[0] = 1;
        tombstones[1] = 1;

        let query = vectors[2].clone();
        let results = cpu_fallback_cosine(&query, &vectors, &norms, &tombstones, 50);
        assert!(results.iter().all(|r| r.0 != 0 && r.0 != 1));
    }

    #[test]
    fn gpu_re_upload_updates_data() {
        let vectors = make_vectors(10, 16, 42);
        let norms: Vec<f32> = vectors
            .iter()
            .map(|v| clawhdf5_accel::vector_norm(v))
            .collect();
        let mut backend = GpuSearchBackend::try_init(&vectors, &norms, 16, 5);

        // Re-upload with more vectors
        let vectors2 = make_vectors(20, 16, 77);
        let norms2: Vec<f32> = vectors2
            .iter()
            .map(|v| clawhdf5_accel::vector_norm(v))
            .collect();
        backend.re_upload(&vectors2, &norms2);

        // Backend should still work (CPU fallback at minimum)
        let tombstones = vec![0u8; 20];
        let query = vectors2[0].clone();
        let results = backend.search_cosine(&query, &vectors2, &norms2, &tombstones, 5);
        assert!(!results.is_empty());
    }

    #[test]
    fn gpu_l2_search_respects_tombstones() {
        let vectors = vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![2.0, 0.0]];
        let mut tombstones = vec![0u8; 3];
        tombstones[0] = 1; // tombstone nearest vector

        let query = vec![0.0, 0.0];
        let results = cpu_fallback_l2(&query, &vectors, &tombstones, 3);
        assert!(results.iter().all(|r| r.0 != 0));
        assert_eq!(results[0].0, 1); // next nearest
    }

    #[test]
    fn device_info_returns_string() {
        let vectors = make_vectors(10, 16, 42);
        let norms: Vec<f32> = vectors
            .iter()
            .map(|v| clawhdf5_accel::vector_norm(v))
            .collect();
        let backend = GpuSearchBackend::try_init(&vectors, &norms, 16, 5);
        let info = backend.device_info();
        assert!(!info.is_empty());
    }
}
