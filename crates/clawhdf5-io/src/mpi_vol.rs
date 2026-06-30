//! MPI-IO VOL connector for parallel HDF5 reads and writes.
//!
//! Enable with the `mpi-io` feature: `cargo build --features mpi-io`.
//!
//! # Parallelism model
//!
//! **Read**: rank 0 reads the full file with `std::fs::read`, parses the
//! requested dataset, then broadcasts the raw bytes to all other ranks via
//! MPI broadcast. This is a root-read + broadcast pattern, *not* true
//! collective I/O (`MPI_File_read_at_all`).
//!
//! **Write**: each rank gathers its data shard to rank 0, which stitches
//! the contributions and writes the merged dataset atomically to disk. A
//! barrier ensures all ranks observe the completed file before continuing.

use crate::vol::{VirtualObjectLayer, VolCapability, VolError};

#[cfg(feature = "mpi-io")]
use mpi::traits::*;

/// Rank within the communicator.
type Rank = i32;

/// MPI-IO Virtual Object Layer connector.
///
/// Wraps an MPI communicator for collective HDF5 file I/O.
pub struct MpiVol {
    location: Option<String>,
    #[cfg(feature = "mpi-io")]
    pub universe: mpi::environment::Universe,
    #[cfg(not(feature = "mpi-io"))]
    _placeholder: (),
}

impl std::fmt::Debug for MpiVol {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MpiVol")
            .field("location", &self.location)
            .finish_non_exhaustive()
    }
}

impl MpiVol {
    /// Create an `MpiVol` using `MPI_COMM_WORLD`.
    ///
    /// Initializes MPI if not already initialized. Call once per process.
    #[cfg(feature = "mpi-io")]
    pub fn new_world() -> Result<Self, VolError> {
        let universe = mpi::initialize()
            .ok_or_else(|| VolError::Unsupported("MPI already finalized or init failed".into()))?;
        Ok(Self {
            location: None,
            universe,
        })
    }

    /// Stub for when the feature is disabled.
    #[cfg(not(feature = "mpi-io"))]
    pub fn new_world() -> Result<Self, VolError> {
        Err(VolError::Unsupported(
            "MPI-IO support requires the `mpi-io` feature".into(),
        ))
    }

    /// Returns the set of capabilities this VOL connector claims.
    ///
    /// This associated function mirrors the trait method and can be used in
    /// tests without constructing a live MPI universe.
    pub fn expected_capabilities() -> Vec<VolCapability> {
        vec![
            VolCapability::ReadData,
            VolCapability::WriteData,
            VolCapability::ListObjects,
            VolCapability::ChunkedStorage,
            VolCapability::ParallelIO,
        ]
    }

    /// Returns the MPI rank within COMM_WORLD (0-based).
    ///
    /// Returns 0 when MPI is not available.
    pub fn rank(&self) -> Rank {
        #[cfg(feature = "mpi-io")]
        {
            self.universe.world().rank()
        }
        #[cfg(not(feature = "mpi-io"))]
        {
            0
        }
    }

    /// Returns the total number of MPI processes.
    ///
    /// Returns 1 when MPI is not available.
    pub fn size(&self) -> Rank {
        #[cfg(feature = "mpi-io")]
        {
            self.universe.world().size()
        }
        #[cfg(not(feature = "mpi-io"))]
        {
            1
        }
    }
}

#[allow(unused_variables)]
impl VirtualObjectLayer for MpiVol {
    fn name(&self) -> &str {
        "mpi-io"
    }

    fn capabilities(&self) -> Vec<VolCapability> {
        vec![
            VolCapability::ReadData,
            VolCapability::WriteData,
            VolCapability::ListObjects,
            VolCapability::ChunkedStorage,
            VolCapability::ParallelIO,
        ]
    }

    fn open(&mut self, location: &str) -> Result<(), VolError> {
        self.location = Some(location.to_string());
        Ok(())
    }

    fn close(&mut self) -> Result<(), VolError> {
        self.location = None;
        Ok(())
    }

    fn read_dataset(&self, path: &str) -> Result<Vec<u8>, VolError> {
        let _loc = self.location.as_deref().ok_or_else(|| {
            VolError::Io(std::io::Error::new(
                std::io::ErrorKind::NotConnected,
                "file not open",
            ))
        })?;

        #[cfg(feature = "mpi-io")]
        {
            mpi_collective_read(self, _loc, path)
        }
        #[cfg(not(feature = "mpi-io"))]
        {
            Err(VolError::Unsupported("mpi-io feature not enabled".into()))
        }
    }

    fn write_dataset(
        &mut self,
        path: &str,
        data: &[u8],
        shape: &[u64],
        dtype: &str,
    ) -> Result<(), VolError> {
        let _loc = self.location.as_deref().ok_or_else(|| {
            VolError::Io(std::io::Error::new(
                std::io::ErrorKind::NotConnected,
                "file not open",
            ))
        })?;

        #[cfg(feature = "mpi-io")]
        {
            mpi_collective_write(self, _loc, path, data, shape, dtype)
        }
        #[cfg(not(feature = "mpi-io"))]
        {
            Err(VolError::Unsupported("mpi-io feature not enabled".into()))
        }
    }
}

/// Collective read: root reads the file, broadcasts the target dataset to all ranks.
#[cfg(feature = "mpi-io")]
fn mpi_collective_read(vol: &MpiVol, location: &str, path: &str) -> Result<Vec<u8>, VolError> {
    use clawhdf5_format::{
        data_layout::DataLayout, data_read::read_raw_data_full, dataspace::Dataspace,
        datatype::Datatype, filter_pipeline::FilterPipeline, group_v2::resolve_path_any,
        message_type::MessageType, object_header::ObjectHeader, signature::find_signature,
        superblock::Superblock,
    };
    use mpi::traits::*;

    let world = vol.universe.world();
    let rank = world.rank();

    let raw_data: Vec<u8>;
    let mut len_buf = [0usize; 1];

    if rank == 0 {
        let bytes = std::fs::read(location).map_err(VolError::Io)?;
        let sig = find_signature(&bytes).map_err(|e| VolError::DataError(e.to_string()))?;
        let sb = Superblock::parse(&bytes, sig).map_err(|e| VolError::DataError(e.to_string()))?;
        let addr = resolve_path_any(&bytes, &sb, path)
            .map_err(|e| VolError::NotFound(format!("{path}: {e}")))?;
        let oh = ObjectHeader::parse(&bytes, addr as usize, sb.offset_size, sb.length_size)
            .map_err(|e| VolError::DataError(e.to_string()))?;
        let dt = oh
            .messages
            .iter()
            .find(|m| m.msg_type == MessageType::Datatype)
            .ok_or_else(|| VolError::DataError("no datatype".into()))?;
        let (datatype, _) =
            Datatype::parse(&dt.data).map_err(|e| VolError::DataError(e.to_string()))?;
        let ds = oh
            .messages
            .iter()
            .find(|m| m.msg_type == MessageType::Dataspace)
            .ok_or_else(|| VolError::DataError("no dataspace".into()))?;
        let dataspace = Dataspace::parse(&ds.data, sb.length_size)
            .map_err(|e| VolError::DataError(e.to_string()))?;
        let dl = oh
            .messages
            .iter()
            .find(|m| m.msg_type == MessageType::DataLayout)
            .ok_or_else(|| VolError::DataError("no data layout".into()))?;
        let layout = DataLayout::parse(&dl.data, sb.offset_size, sb.length_size)
            .map_err(|e| VolError::DataError(e.to_string()))?;
        let pipeline = oh
            .messages
            .iter()
            .find(|m| m.msg_type == MessageType::FilterPipeline)
            .and_then(|m| FilterPipeline::parse(&m.data).ok());

        raw_data = read_raw_data_full(
            &bytes,
            &layout,
            &dataspace,
            &datatype,
            pipeline.as_ref(),
            sb.offset_size,
            sb.length_size,
        )
        .map_err(|e| VolError::DataError(e.to_string()))?;
        len_buf[0] = raw_data.len();
    } else {
        raw_data = Vec::new();
    }

    // Broadcast length then data
    world.process_at_rank(0).broadcast_into(&mut len_buf);
    let mut result = vec![0u8; len_buf[0]];
    if rank == 0 {
        result.copy_from_slice(&raw_data);
    }
    world.process_at_rank(0).broadcast_into(&mut result);
    Ok(result)
}

/// Collective write: rank 0 accumulates all contributions and writes atomically.
///
/// In a real parallel workload each rank provides its own data shard for a
/// different hyperslab. Here we demonstrate the pattern: all ranks send their
/// data to rank 0 which stitches and writes.
#[cfg(feature = "mpi-io")]
fn mpi_collective_write(
    vol: &MpiVol,
    location: &str,
    path: &str,
    data: &[u8],
    shape: &[u64],
    dtype: &str,
) -> Result<(), VolError> {
    use clawhdf5_format::file_writer::FileWriter as FmtWriter;
    use mpi::traits::*;

    let world = vol.universe.world();
    let size = world.size() as usize;

    // Each rank sends its data length to root
    let local_len = data.len();
    let mut all_lens = if world.rank() == 0 {
        vec![0usize; size]
    } else {
        Vec::new()
    };
    world
        .process_at_rank(0)
        .gather_into_root(&local_len, &mut all_lens);

    // Root collects all contributions and writes
    if world.rank() == 0 {
        let total: usize = all_lens.iter().sum();
        let mut merged = Vec::with_capacity(total);
        // Rank 0's own contribution first
        merged.extend_from_slice(data);
        // Receive from ranks 1..size
        for r in 1..size as i32 {
            let expected = all_lens[r as usize];
            let mut buf = vec![0u8; expected];
            world.process_at_rank(r).receive_into(&mut buf);
            merged.extend_from_slice(&buf);
        }

        // Write merged data via FileWriter
        let mut fw = FmtWriter::new();
        match dtype {
            "f64" => {
                let values: Vec<f64> = merged
                    .chunks_exact(8)
                    .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
                    .collect();
                fw.create_dataset(path).with_f64_data(&values);
            }
            "f32" => {
                let values: Vec<f32> = merged
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                    .collect();
                fw.create_dataset(path).with_f32_data(&values);
            }
            _ => {
                return Err(VolError::Unsupported(format!(
                    "mpi-io write: unsupported dtype {dtype}"
                )));
            }
        }

        let bytes = fw
            .finish()
            .map_err(|e| VolError::DataError(e.to_string()))?;
        std::fs::write(location, &bytes).map_err(VolError::Io)?;
    } else {
        // Non-root ranks send their data to root
        world.process_at_rank(0).send(data);
    }

    // Barrier: all ranks wait until root finishes writing
    world.barrier();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mpi_vol_no_feature_returns_unsupported() {
        #[cfg(not(feature = "mpi-io"))]
        {
            let result = MpiVol::new_world();
            assert!(
                matches!(result, Err(VolError::Unsupported(_))),
                "expected Unsupported error without mpi-io feature"
            );
        }
        #[cfg(feature = "mpi-io")]
        {
            // With MPI enabled, new_world() may succeed if MPI is installed.
            // Just verify it doesn't panic.
            let _ = MpiVol::new_world();
        }
    }

    #[test]
    fn mpi_vol_capabilities_include_parallel_io() {
        let caps = MpiVol::expected_capabilities();
        assert!(
            caps.contains(&VolCapability::ParallelIO),
            "expected ParallelIO in {caps:?}"
        );
        assert!(caps.contains(&VolCapability::ReadData));
        assert!(caps.contains(&VolCapability::WriteData));
    }

    #[test]
    fn no_feature_error_contains_feature_name() {
        #[cfg(not(feature = "mpi-io"))]
        {
            let e = MpiVol::new_world().unwrap_err();
            assert!(
                e.to_string().contains("mpi-io"),
                "error should mention 'mpi-io': {e}"
            );
        }
        #[cfg(feature = "mpi-io")]
        {
            // With mpi-io enabled this test is vacuous; the feature-off path
            // is what we're documenting.
        }
    }

    #[test]
    #[cfg(feature = "mpi-io")]
    fn collective_read_all_ranks_get_same_data() {
        use crate::vol::VirtualObjectLayer;
        use tempfile::TempDir;

        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("test.h5");
        {
            use clawhdf5_format::file_writer::FileWriter as FmtWriter;
            let mut fw = FmtWriter::new();
            fw.create_dataset("temperature")
                .with_f64_data(&[1.0, 2.0, 3.0, 4.0, 5.0]);
            let bytes = fw.finish().unwrap();
            std::fs::write(&path, &bytes).unwrap();
        }

        let mut vol = MpiVol::new_world().expect("MPI init failed");
        vol.open(path.to_str().unwrap()).unwrap();
        let data = vol.read_dataset("temperature").unwrap();

        assert_eq!(
            data.len(),
            40,
            "rank {} got {} bytes",
            vol.rank(),
            data.len()
        );

        let values: Vec<f64> = data
            .chunks_exact(8)
            .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(
            values,
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
            "rank {} got wrong data",
            vol.rank()
        );
    }

    #[test]
    #[cfg(feature = "mpi-io")]
    fn collective_write_assembles_all_shards() {
        use crate::vol::VirtualObjectLayer;
        use mpi::traits::*;
        use tempfile::TempDir;

        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("parallel_out.h5");

        let mut vol = MpiVol::new_world().expect("MPI init failed");
        vol.open(path.to_str().unwrap()).unwrap();

        let world = vol.universe.world();
        let rank = world.rank() as usize;
        let shard = ((rank as f64) * 10.0f64).to_le_bytes().to_vec();

        vol.write_dataset("values", &shard, &[world.size() as u64], "f64")
            .unwrap();

        let total_size = world.size() as usize;
        if rank == 0 {
            let bytes = std::fs::read(&path).unwrap();
            use clawhdf5_format::{
                data_layout::DataLayout, data_read::read_raw_data_full, dataspace::Dataspace,
                datatype::Datatype, group_v2::resolve_path_any, message_type::MessageType,
                object_header::ObjectHeader, signature::find_signature, superblock::Superblock,
            };
            let sig = find_signature(&bytes).unwrap();
            let sb = Superblock::parse(&bytes, sig).unwrap();
            let addr = resolve_path_any(&bytes, &sb, "values").unwrap();
            let oh =
                ObjectHeader::parse(&bytes, addr as usize, sb.offset_size, sb.length_size).unwrap();
            let (dt, _) = Datatype::parse(
                &oh.messages
                    .iter()
                    .find(|m| m.msg_type == MessageType::Datatype)
                    .unwrap()
                    .data,
            )
            .unwrap();
            let ds = Dataspace::parse(
                &oh.messages
                    .iter()
                    .find(|m| m.msg_type == MessageType::Dataspace)
                    .unwrap()
                    .data,
                sb.length_size,
            )
            .unwrap();
            let dl = DataLayout::parse(
                &oh.messages
                    .iter()
                    .find(|m| m.msg_type == MessageType::DataLayout)
                    .unwrap()
                    .data,
                sb.offset_size,
                sb.length_size,
            )
            .unwrap();
            let raw =
                read_raw_data_full(&bytes, &dl, &ds, &dt, None, sb.offset_size, sb.length_size)
                    .unwrap();
            assert_eq!(
                raw.len(),
                total_size * 8,
                "expected {} f64 values",
                total_size
            );
            let values: Vec<f64> = raw
                .chunks_exact(8)
                .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
                .collect();
            for (i, &v) in values.iter().enumerate() {
                assert!(
                    (v - (i as f64 * 10.0)).abs() < 1e-9,
                    "rank {i} shard wrong: got {v}"
                );
            }
        }
        world.barrier();
    }
}
