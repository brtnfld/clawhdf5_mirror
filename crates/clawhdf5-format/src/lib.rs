//! Pure-Rust HDF5 binary format parsing and writing.
//!
//! `clawhdf5-format` is a zero-dependency (no C libhdf5) crate for reading and
//! writing HDF5 files.  It supports `no_std` environments with the `alloc` crate.
//!
//! # Writing files
//!
//! Use [`file_writer::FileWriter`] to create HDF5 files:
//!
//! ```rust
//! use clawhdf5_format::file_writer::{FileWriter, AttrValue};
//!
//! let mut fw = FileWriter::new();
//! fw.create_dataset("data")
//!     .with_f64_data(&[1.0, 2.0, 3.0])
//!     .with_shape(&[3])
//!     .set_attr("unit", AttrValue::String("m/s".into()));
//! let bytes = fw.finish().unwrap();
//! ```
//!
//! # Reading files
//!
//! Parsing follows the HDF5 object model: superblock → object header → messages.
//!
//! ```rust,no_run
//! use clawhdf5_format::{signature, superblock, object_header, group_v2,
//!     datatype, dataspace, data_layout, data_read, message_type::MessageType};
//!
//! let file_data = std::fs::read("output.h5").unwrap();
//! let sig = signature::find_signature(&file_data).unwrap();
//! let sb  = superblock::Superblock::parse(&file_data, sig).unwrap();
//! let addr = group_v2::resolve_path_any(&file_data, &sb, "data").unwrap();
//! let hdr = object_header::ObjectHeader::parse(
//!     &file_data, addr as usize, sb.offset_size, sb.length_size).unwrap();
//! ```
//!
//! # Features
//!
//! | Feature | Default | Description |
//! |---------|---------|-------------|
//! | `std` | yes | Standard library support |
//! | `checksum` | yes | Jenkins lookup3 checksum validation |
//! | `deflate` | yes | Deflate (gzip) compression via `flate2` |
//! | `provenance` | yes | SHINES provenance — SHA-256 hashing & verification |

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(not(feature = "std"))]
extern crate alloc;

pub mod attribute;
pub mod attribute_info;
#[cfg(feature = "baselines")]
pub mod baselines;
pub mod btree_v1;
pub mod btree_v2;
pub mod checksum;
pub mod chunk_cache;
pub mod chunk_index;
pub mod chunked_read;
pub mod chunked_write;
pub mod data_layout;
pub mod data_read;
pub mod dataspace;
pub mod datatype;
pub mod dict_encoding;
pub mod ea_writer;
pub mod error;
pub mod extensible_array;
pub mod file_writer;
pub mod filter_pipeline;
pub mod filters;
pub mod fixed_array;
pub mod fractal_heap;
pub mod global_heap;
pub mod group_info;
pub mod group_v1;
pub mod group_v2;
#[cfg(feature = "parallel")]
pub mod lane_partition;
pub mod link_info;
pub mod link_message;
pub mod local_heap;
#[cfg(feature = "merkle")]
pub mod merkle;
#[cfg(feature = "merkle")]
pub mod merkle_journal;
#[cfg(feature = "merkle")]
pub mod merkle_recovery;
pub mod message_type;
pub mod metadata_cache;
pub mod metadata_index;
pub mod object_header;
pub mod object_header_writer;
#[cfg(feature = "parallel")]
pub mod parallel_read;
// The profiler's counters are AtomicU64, which some no_std targets
// (e.g. thumbv7em) don't provide; profiling is meaningless there anyway.
#[cfg(target_has_atomic = "64")]
pub mod profiling;
pub mod property_list;
pub mod selection;
pub mod shared_message;
pub mod signature;
#[cfg(feature = "merkle")]
pub mod subset_proof;
pub mod superblock;
pub mod symbol_table;
pub mod type_builders;
pub mod vl_data;

#[cfg(feature = "provenance")]
pub mod provenance;
