//! P2.5: Cloud / WAN evaluation harness (S2-D2-Yr2 §7, §"Cloud Object-Store
//! Evaluation").
//!
//! Measures proof-path retrieval latency and bandwidth over real S3
//! range-GETs, comparing `merkle_range_get` (fetch only the O(log N)
//! companion-dataset nodes + one chunk's data needed to verify a single
//! chunk) against `full_redownload` (fetch the entire object, as a client
//! without Merkle proofs would have to). Writes
//! `crates/clawhdf5-format/benches/results/phase2-cloud.csv` in the schema
//! required by S2-D2-Yr2 §7.5, feeding Figure 6 (right panel, RQ6, §7.6).
//!
//! # Requires real AWS credentials and an S3 bucket
//!
//! This is not runnable without network access to S3: it uploads test
//! files, then measures real range-GET latency against them. Configure via
//! environment variables (standard AWS credential chain — env vars,
//! `~/.aws/config`, IMDS — is used for auth; nothing is read from
//! `.cargo/config.toml`):
//!
//!   CLAW_S3_BUCKET               (required) target bucket, must already exist
//!   AWS_REGION                   (required) region the bucket lives in
//!   CLAW_CLOUD_EVAL_REGION_PAIR  (default: "<AWS_REGION>_to_<AWS_REGION>")
//!                                 label recorded in the region_pair column;
//!                                 set this to e.g. "us-east-1_to_eu-west-1"
//!                                 when you run this binary from a host in a
//!                                 *different* region than the bucket, to
//!                                 collect the cross-region WAN numbers the
//!                                 paper asks for. This harness cannot detect
//!                                 its own network position, so the label is
//!                                 the operator's responsibility.
//!   CLAW_CLOUD_EVAL_SIZES_MB     (default: "64") comma-separated dataset
//!                                 sizes to upload and measure, in MB. Must
//!                                 stay under 5120 (5 GiB) per entry —
//!                                 `s3::upload` is a single PutObject, and
//!                                 S3 hard-caps that at 5 GiB; genuinely
//!                                 larger surrogates need multipart upload,
//!                                 which this harness doesn't implement.
//!                                 P2.5 step 2 wants "1 GB and a larger
//!                                 surrogate for 1 TB-scale behavior" — e.g.
//!                                 set this to "1024,4096" for the real
//!                                 run; the 64 MB default is a fast smoke
//!                                 test.
//!   CLAW_CLOUD_EVAL_CHUNK_BYTES  (default: "4096") Merkle leaf chunk size;
//!                                 must be > 0 (n_chunks = dataset_size /
//!                                 chunk_bytes)
//!   CLAW_CLOUD_EVAL_TRIALS       (default: "30") per the statistical
//!                                 protocol (S2-D2-Yr2 §"Statistical
//!                                 Protocol": min. 30 trials, 5 warmup);
//!                                 must be > 0
//!   CLAW_CLOUD_EVAL_WARMUP       (default: "5")
//!
//! Usage:
//!   CLAW_S3_BUCKET=my-test-bucket AWS_REGION=us-east-1 \
//!     cargo run --release -p clawhdf5-io --example cloud_eval --features s3
//!
//! Output is merged into the CSV keyed by (region_pair, dataset_size_gb):
//! re-running for one region pair/size refreshes only those rows, the same
//! merge-by-key convention `clawhdf5-sign`'s `signing_bench` example uses
//! for its platform column.

#[cfg(not(feature = "s3"))]
fn main() {
    eprintln!("cloud_eval requires --features s3");
    std::process::exit(1);
}

#[cfg(feature = "s3")]
mod run {
    use clawhdf5_format::file_writer::FileWriter as FmtWriter;
    use clawhdf5_format::merkle::{
        HashAlg, INLINE_CHUNK_THRESHOLD, MerkleCompanionResult, MerkleTree, write_merkle_companion,
    };
    use clawhdf5_io::async_read::AsyncHDF5Read;
    use clawhdf5_io::s3::{self, S3Location, S3Reader};
    use std::io::Write as IoWrite;
    use std::time::Instant;

    const DATASET_NAME: &str = "cloud_eval_data";

    struct Row {
        method: &'static str,
        measurement: &'static str,
        dataset_size_gb: f64,
        n_chunks: usize,
        region_pair: String,
        value: f64,
        trial: usize,
    }

    fn env_or(name: &str, default: &str) -> String {
        std::env::var(name).unwrap_or_else(|_| default.to_string())
    }

    /// Unset -> `default`. Set but unparseable -> a clear panic naming the
    /// variable and the bad value, matching `CLAW_CLOUD_EVAL_SIZES_MB`'s
    /// `.expect(...)` pattern below — silently falling back to `default` on
    /// a typo'd value (e.g. `CLAW_CLOUD_EVAL_TRIALS=3O`, letter O) would
    /// otherwise burn real S3 requests under a configuration the operator
    /// didn't actually set.
    fn env_usize(name: &str, default: usize) -> usize {
        match std::env::var(name) {
            Err(_) => default,
            Ok(v) => v
                .trim()
                .parse()
                .unwrap_or_else(|_| panic!("{name} must be a positive integer, got {v:?}")),
        }
    }

    /// Build an in-memory HDF5 file with `n_chunks` chunks of `chunk_bytes`
    /// each, plus a Merkle companion dataset, mirroring the on-disk layout
    /// `merkle::write_merkle_companion` produces (the same layout
    /// `clawhdf5_io::s3::resolve_companion_layout` parses on the read side).
    fn build_test_file(n_chunks: usize, chunk_bytes: usize) -> (Vec<u8>, MerkleTree) {
        let chunks: Vec<Vec<u8>> = (0..n_chunks)
            .map(|i| {
                let mut chunk = vec![0u8; chunk_bytes];
                for (j, byte) in chunk.iter_mut().enumerate() {
                    *byte = ((i.wrapping_add(j)) % 256) as u8;
                }
                chunk
            })
            .collect();
        let refs: Vec<&[u8]> = chunks.iter().map(|c| c.as_slice()).collect();
        let tree = MerkleTree::from_chunks(&refs, HashAlg::Blake3);

        let mut fw = FmtWriter::new();
        let result = write_merkle_companion(&mut fw, DATASET_NAME, &tree)
            .expect("write_merkle_companion should succeed");
        assert!(
            matches!(result, MerkleCompanionResult::Dataset { .. }),
            "n_chunks={n_chunks} is at or below INLINE_CHUNK_THRESHOLD ({INLINE_CHUNK_THRESHOLD}), \
             so the companion tree was written inline into the attribute instead of as a separate \
             /merkle/{{name}} dataset this harness's S3 range-GET path requires — increase \
             CLAW_CLOUD_EVAL_SIZES_MB or lower CLAW_CLOUD_EVAL_CHUNK_BYTES so n_chunks exceeds \
             {INLINE_CHUNK_THRESHOLD}"
        );
        let ds = fw.create_dataset(DATASET_NAME);
        let all_data: Vec<u8> = chunks.into_iter().flatten().collect();
        ds.with_u8_data(&all_data);

        let file_bytes = fw.finish().expect("file should build");
        (file_bytes, tree)
    }

    /// Writes the header comment + column header. `region_pairs` and
    /// `dataset_sizes_gb` should cover every row that will end up in the
    /// file below (preserved rows from prior runs *and* this run's new
    /// rows) — S2-D2-Yr2 §7.5 asks the header to "record the S3 bucket
    /// name, region pairs [plural], and file sizes," and a header derived
    /// from only the current run would silently go stale (misrepresenting
    /// what the file actually contains) the moment a second region pair or
    /// size gets merged in by a later run.
    fn write_header(
        w: &mut impl IoWrite,
        bucket: &str,
        region_pairs: &[String],
        dataset_sizes_gb: &[String],
    ) -> std::io::Result<()> {
        let date = std::process::Command::new("date")
            .arg("+%Y-%m-%dT%H:%M:%SZ")
            .output()
            .ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .unwrap_or_default();
        writeln!(
            w,
            "# bucket={bucket} region_pairs={} dataset_sizes_gb={} date={}",
            region_pairs.join("|"),
            dataset_sizes_gb.join("|"),
            date.trim()
        )?;
        writeln!(
            w,
            "method,measurement,dataset_size_gb,n_chunks,region_pair,value,trial"
        )
    }

    fn write_row(w: &mut impl IoWrite, r: &Row) -> std::io::Result<()> {
        writeln!(
            w,
            "{},{},{:.6},{},{},{:.3},{}",
            r.method, r.measurement, r.dataset_size_gb, r.n_chunks, r.region_pair, r.value, r.trial
        )
    }

    /// Merge `new_rows` into `csv_path`, replacing any existing rows whose
    /// (region_pair, dataset_size_gb) match a row being (re-)collected,
    /// while preserving rows for every other region pair / size already on
    /// disk. Mirrors `clawhdf5-sign`'s `signing_bench` merge-by-platform
    /// convention (crates/clawhdf5-sign/examples/signing_bench.rs).
    fn merge_csv(
        csv_path: &std::path::Path,
        bucket: &str,
        region_pair: &str,
        new_rows: &[Row],
    ) -> std::io::Result<usize> {
        let refreshed_keys: std::collections::HashSet<(String, String)> = new_rows
            .iter()
            .map(|r| (r.region_pair.clone(), format!("{:.6}", r.dataset_size_gb)))
            .collect();

        // Accumulated across every row that will actually end up in the
        // file (preserved + new), so the header comment written below
        // never describes less than what's really in the file.
        let mut all_region_pairs: std::collections::BTreeSet<String> =
            std::collections::BTreeSet::new();
        let mut all_sizes_gb: std::collections::BTreeSet<String> =
            std::collections::BTreeSet::new();

        let mut kept_lines: Vec<String> = Vec::new();
        if csv_path.exists() {
            let existing = std::fs::read_to_string(csv_path)?;
            for line in existing.lines() {
                if line.starts_with('#') || line.starts_with("method,") {
                    continue;
                }
                let mut fields = line.split(',');
                let _method = fields.next();
                let _measurement = fields.next();
                let size_gb = fields.next().unwrap_or("");
                let _n_chunks = fields.next();
                let row_region_pair = fields.next().unwrap_or("");
                let key = (row_region_pair.to_string(), size_gb.to_string());
                if !refreshed_keys.contains(&key) {
                    all_region_pairs.insert(row_region_pair.to_string());
                    all_sizes_gb.insert(size_gb.to_string());
                    kept_lines.push(line.to_string());
                }
            }
        }
        for r in new_rows {
            all_region_pairs.insert(r.region_pair.clone());
            all_sizes_gb.insert(format!("{:.6}", r.dataset_size_gb));
        }
        // Fall back to the current run's region_pair if, for whatever
        // reason, no rows end up in the file at all — keeps the header
        // non-misleading rather than empty.
        if all_region_pairs.is_empty() {
            all_region_pairs.insert(region_pair.to_string());
        }
        let region_pairs: Vec<String> = all_region_pairs.into_iter().collect();
        let dataset_sizes_gb: Vec<String> = all_sizes_gb.into_iter().collect();

        let mut f = std::fs::File::create(csv_path)?;
        write_header(&mut f, bucket, &region_pairs, &dataset_sizes_gb)?;
        for line in &kept_lines {
            writeln!(f, "{line}")?;
        }
        for r in new_rows {
            write_row(&mut f, r)?;
        }
        Ok(kept_lines.len() + new_rows.len())
    }

    pub async fn main() -> Result<(), Box<dyn std::error::Error>> {
        let bucket = std::env::var("CLAW_S3_BUCKET").map_err(
            |_| "CLAW_S3_BUCKET must be set (see module docs for the full env var list)",
        )?;
        let aws_region = env_or("AWS_REGION", "us-east-1");
        let region_pair = env_or(
            "CLAW_CLOUD_EVAL_REGION_PAIR",
            &format!("{aws_region}_to_{aws_region}"),
        );
        // 5 GiB, not 5 * 1000^3 — matches S3's actual single-PutObject limit
        // (s3::upload doesn't implement multipart upload; see its doc comment).
        const MAX_SINGLE_PUT_MB: usize = 5 * 1024;
        let sizes_mb: Vec<usize> = env_or("CLAW_CLOUD_EVAL_SIZES_MB", "64")
            .split(',')
            .map(|s| {
                let mb: usize = s
                    .trim()
                    .parse()
                    .expect("CLAW_CLOUD_EVAL_SIZES_MB must be a comma list of integers");
                assert!(
                    mb < MAX_SINGLE_PUT_MB,
                    "CLAW_CLOUD_EVAL_SIZES_MB entry {mb} MB exceeds S3's 5 GiB single-PutObject \
                     limit ({MAX_SINGLE_PUT_MB} MB) — s3::upload doesn't implement multipart \
                     upload, so this would fail at the upload step, not just run slow"
                );
                mb
            })
            .collect();
        let chunk_bytes = env_usize("CLAW_CLOUD_EVAL_CHUNK_BYTES", 4096);
        assert!(
            chunk_bytes > 0,
            "CLAW_CLOUD_EVAL_CHUNK_BYTES must be greater than 0 (n_chunks is computed as \
             dataset_size / chunk_bytes)"
        );
        let trials = env_usize("CLAW_CLOUD_EVAL_TRIALS", 30);
        assert!(
            trials > 0,
            "CLAW_CLOUD_EVAL_TRIALS must be greater than 0 (the statistical protocol needs at \
             least one trial; leaf-index sampling divides by it)"
        );
        let warmup = env_usize("CLAW_CLOUD_EVAL_WARMUP", 5);

        println!("P2.5: Cloud / WAN evaluation (RQ6, §7.6)");
        println!("  bucket       : {bucket}");
        println!("  region_pair  : {region_pair}");
        println!("  sizes (MB)   : {sizes_mb:?}");
        println!("  chunk bytes  : {chunk_bytes}");
        println!("  trials/warmup: {trials}/{warmup}");
        println!();

        let mut all_rows: Vec<Row> = Vec::new();

        for &size_mb in &sizes_mb {
            let n_chunks = (size_mb * 1024 * 1024) / chunk_bytes;
            println!("Building {size_mb} MB test file ({n_chunks} chunks of {chunk_bytes}B)...");
            let (file_bytes, tree) = build_test_file(n_chunks, chunk_bytes);
            let object_len = file_bytes.len() as u64;
            let dataset_size_gb = object_len as f64 / (1024.0 * 1024.0 * 1024.0);

            let key = format!("cloud-eval/{size_mb}mb.h5");
            let location = S3Location {
                bucket: bucket.clone(),
                key: key.clone(),
            };

            println!(
                "Uploading {} bytes to s3://{bucket}/{key} ...",
                file_bytes.len()
            );
            let client = {
                let config = aws_config::load_defaults(aws_config::BehaviorVersion::latest()).await;
                aws_sdk_s3::Client::new(&config)
            };
            s3::upload(&client, &location, file_bytes).await?;

            let reader = S3Reader::new(client.clone(), location.clone());

            // Warmup: not recorded, just to avoid attributing connection
            // setup / DNS / TLS handshake cost to trial 1.
            for _ in 0..warmup {
                let layout = s3::resolve_companion_layout(
                    &reader,
                    DATASET_NAME,
                    s3::DEFAULT_METADATA_PREFETCH,
                )
                .await?;
                let padded = tree.padded_leaf_count() as u64;
                let _ = s3::fetch_chunk_proof(&reader, layout, 0, padded).await?;
            }

            // Spread sampled leaf indices evenly across the whole tree rather
            // than clustering near index 0: with `trials` << `n_chunks`,
            // `trial % n_chunks` degenerates to just `trial`, so every trial
            // would probe nearly-adjacent leaves (and nearly-identical proof
            // spans) instead of the varied tree positions a valid benchmark
            // needs to sample.
            let leaf_stride = (n_chunks / trials).max(1);

            for trial in 1..=trials {
                let leaf_idx = (((trial - 1) * leaf_stride) % n_chunks) as u64;
                let padded_leaf_count = tree.padded_leaf_count() as u64;

                // ── merkle_range_get: first-chunk proof latency + bytes ──
                // One metadata-prefix range-GET resolves both the companion
                // dataset's and the primary dataset's byte ranges (they
                // live in the same object) — matching what a real client
                // would do rather than re-fetching the same prefix twice.
                let t0 = Instant::now();
                let prefix = reader
                    .read_at(0, s3::DEFAULT_METADATA_PREFETCH as usize)
                    .await?;
                let companion_layout =
                    s3::parse_contiguous_layout(&prefix, &format!("merkle/{DATASET_NAME}"))?;
                let data_layout = s3::parse_contiguous_layout(&prefix, DATASET_NAME)?;
                let proof =
                    s3::fetch_chunk_proof(&reader, companion_layout, leaf_idx, padded_leaf_count)
                        .await?;
                let chunk_bytes_data = reader
                    .read_at(
                        data_layout.address + leaf_idx * chunk_bytes as u64,
                        chunk_bytes,
                    )
                    .await?;
                let leaf_hash_from_proof = proof.nodes[&proof.leaf_node_index];
                let recomputed_leaf_hash = HashAlg::Blake3.hash_leaf(&chunk_bytes_data);
                assert_eq!(
                    leaf_hash_from_proof, recomputed_leaf_hash,
                    "fetched chunk data does not match its proof-supplied leaf hash \
                     (S3 range-GET returned inconsistent bytes, or the layout resolver is wrong)"
                );
                let first_chunk_ms = t0.elapsed().as_secs_f64() * 1000.0;
                let range_get_bytes = (s3::DEFAULT_METADATA_PREFETCH as usize
                    + proof.bytes_transferred as usize
                    + chunk_bytes_data.len()) as f64;

                all_rows.push(Row {
                    method: "merkle_range_get",
                    measurement: "first_chunk_proof_latency_ms",
                    dataset_size_gb,
                    n_chunks,
                    region_pair: region_pair.clone(),
                    value: first_chunk_ms,
                    trial,
                });
                all_rows.push(Row {
                    method: "merkle_range_get",
                    measurement: "bytes_transferred",
                    dataset_size_gb,
                    n_chunks,
                    region_pair: region_pair.clone(),
                    value: range_get_bytes,
                    trial,
                });

                // ── merkle_range_get: full-dataset verify latency ──
                // Still requires reading every chunk byte (no way around
                // that for a *full* verification), but reads structure via
                // the companion range-GET rather than re-deriving it.
                let t0 = Instant::now();
                let full_data = reader
                    .read_at(data_layout.address, data_layout.size as usize)
                    .await?;
                let rebuilt: Vec<&[u8]> = full_data.chunks(chunk_bytes).collect();
                let rebuilt_tree = MerkleTree::from_chunks(&rebuilt, HashAlg::Blake3);
                assert_eq!(
                    rebuilt_tree.root(),
                    tree.root(),
                    "rebuilt root from downloaded data does not match the locally-known root"
                );
                let full_verify_ms = t0.elapsed().as_secs_f64() * 1000.0;
                all_rows.push(Row {
                    method: "merkle_range_get",
                    measurement: "full_verify_latency_ms",
                    dataset_size_gb,
                    n_chunks,
                    region_pair: region_pair.clone(),
                    value: full_verify_ms,
                    trial,
                });

                // ── full_redownload baseline: whole-object GetObject + rehash ──
                // Without Merkle proofs, there's no cheap way to check "just
                // one chunk" — verifying anything at all requires downloading
                // the whole object AND rehashing every chunk to rebuild the
                // root, so this baseline's cost (unlike merkle_range_get's)
                // is identical whether the caller wanted to check one chunk
                // or the whole dataset. Both timings below include that
                // rehash, not just the download — comparing raw download time
                // against merkle_range_get's fully-verified latency would
                // understate this baseline's real cost.
                let t0 = Instant::now();
                let whole_object = reader.read_at(0, object_len as usize).await?;
                let data_start = data_layout.address as usize;
                let data_end = data_start + data_layout.size as usize;
                let rebuilt: Vec<&[u8]> = whole_object[data_start..data_end]
                    .chunks(chunk_bytes)
                    .collect();
                let rebuilt_tree = MerkleTree::from_chunks(&rebuilt, HashAlg::Blake3);
                assert_eq!(
                    rebuilt_tree.root(),
                    tree.root(),
                    "full_redownload: rebuilt root from downloaded data does not match the \
                     locally-known root"
                );
                let whole_object_ms = t0.elapsed().as_secs_f64() * 1000.0;
                all_rows.push(Row {
                    method: "full_redownload",
                    measurement: "first_chunk_proof_latency_ms",
                    dataset_size_gb,
                    n_chunks,
                    region_pair: region_pair.clone(),
                    value: whole_object_ms,
                    trial,
                });
                all_rows.push(Row {
                    method: "full_redownload",
                    measurement: "full_verify_latency_ms",
                    dataset_size_gb,
                    n_chunks,
                    region_pair: region_pair.clone(),
                    value: whole_object_ms,
                    trial,
                });
                all_rows.push(Row {
                    method: "full_redownload",
                    measurement: "bytes_transferred",
                    dataset_size_gb,
                    n_chunks,
                    region_pair: region_pair.clone(),
                    value: whole_object.len() as f64,
                    trial,
                });

                if trial % 5 == 0 || trial == trials {
                    println!("  [{size_mb}MB] trial {trial}/{trials} done");
                }
            }
        }

        let out_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../crates/clawhdf5-format/benches/results");
        std::fs::create_dir_all(&out_dir)?;
        let csv_path = out_dir.join("phase2-cloud.csv");

        println!(
            "\nMerging {} row(s) for region_pair={region_pair} into {}",
            all_rows.len(),
            csv_path.display()
        );
        let total = merge_csv(&csv_path, &bucket, &region_pair, &all_rows)?;
        println!(
            "Done — {} new rows, {total} total rows in file.",
            all_rows.len()
        );

        Ok(())
    }
}

#[cfg(feature = "s3")]
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    run::main().await
}
