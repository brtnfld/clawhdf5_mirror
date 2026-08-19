//! Per-storage-class I/O instrumentation shared by the P2.8c contiguous
//! benchmarks.
//!
//! Kept in one place because the two benches must agree on what
//! `bytes_transferred` means: a byte column whose provenance differs between
//! the measurement and its baseline cannot be compared, and the failure mode
//! on a parallel filesystem is a silent zero rather than an error.
//!
//! Included with `#[path = "common/storage_io.rs"] mod storage_io;` -- a
//! subdirectory of `benches/` so Cargo does not auto-discover it as a bench
//! target of its own.

use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom};
use std::os::unix::fs::{FileExt, OpenOptionsExt};
use std::os::unix::io::AsRawFd;
use std::path::Path;
use std::process::Command;
use std::time::Instant;

// ===== measured-I/O plumbing =====
//
// "Bytes transferred" is the load-bearing measurement of P2.8c, and the
// counter that produces it is not portable across storage classes.
// `/proc/self/io`'s `read_bytes` counts traffic issued to a *block device*.
// A parallel filesystem has no local block device in the path, so on Lustre
// that counter sits at ~0 and a column filled from it reads as a plausible
// "almost no I/O" result on precisely the class the experiment exists to
// characterise -- a silent failure, not a noisy one. The counter is therefore
// chosen per class and its identity recorded in `bytes_source` beside the
// number it produced.

/// Which counter supplied `bytes_transferred` for a row.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BytesSource {
    /// Block-layer bytes from `/proc/self/io` (`read_bytes`). Local devices.
    ProcSelfIo,
    /// Network bytes fetched from the OSTs, from `osc.*.stats` (`ost_read`).
    /// The device-level number on Lustre, and the only one that stays correct
    /// when readahead is in the path.
    OscStats,
    /// Lustre client bytes from `llite.*.stats` (`read_bytes`). VFS-level:
    /// what the application asked for, not what crossed the wire. Kept as a
    /// last resort and labelled as such, because under readahead it
    /// understates device traffic and under a cache hit it reports traffic
    /// that did not happen.
    LliteStats,
    /// Bytes the process issued under `O_DIRECT`. With the page cache and
    /// readahead both out of the path, issued *is* transferred by
    /// construction. This is the counter that survives a site where
    /// `llite.*.stats` lives in root-only debugfs, which is the common case
    /// on Lustre 2.10+ and cannot be assumed away inside a batch allocation.
    ODirectIssued,
}

impl BytesSource {
    pub fn as_str(self) -> &'static str {
        match self {
            BytesSource::ProcSelfIo => "proc_self_io",
            BytesSource::OscStats => "osc_stats",
            BytesSource::LliteStats => "llite_stats_vfs_level",
            BytesSource::ODirectIssued => "o_direct_issued",
        }
    }
}

/// Bytes this process has read from the block layer, per `/proc/self/io`.
/// Meaningful on NVMe and rotational media; ~0 on a network filesystem.
pub fn proc_read_bytes() -> u64 {
    let s = std::fs::read_to_string("/proc/self/io").unwrap_or_default();
    for line in s.lines() {
        if let Some(v) = line.strip_prefix("read_bytes:") {
            return v.trim().parse().unwrap_or(0);
        }
    }
    0
}

/// `lctl get_param -n <param>`, or `None` when lctl is absent or the
/// parameter is unreadable. Since Lustre 2.10 the stats files live in
/// debugfs, which is root-only at many sites; every caller therefore treats
/// absence as a fact to record rather than an error to abort on.
pub fn lctl_get(param: &str) -> Option<String> {
    let out = Command::new("lctl")
        .args(["get_param", "-n", param])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8_lossy(&out.stdout).into_owned();
    if s.trim().is_empty() { None } else { Some(s) }
}

/// Sum one Lustre stats counter across every client mount. Stats lines read
/// `name <count> samples [unit] <min> <max> <sum>`; `want_sum` picks the
/// trailing byte total, otherwise the sample count (which is the RPC count
/// for `osc.*.stats`).
pub fn lustre_stat(param: &str, key: &str, want_sum: bool) -> Option<u64> {
    let text = lctl_get(param)?;
    let mut total = 0u64;
    let mut seen = false;
    for line in text.lines() {
        let f: Vec<&str> = line.split_whitespace().collect();
        if f.len() >= 7 && f[0] == key {
            let v = if want_sum { f[f.len() - 1] } else { f[1] };
            if let Ok(n) = v.parse::<u64>() {
                total += n;
                seen = true;
            }
        }
    }
    if seen { Some(total) } else { None }
}

pub fn llite_read_bytes() -> Option<u64> {
    lustre_stat("llite.*.stats", "read_bytes", true)
}

/// Per-OSC read RPC count -- the Lustre analogue of a discontiguous I/O
/// operation, and what the `io_ops` column means on this class.
pub fn osc_read_rpcs() -> Option<u64> {
    lustre_stat("osc.*.stats", "ost_read", false)
}

/// Network-level bytes fetched from the OSTs, when the client records
/// `ost_read` in bytes rather than in microseconds. This is the counter the
/// *buffered* arm needs: `llite.*.stats` `read_bytes` is VFS-level -- it
/// counts what the application asked for, so under readahead it understates
/// what crossed the wire, and under a cache hit it reports traffic that never
/// happened. The unit token is checked rather than assumed because the schema
/// varies across Lustre releases; where it is not bytes, the caller records
/// that the device-byte column is unavailable instead of substituting a
/// number that means something else.
pub fn osc_read_bytes() -> Option<u64> {
    let text = lctl_get("osc.*.stats")?;
    let mut total = 0u64;
    let mut seen = false;
    for line in text.lines() {
        let f: Vec<&str> = line.split_whitespace().collect();
        if f.len() >= 7 && f[0] == "ost_read" && f[3].contains("byte") {
            if let Ok(n) = f[f.len() - 1].parse::<u64>() {
                total += n;
                seen = true;
            }
        }
    }
    if seen { Some(total) } else { None }
}

/// One snapshot of every counter, taken on both sides of the measured region.
#[derive(Clone, Copy, Default)]
pub struct Counters {
    pub proc_bytes: u64,
    pub llite_bytes: Option<u64>,
    pub osc_bytes: Option<u64>,
    pub osc_rpcs: Option<u64>,
}

impl Counters {
    pub fn snapshot(lustre: bool) -> Self {
        Self {
            proc_bytes: proc_read_bytes(),
            llite_bytes: if lustre { llite_read_bytes() } else { None },
            osc_bytes: if lustre { osc_read_bytes() } else { None },
            osc_rpcs: if lustre { osc_read_rpcs() } else { None },
        }
    }
}

pub fn delta(before: Option<u64>, after: Option<u64>) -> Option<u64> {
    match (before, after) {
        (Some(b), Some(a)) => Some(a.saturating_sub(b)),
        _ => None,
    }
}

/// Dump the client readahead counters, so the share of measured bytes that
/// readahead rather than the selection is responsible for stays separable.
/// Written to stderr, which the job script captures next to the CSV.
pub fn dump_readahead_stats(when: &str) {
    if let Some(s) = lctl_get("llite.*.read_ahead_stats") {
        eprintln!("--- llite read_ahead_stats ({when}) ---");
        eprint!("{s}");
    }
}

// ===== O_DIRECT read path =====

/// `O_DIRECT` alignment unit. Lustre wants page alignment on the buffer, the
/// file offset and the length; 4 KiB satisfies every device tested here.
pub const ALIGN: usize = 4096;

/// Minimum alignment a short `O_DIRECT` return can land on: the device
/// logical block size, 512 on every device tested. `ALIGN` is a safe
/// superset for the *requests* the benchmark issues.
pub const DIO_MIN_ALIGN: usize = 512;

pub fn align_down(x: u64) -> u64 {
    x & !(ALIGN as u64 - 1)
}

pub fn align_up(x: u64) -> u64 {
    (x + ALIGN as u64 - 1) & !(ALIGN as u64 - 1)
}

/// A page-aligned buffer. Over-allocating by one page and slicing forward to
/// the next boundary gets `O_DIRECT`'s alignment requirement without a custom
/// allocator; the `Vec` never reallocates, so the offset stays valid.
pub struct AlignedBuf {
    pub raw: Vec<u8>,
    pub off: usize,
    pub len: usize,
}

impl AlignedBuf {
    pub fn new(len: usize) -> Self {
        let raw = vec![0u8; len + ALIGN];
        let off = (ALIGN - (raw.as_ptr() as usize % ALIGN)) % ALIGN;
        Self { raw, off, len }
    }
    pub fn as_mut(&mut self) -> &mut [u8] {
        let (o, l) = (self.off, self.len);
        &mut self.raw[o..o + l]
    }
}

/// `pread` until the buffer is full or EOF, returning bytes obtained. The
/// aligned span of the last run in the file necessarily runs past EOF (a
/// 4 GB dataset is not a whole number of pages), so a short read at the end
/// is expected rather than an error.
pub fn pread_upto(f: &File, buf: &mut [u8], off: u64) -> std::io::Result<usize> {
    let mut done = 0usize;
    while done < buf.len() {
        let n = f.read_at(&mut buf[done..], off + done as u64)?;
        if n == 0 {
            break;
        }
        done += n;
        // A large O_DIRECT read may come back short -- observed on ext4 at a
        // 4 MiB request. Continuing is safe as long as the next offset and
        // the next buffer address are still device-aligned, which they are
        // whenever the partial return is a multiple of the logical block
        // size. Anything else would make the next pread fail with EINVAL, so
        // fail loudly here rather than silently mis-measure.
        if done < buf.len() && done % DIO_MIN_ALIGN != 0 {
            return Err(std::io::Error::other(format!(
                "partial pread of {done} bytes is not {DIO_MIN_ALIGN}-aligned;                  O_DIRECT continuation would fail"
            )));
        }
    }
    Ok(done)
}

pub fn open_maybe_direct(path: &Path, direct: bool) -> std::io::Result<File> {
    let mut opts = OpenOptions::new();
    opts.read(true);
    if direct {
        opts.custom_flags(libc::O_DIRECT);
    }
    opts.open(path)
}

/// Random-access reader over the dataset's byte runs.
///
/// `O_DIRECT` is how this benchmark controls two confounds at once on a
/// parallel filesystem. The 4 GB working set fits entirely in a compute
/// node's RAM, so from trial 2 onward a buffered read measures memory
/// bandwidth; and Lustre reads ahead on the order of 64 MiB per file by
/// default, which exceeds every run length in the sweep and would flatten
/// the predicted run-length cliff by configuration rather than by physics.
/// Dropping caches needs root, which a batch allocation does not have.
/// `O_DIRECT` needs no privilege and removes both at once.
pub struct RunReader {
    pub file: File,
    pub direct: bool,
    pub bounce: AlignedBuf,
    pub issued_bytes: u64,
    pub issued_ops: u64,
}

impl RunReader {
    pub fn open(path: &Path, direct: bool, max_run: u64) -> std::io::Result<Self> {
        let file = open_maybe_direct(path, direct)?;
        // A run can be unaligned at both ends, so its aligned span exceeds it
        // by up to two pages.
        let cap = if direct {
            align_up(max_run + 2 * ALIGN as u64) as usize
        } else {
            ALIGN
        };
        Ok(Self {
            file,
            direct,
            bounce: AlignedBuf::new(cap),
            issued_bytes: 0,
            issued_ops: 0,
        })
    }

    /// Read the byte run `[off, off+len)` into `out`, counting what the
    /// process actually asked the device for. Under `O_DIRECT` the aligned
    /// span *is* what moved, which is the block tax the cost model predicts,
    /// so counting it here is a measurement and not a model.
    pub fn read_run(&mut self, off: u64, len: u64, out: &mut [u8]) -> std::io::Result<()> {
        if !self.direct {
            self.file.seek(SeekFrom::Start(off))?;
            self.file.read_exact(out)?;
            self.issued_bytes += len;
            self.issued_ops += 1;
            return Ok(());
        }
        let lo = align_down(off);
        let hi = align_up(off + len);
        let span = (hi - lo) as usize;
        let need = (off + len - lo) as usize;
        let got = {
            let buf = &mut self.bounce.as_mut()[..span];
            pread_upto(&self.file, buf, lo)?
        };
        if got < need {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                format!("short O_DIRECT read at {lo}: got {got}, need {need}"),
            ));
        }
        let skip = (off - lo) as usize;
        out.copy_from_slice(&self.bounce.as_mut()[skip..skip + len as usize]);
        self.issued_bytes += got as u64;
        self.issued_ops += 1;
        Ok(())
    }
}

/// Sequential reader used by the tree builds and the audit throughput run.
/// Serves arbitrary-size reads from a page-aligned internal buffer, so a
/// caller asking for 4,000,000-byte planes still works under `O_DIRECT`.
pub struct SeqReader {
    pub file: File,
    pub buf: AlignedBuf,
    pub base: u64,
    pub filled: usize,
    pub cursor: usize,
}

impl SeqReader {
    pub fn open(path: &Path, direct: bool) -> std::io::Result<Self> {
        Ok(Self {
            file: open_maybe_direct(path, direct)?,
            buf: AlignedBuf::new(4 << 20),
            base: 0,
            filled: 0,
            cursor: 0,
        })
    }
}

impl Read for SeqReader {
    fn read(&mut self, out: &mut [u8]) -> std::io::Result<usize> {
        if self.cursor == self.filled {
            let base = self.base;
            let got = {
                let buf = self.buf.as_mut();
                pread_upto(&self.file, buf, base)?
            };
            if got == 0 {
                return Ok(0);
            }
            self.filled = got;
            self.cursor = 0;
            self.base += got as u64;
        }
        let n = out.len().min(self.filled - self.cursor);
        let c = self.cursor;
        out[..n].copy_from_slice(&self.buf.as_mut()[c..c + n]);
        self.cursor += n;
        Ok(n)
    }
}

/// Evict `path`'s clean page-cache pages. Returns whether the call reported
/// success -- which on Lustre is *not* proof that pages were dropped, hence
/// `O_DIRECT` for the runs that matter.
/// Evict `path` from the client page cache.
///
/// `/proc/sys/vm/drop_caches` needs root, which a batch allocation does not
/// have, and `lctl set_param` is normally refused to unprivileged users. Two
/// mechanisms remain, both unprivileged: `posix_fadvise(DONTNEED)`, and on
/// Lustre `lfs ladvise -a dontneed`, which is the client-native equivalent
/// and the one that works when llite does not wire fadvise through.
///
/// The return value says a mechanism *reported* success, which is not proof
/// pages were dropped -- see `eviction_works`, which measures it.
pub fn evict_cache(path: &Path) -> bool {
    let mut ok = false;
    if let Ok(f) = File::open(path) {
        ok = unsafe { libc::posix_fadvise(f.as_raw_fd(), 0, 0, libc::POSIX_FADV_DONTNEED) == 0 };
    }
    if is_lustre(path) {
        let advised = Command::new("lfs")
            .args(["ladvise", "-a", "dontneed"])
            .arg(path)
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false);
        ok = ok || advised;
    }
    ok
}

/// Does cache eviction actually work on this filesystem?
///
/// Warm the first 256 MiB, evict, read it again, and see whether the client
/// went back to the servers. On Lustre that is answered exactly by the OST
/// read RPC count: zero RPCs means the read was served from cache and every
/// buffered trial in the run would be measuring memory bandwidth. Off Lustre
/// it falls back to comparing elapsed time, which is weaker but still catches
/// a total failure to evict.
///
/// Run once per job and recorded, so a cache-warm campaign is a stated fact
/// rather than a spectacular storage result nobody questioned.
pub fn eviction_works(path: &Path, lustre: bool) -> Option<bool> {
    const SPAN: usize = 256 * 1024 * 1024;
    let warm = |p: &Path| -> std::io::Result<(f64, Option<u64>)> {
        let before = if lustre { osc_read_rpcs() } else { None };
        let mut f = SeqReader::open(p, false)?;
        let mut buf = vec![0u8; 8 << 20];
        let mut read = 0usize;
        let t = Instant::now();
        while read < SPAN {
            let n = f.read(&mut buf)?;
            if n == 0 {
                break;
            }
            read += n;
        }
        let dt = t.elapsed().as_secs_f64();
        let after = if lustre { osc_read_rpcs() } else { None };
        Ok((dt, delta(before, after)))
    };

    let _ = warm(path).ok()?; // populate the cache
    let evicted = evict_cache(path);
    let (dt_cold, rpcs) = warm(path).ok()?;
    match rpcs {
        Some(n) => {
            eprintln!(
                "### cache eviction check: {} OST read RPCs after evict ({}), {:.2}s",
                n,
                if evicted { "reported ok" } else { "reported failure" },
                dt_cold
            );
            Some(n > 0)
        }
        None => {
            eprintln!("### cache eviction check: no RPC counter; cold re-read took {dt_cold:.2}s");
            None
        }
    }
}

// ===== device and layout parameters =====

pub fn is_lustre(path: &Path) -> bool {
    Command::new("lfs")
        .arg("getstripe")
        .arg("-d")
        .arg(path)
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// One `lfs getstripe` field for a file. A composite (PFL) layout prints one
/// value per component; distinct values are reported as `composite`, because
/// a file whose first component lives on a flash tier and whose tail lives on
/// disk is not one device and must not be labelled as one.
pub fn lfs_field(path: &Path, flag: &str) -> Option<String> {
    let out = Command::new("lfs")
        .args(["getstripe", flag])
        .arg(path)
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let vals: Vec<&str> = text
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && l.chars().all(|c| c.is_ascii_digit() || c == '-'))
        .collect();
    if vals.is_empty() {
        return None;
    }
    let first = vals[0].to_string();
    if vals.iter().any(|v| *v != vals[0]) {
        eprintln!(
            "WARNING: {} has a composite (PFL) layout -- {} differs across components ({:?}).",
            path.display(),
            flag,
            vals
        );
        eprintln!("         This row is not a single device. Apply an explicit flat");
        eprintln!("         `lfs setstripe` and regenerate the dataset before trusting it.");
        return Some("composite".to_string());
    }
    Some(first)
}

/// Client RPC size in bytes: `max_pages_per_rpc` x page size.
pub fn lustre_rpc_bytes() -> Option<u64> {
    let text = lctl_get("osc.*.max_pages_per_rpc")?;
    let pages: u64 = text
        .lines()
        .filter_map(|l| l.trim().parse::<u64>().ok())
        .max()?;
    let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as u64;
    Some(pages * page)
}

pub fn lustre_readahead_bytes() -> Option<u64> {
    let text = lctl_get("llite.*.max_read_ahead_per_file_mb")?;
    let mb: u64 = text.lines().filter_map(|l| l.trim().parse::<u64>().ok()).next()?;
    Some(mb * 1024 * 1024)
}

/// The governing block size *B* for the run-length rule, and which parameter
/// supplied it. On Lustre *B* is not a single number: it is the smaller of
/// the stripe size and the client RPC size, and at stripe count 1 the RPC
/// size governs alone. A hardcoded 512 here would make
/// `model_bytes_predicted` meaningless on exactly the class this run exists
/// to add.
pub fn detect_block_size(path: &Path) -> (u64, String) {
    if is_lustre(path) {
        let stripe = lfs_field(path, "-S").and_then(|s| s.parse::<u64>().ok());
        let count = lfs_field(path, "-c").and_then(|s| s.parse::<i64>().ok());
        let rpc = lustre_rpc_bytes();
        return match (stripe, rpc) {
            (Some(s), Some(r)) if count == Some(1) => {
                (r.min(s), "osc max_pages_per_rpc (stripe_count=1)".into())
            }
            (Some(s), Some(r)) if r < s => (r, "osc max_pages_per_rpc (< stripe_size)".into()),
            (Some(s), Some(_)) => (s, "lfs stripe_size (<= rpc size)".into()),
            (Some(s), None) => (s, "lfs stripe_size (rpc size unknown)".into()),
            (None, Some(r)) => (r, "osc max_pages_per_rpc (stripe size unknown)".into()),
            (None, None) => (512, "default (lustre parameters unreadable)".into()),
        };
    }
    (512, "default logical block size".into())
}

/// Everything about the device, the layout and the measurement mode that
/// every row must carry for the cost model to be re-evaluable on hardware
/// not tested here. Flag beats environment beats probe.
pub struct BenchEnv {
    pub storage_class: String,
    pub block_size: u64,
    pub block_size_source: String,
    pub stripe_count: String,
    pub stripe_size: String,
    pub readahead_bytes: String,
    pub o_direct: bool,
    pub lustre: bool,
    pub seq_read_mbps: f64,
    pub seq_write_mbps: String,
}

impl BenchEnv {
    pub fn csv_header() -> &'static str {
        "storage_class,block_size_bytes,block_size_source,stripe_count,stripe_size_bytes,\
         readahead_bytes,o_direct,seq_read_mbps,seq_write_mbps"
    }
    pub fn csv_fields(&self) -> String {
        format!(
            "{},{},{},{},{},{},{},{:.0},{}",
            self.storage_class,
            self.block_size,
            self.block_size_source,
            self.stripe_count,
            self.stripe_size,
            self.readahead_bytes,
            u8::from(self.o_direct),
            self.seq_read_mbps,
            self.seq_write_mbps,
        )
    }
}

impl BenchEnv {
    /// Read every device, layout and mode parameter this row must carry.
    /// Flag beats environment beats probe; the environment names are the ones
    /// `benches/scripts/run-storage-bench.sh` exports, so the same binary
    /// runs standalone and under the sweep driver without a wrapper.
    ///
    /// `path` is the data file, not the directory: a Lustre layout is fixed
    /// at file creation, so a directory that was `lfs setstripe`d after the
    /// file was written would report a layout the file does not have.
    pub fn detect<F>(get: F, path: &Path, o_direct: bool) -> std::io::Result<Self>
    where
        F: Fn(&str, &str) -> String,
    {
        let get_env = |k: &str, e: &str, d: &str| -> String {
            let v = get(k, "");
            if !v.is_empty() {
                return v;
            }
            std::env::var(e)
                .ok()
                .filter(|s| !s.trim().is_empty())
                .unwrap_or_else(|| d.to_string())
        };

        let lustre = is_lustre(path);
        let (det_block, det_source) = detect_block_size(path);
        let (block_size, block_size_source) =
            match get_env("--block-size", "BENCH_BLOCK_SIZE_BYTES", "").parse::<u64>() {
                Ok(b) if b > 0 => (
                    b,
                    get("--block-size-source", "supplied on the command line"),
                ),
                _ => (det_block, det_source),
            };
        let mut stripe_count = get_env("--stripe-count", "BENCH_STRIPE_COUNT", "");
        let mut stripe_size = get_env("--stripe-size", "BENCH_STRIPE_SIZE_BYTES", "");
        let mut readahead_bytes = get("--readahead-bytes", "");
        if lustre {
            if stripe_count.is_empty() {
                stripe_count = lfs_field(path, "-c").unwrap_or_default();
            }
            if stripe_size.is_empty() {
                stripe_size = lfs_field(path, "-S").unwrap_or_default();
            }
            if readahead_bytes.is_empty() {
                readahead_bytes = lustre_readahead_bytes()
                    .map(|v| v.to_string())
                    .unwrap_or_default();
            }
        }

        let seq = seq_read_mbps(path, o_direct)?;
        eprintln!(
            "sequential read bandwidth: {seq:.0} MB/s ({})",
            if o_direct { "O_DIRECT" } else { "buffered" }
        );

        let env = BenchEnv {
            storage_class: get_env(
                "--storage-class",
                "BENCH_STORAGE_CLASS",
                if lustre { "lustre" } else { "unknown" },
            ),
            block_size,
            block_size_source,
            stripe_count,
            stripe_size,
            readahead_bytes,
            o_direct,
            lustre,
            seq_read_mbps: seq,
            seq_write_mbps: get_env("--seq-write-mbps", "BENCH_SEQ_WRITE_MBPS", ""),
        };

        eprintln!(
            "class={} B={} ({}) stripe={}x{} readahead={} o_direct={}",
            env.storage_class,
            env.block_size,
            env.block_size_source,
            if env.stripe_count.is_empty() { "-" } else { env.stripe_count.as_str() },
            if env.stripe_size.is_empty() { "-" } else { env.stripe_size.as_str() },
            if env.readahead_bytes.is_empty() { "-" } else { env.readahead_bytes.as_str() },
            u8::from(env.o_direct)
        );
        if !env.o_direct {
            // The 4 GB working set fits in a compute node's RAM many times
            // over, so a buffered run whose eviction silently fails measures
            // memory bandwidth and presents as a spectacular storage result.
            match eviction_works(path, env.lustre) {
                Some(true) => eprintln!("### cache eviction verified"),
                Some(false) => {
                    eprintln!("WARNING: cache eviction does NOT work here -- the re-read was");
                    eprintln!("         served entirely from cache. Buffered rows from this");
                    eprintln!("         run are cache-warm and must not be pooled with cold");
                    eprintln!("         ones. Use the O_DIRECT arm instead.");
                }
                None => eprintln!("### cache eviction unverified (no RPC counter)"),
            }
        }
        if env.lustre {
            dump_readahead_stats("before");
            if llite_read_bytes().is_none() && !env.o_direct {
                eprintln!("WARNING: llite.*.stats is unreadable (root-only debugfs?) and O_DIRECT");
                eprintln!("         is off. /proc/self/io reads ~0 on Lustre, so");
                eprintln!("         bytes_transferred would be a silent zero on exactly the");
                eprintln!("         class this run exists to add. Rerun with --o-direct.");
            }
        }
        Ok(env)
    }
}

/// Pick the counter that actually measures something on this class, once per
/// run rather than per row, and say so on stderr when the choice is forced.
pub fn choose_bytes_source(env: &BenchEnv) -> BytesSource {
    if !env.lustre {
        return BytesSource::ProcSelfIo;
    }
    // Device-level first. Under O_DIRECT the issued count is exact and needs
    // no privileged parameter, which is why that arm always has a trustworthy
    // byte column; the buffered arm depends on the site exposing osc stats.
    if osc_read_bytes().is_some() {
        return BytesSource::OscStats;
    }
    if env.o_direct {
        return BytesSource::ODirectIssued;
    }
    if llite_read_bytes().is_some() {
        eprintln!("WARNING: falling back to llite.*.stats, which is VFS-level: the");
        eprintln!("         bytes_transferred column will report what was requested,");
        eprintln!("         not what crossed the wire. Prefer the O_DIRECT arm for any");
        eprintln!("         claim that rests on bytes.");
        return BytesSource::LliteStats;
    }
    BytesSource::ProcSelfIo
}

/// Measured sequential read bandwidth for the device holding `path`, so the
/// cost model can be re-evaluated for untested hardware from this row alone.
/// Reads a 512 MiB prefix; under `O_DIRECT` that is device bandwidth by
/// construction, otherwise the file's cache is evicted first.
pub fn seq_read_mbps(path: &Path, direct: bool) -> std::io::Result<f64> {
    const SPAN: usize = 512 * 1024 * 1024;
    if !direct {
        evict_cache(path);
    }
    let mut f = SeqReader::open(path, direct)?;
    let mut buf = vec![0u8; 8 * 1024 * 1024];
    let mut read = 0usize;
    let t = Instant::now();
    while read < SPAN {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        read += n;
    }
    let dt = t.elapsed().as_secs_f64();
    Ok((read as f64) / dt / 1.0e6)
}

