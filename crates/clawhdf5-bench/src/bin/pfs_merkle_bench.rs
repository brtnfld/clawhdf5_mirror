//! Parallel-filesystem study of the Merkle verification grid (S2-D2 Paper A).
//!
//! Every Lustre number before this binary came from one process on one node
//! (`srun -N1 -n1`), which characterises how a single client sees the
//! filesystem, not how the construction behaves under parallel access. This
//! bench adds the three experiments that need many clients:
//!
//! * **E1 `build`** -- distributed tree construction. Each rank owns a
//!   power-of-two-aligned block of `S = padded / P` leaves, reads and hashes
//!   it, and builds its own subtree. Because the block is aligned, the
//!   subtree occupies one contiguous slice of every level of the level-order
//!   node array (positions `[r*2^j, (r+1)*2^j)` at depth `k + j`, with
//!   `k = log2 P`), so only the `P` subtree roots ever cross the network and
//!   every rank can write its share of the companion without coordination.
//!   Compared against `flat`: the whole-dataset hash ClawHDF5 ships today,
//!   which is a single stream by definition and cannot use a second client.
//! * **E2 `verify`** -- concurrent partial verification. Every rank verifies
//!   random `k`-leaf selections against the companion at once, the access
//!   pattern of an analysis campaign. Measures aggregate verified bandwidth
//!   and whether the (small, single-OST) companion becomes the hotspot.
//! * **E3 `update`** -- parallel writers. Eager (every chunk update writes
//!   its full path to the companion, the shared top levels serialised through
//!   rank 0) versus lazy (paths maintained in memory, one pass per epoch).
//!   Measures write amplification and update throughput as writers scale.
//!
//! `read` is the pure-I/O control (same ranks, same byte ranges, no hashing),
//! and `gen` writes a deterministic synthetic file in parallel, which a
//! multi-terabyte run needs.
//!
//! The companion here is a raw file holding the level-order node array
//! (`MerkleTree::nodes()` byte for byte), not an HDF5 dataset: ClawHDF5's
//! `FileWriter` is serial, so a raw file is the only way to measure a
//! parallel companion write at all. `--check` rebuilds the tree serially with
//! `MerkleTree::from_leaf_hashes` and compares root and companion bytes.
//!
//! ```text
//! mpirun -np 8 pfs_merkle_bench --mode gen    --data F --size 4G
//! mpirun -np 8 pfs_merkle_bench --mode build  --data F --companion C --leaf-bytes 1M --strategy distributed
//! mpirun -np 8 pfs_merkle_bench --mode verify --data F --companion C --k 16
//! mpirun -np 8 pfs_merkle_bench --mode update --data F --companion C --strategy lazy --updates 64
//! ```
//!
//! Common flags: `--alg sha256|blake3|k12`, `--leaf-bytes 1M` or
//! `--dims 1000x1000x1000 --elem 4 [--target 1M]` (the prefix grid),
//! `--bytes N` (use only the first N bytes), `--o-direct`, `--threads T`
//! (rayon threads per rank), `--trials 5 --warmup 1`, `--check`,
//! `--warm-server` (skip dropping the Lustre servers' cache before each trial;
//! the default asks for it, since O_DIRECT bypasses only the client cache),
//! `--offset B --offset-step S` (read/flat/build: trial t uses the region at
//! B + t*S, so no trial rereads data; Orion keeps repeatedly read data in
//! server memory whatever ladvise says).
//! The rank count must be a power of two.

#[cfg(feature = "pfs")]
#[path = "../../../clawhdf5-format/benches/common/storage_io.rs"]
#[allow(dead_code)]
mod storage_io;

#[cfg(feature = "pfs")]
fn main() {
    pfs::main();
}

#[cfg(not(feature = "pfs"))]
fn main() {
    eprintln!("pfs_merkle_bench requires the `pfs` feature.");
    eprintln!("Run: cargo build --release -p clawhdf5-bench --features pfs --bin pfs_merkle_bench");
    std::process::exit(1);
}

#[cfg(feature = "pfs")]
mod pfs {
    use crate::storage_io::{
        ALIGN, AlignedBuf, align_down, align_up, evict_cache, is_lustre, lfs_field,
        open_maybe_direct, pread_upto,
    };
    use clawhdf5_format::merkle::{HashAlg, MerkleTree};
    use clawhdf5_format::verification_grid;
    use mpi::collective::SystemOperation;
    use mpi::topology::SimpleCommunicator;
    use mpi::traits::*;
    use rayon::prelude::*;
    use std::fs::{File, OpenOptions};
    use std::ops::Range;
    use std::os::unix::fs::{FileExt, OpenOptionsExt};
    use std::path::{Path, PathBuf};
    use std::time::Instant;

    type H = [u8; 32];
    const NODE: u64 = 32;
    const TAG_EAGER: i32 = 7;
    /// `--check` re-reads the whole file on rank 0; refuse it where that
    /// would dominate the allocation.
    const CHECK_LIMIT: u64 = 16 << 30;

    // ===== arguments =====

    struct Args(Vec<String>);

    impl Args {
        fn get(&self, k: &str, d: &str) -> String {
            self.0
                .iter()
                .position(|a| a == k)
                .and_then(|i| self.0.get(i + 1))
                .cloned()
                .unwrap_or_else(|| d.to_string())
        }
        fn has(&self, k: &str) -> bool {
            self.0.iter().any(|a| a == k)
        }
        fn num(&self, k: &str, d: &str) -> u64 {
            let v = self.get(k, d);
            parse_size(&v).unwrap_or_else(|| die(&format!("{k}: cannot parse '{v}'")))
        }
    }

    /// Every rank parses the same arguments and reaches the same verdict, so
    /// exiting on all of them is a clean shutdown rather than a hang.
    fn die(msg: &str) -> ! {
        eprintln!("pfs_merkle_bench: {msg}");
        std::process::exit(2);
    }

    /// `4096`, `64K`, `1M`, `4G`, `2T` -- binary multiples.
    fn parse_size(s: &str) -> Option<u64> {
        let s = s.trim();
        let (num, mul) = match s.chars().last()? {
            'K' | 'k' => (&s[..s.len() - 1], 1u64 << 10),
            'M' | 'm' => (&s[..s.len() - 1], 1 << 20),
            'G' | 'g' => (&s[..s.len() - 1], 1 << 30),
            'T' | 't' => (&s[..s.len() - 1], 1 << 40),
            _ => (s, 1),
        };
        num.parse::<u64>().ok().map(|n| n * mul)
    }

    fn parse_alg(s: &str) -> HashAlg {
        match s {
            "sha256" => HashAlg::Sha256,
            "blake3" => HashAlg::Blake3,
            "k12" => HashAlg::K12,
            _ => die(&format!("--alg must be sha256, blake3 or k12, not '{s}'")),
        }
    }

    fn alg_name(a: HashAlg) -> &'static str {
        match a {
            HashAlg::Sha256 => "sha256",
            HashAlg::Blake3 => "blake3",
            HashAlg::K12 => "k12",
        }
    }

    // ===== leaf layout =====

    /// How the dataset's bytes are cut into leaves. `Fixed` is a 1-D grid of
    /// equal byte runs; `Grid` is the shipped dimension-aligned prefix grid
    /// (`verification_grid`), whose leaves are also single contiguous runs in
    /// file order, so both reduce to a sequence of consecutive byte ranges.
    enum Geom {
        Fixed(u64),
        Grid { grid: Vec<u64>, dims: Vec<u64>, elem: u32 },
    }

    struct Layout {
        file_bytes: u64,
        geom: Geom,
        n_leaves: u64,
        padded: u64,
        /// log2(padded): the global depth of the leaf level.
        d: u32,
        ranks: u64,
        /// log2(ranks): the global depth of the subtree roots.
        k: u32,
        /// Padded leaves per rank.
        s: u64,
        /// log2(s): levels in a rank's subtree above its leaves.
        sub: u32,
        /// File offset of the dataset's first byte. Nonzero with --offset,
        /// and advanced per trial by --offset-step so that no trial rereads
        /// bytes an earlier one read (Orion's servers cache repeatedly read
        /// data, and lfs ladvise dontneed does not evict it).
        base: u64,
    }

    impl Layout {
        fn new(file_bytes: u64, geom: Geom, ranks: u64) -> Result<Self, String> {
            let n_leaves = match &geom {
                Geom::Fixed(l) => file_bytes.div_ceil(*l),
                Geom::Grid { grid, dims, .. } => {
                    dims.iter().zip(grid).map(|(d, g)| d.div_ceil(*g)).product()
                }
            };
            if n_leaves == 0 {
                return Err("dataset has no leaves".into());
            }
            if !ranks.is_power_of_two() {
                return Err(format!(
                    "rank count {ranks} is not a power of two; aligned subtrees need one"
                ));
            }
            let padded = n_leaves.next_power_of_two();
            if ranks > padded {
                return Err(format!("{ranks} ranks but only {padded} padded leaves"));
            }
            let s = padded / ranks;
            Ok(Self {
                file_bytes,
                geom,
                n_leaves,
                padded,
                d: padded.trailing_zeros(),
                ranks,
                k: ranks.trailing_zeros(),
                s,
                sub: s.trailing_zeros(),
                base: 0,
            })
        }

        fn leaf_range(&self, i: u64) -> Range<u64> {
            match &self.geom {
                Geom::Fixed(l) => self.base + i * l..self.base + ((i + 1) * l).min(self.file_bytes),
                Geom::Grid { grid, dims, elem } => {
                    let r = verification_grid::leaf_byte_range(grid, dims, *elem, i);
                    self.base + r.start..self.base + r.end
                }
            }
        }

        fn leaf_bytes_max(&self) -> u64 {
            self.leaf_range(0).end
        }

        /// Real (non-padding) leaves owned by rank `r`; empty when the
        /// rank's block lies wholly in the padding.
        fn block(&self, r: u64) -> Range<u64> {
            let lo = (r * self.s).min(self.n_leaves);
            lo..((r + 1) * self.s).min(self.n_leaves)
        }

        fn desc(&self) -> String {
            match &self.geom {
                Geom::Fixed(l) => l.to_string(),
                Geom::Grid { grid, .. } => format!(
                    "grid:{}",
                    grid.iter().map(u64::to_string).collect::<Vec<_>>().join("x")
                ),
            }
        }

        fn companion_bytes(&self) -> u64 {
            (2 * self.padded - 1) * NODE
        }
    }

    /// Byte offset of the node at `(level, pos)` in the level-order array:
    /// level `l` holds `2^l` nodes starting at index `2^l - 1`.
    fn node_off(level: u32, pos: u64) -> u64 {
        ((1u64 << level) - 1 + pos) * NODE
    }

    // ===== streaming reads =====

    /// Group consecutive leaves into read batches of at most `budget` bytes
    /// (always at least one leaf, so an oversized leaf cannot stall).
    fn leaf_batches(lay: &Layout, leaves: Range<u64>, budget: u64) -> Vec<Range<u64>> {
        let mut out = Vec::new();
        let mut i = leaves.start;
        while i < leaves.end {
            let base = lay.leaf_range(i).start;
            let mut j = i + 1;
            while j < leaves.end && lay.leaf_range(j).end - base <= budget {
                j += 1;
            }
            out.push(i..j);
            i = j;
        }
        out
    }

    struct StreamStats {
        /// Time the consumer sat waiting for the next batch: the I/O the
        /// overlap failed to hide.
        io_wait: f64,
        /// Time spent in the consumer's work (hashing).
        work: f64,
        bytes: u64,
    }

    /// Read byte `spans` in order, one batch ahead of `consume`, the same
    /// producer/consumer overlap as `contiguous_tree_streaming`. Reads are
    /// widened to page alignment so the same code path serves `O_DIRECT`.
    fn stream_spans<F>(file: &File, spans: &[Range<u64>], mut consume: F) -> std::io::Result<StreamStats>
    where
        F: FnMut(usize, &[u8]),
    {
        let max_len = spans
            .iter()
            .map(|r| align_up(r.end) - align_down(r.start))
            .max()
            .unwrap_or(0) as usize;
        let mut st = StreamStats { io_wait: 0.0, work: 0.0, bytes: 0 };
        if spans.is_empty() {
            return Ok(st);
        }
        std::thread::scope(|scope| -> std::io::Result<()> {
            type Msg = std::io::Result<(AlignedBuf, usize, usize, usize)>;
            let (full_tx, full_rx) = std::sync::mpsc::sync_channel::<Msg>(1);
            let (empty_tx, empty_rx) = std::sync::mpsc::channel::<AlignedBuf>();
            let _ = empty_tx.send(AlignedBuf::new(max_len));
            let _ = empty_tx.send(AlignedBuf::new(max_len));
            scope.spawn(move || {
                for (idx, span) in spans.iter().enumerate() {
                    let mut buf = empty_rx.recv().unwrap_or_else(|_| AlignedBuf::new(max_len));
                    let al = align_down(span.start);
                    let len = (align_up(span.end) - al) as usize;
                    let res = pread_upto(file, &mut buf.as_mut()[..len], al).and_then(|got| {
                        if (got as u64) < span.end - al {
                            Err(std::io::Error::new(
                                std::io::ErrorKind::UnexpectedEof,
                                format!("short read at {al}: {got} of {} bytes", span.end - al),
                            ))
                        } else {
                            Ok(())
                        }
                    });
                    let msg = res.map(|()| {
                        let o = (span.start - al) as usize;
                        (buf, idx, o, (span.end - span.start) as usize)
                    });
                    let failed = msg.is_err();
                    if full_tx.send(msg).is_err() || failed {
                        return;
                    }
                }
            });
            loop {
                let t = Instant::now();
                let Ok(msg) = full_rx.recv() else { break };
                st.io_wait += t.elapsed().as_secs_f64();
                let (buf, idx, o, len) = msg?;
                let t = Instant::now();
                consume(idx, &buf.raw[buf.off + o..buf.off + o + len]);
                st.work += t.elapsed().as_secs_f64();
                st.bytes += len as u64;
                let _ = empty_tx.send(buf);
            }
            Ok(())
        })?;
        Ok(st)
    }

    /// Hash `leaves` (consecutive, owned by this rank) with leaf domain
    /// separation, parallel within each batch.
    fn hash_leaves(
        file: &File,
        lay: &Layout,
        leaves: Range<u64>,
        budget: u64,
        alg: HashAlg,
        hash: bool,
    ) -> std::io::Result<(Vec<H>, StreamStats)> {
        let batches = leaf_batches(lay, leaves.clone(), budget);
        let spans: Vec<Range<u64>> = batches
            .iter()
            .map(|b| lay.leaf_range(b.start).start..lay.leaf_range(b.end - 1).end)
            .collect();
        let mut out: Vec<H> = Vec::with_capacity((leaves.end - leaves.start) as usize);
        let st = stream_spans(file, &spans, |idx, bytes| {
            if !hash {
                return;
            }
            let b = &batches[idx];
            let base = spans[idx].start;
            let slices: Vec<&[u8]> = (b.start..b.end)
                .map(|i| {
                    let r = lay.leaf_range(i);
                    &bytes[(r.start - base) as usize..(r.end - base) as usize]
                })
                .collect();
            let mut part: Vec<H> = Vec::new();
            slices.par_iter().map(|s| alg.hash_leaf(s)).collect_into_vec(&mut part);
            out.extend_from_slice(&part);
        })?;
        Ok((out, st))
    }

    // ===== tree pieces =====

    /// `nulls[h]`: the root of an all-padding subtree of height `h`.
    fn null_table(alg: HashAlg, levels: u32) -> Vec<H> {
        let mut t = vec![alg.null_sentinel()];
        for h in 0..levels as usize {
            t.push(alg.hash_pair(&t[h], &t[h]));
        }
        t
    }

    fn hash_level(prev: &[H], alg: HashAlg) -> Vec<H> {
        if prev.len() >= 8192 {
            prev.par_chunks(2).map(|c| alg.hash_pair(&c[0], &c[1])).collect()
        } else {
            prev.chunks(2).map(|c| alg.hash_pair(&c[0], &c[1])).collect()
        }
    }

    /// Levels of a subtree over `width` (power of two) leaf slots, bottom up:
    /// `levels[0]` the leaves, `levels.last()` the single root. Missing
    /// trailing leaves are padded with the null sentinel, exactly as
    /// `MerkleTree::from_leaf_hashes` pads.
    fn build_levels(mut leaves: Vec<H>, width: u64, alg: HashAlg, nulls: &[H]) -> Vec<Vec<H>> {
        let height = width.trailing_zeros() as usize;
        if leaves.is_empty() {
            return (0..=height).map(|h| vec![nulls[h]; (width >> h) as usize]).collect();
        }
        leaves.resize(width as usize, nulls[0]);
        let mut levels = vec![leaves];
        while levels.last().map_or(0, Vec::len) > 1 {
            let next = hash_level(levels.last().unwrap(), alg);
            levels.push(next);
        }
        levels
    }

    fn flatten(nodes: &[H]) -> Vec<u8> {
        nodes.iter().flat_map(|h| h.iter().copied()).collect()
    }

    fn unflatten(bytes: &[u8]) -> Vec<H> {
        bytes
            .chunks_exact(32)
            .map(|c| {
                let mut h = [0u8; 32];
                h.copy_from_slice(c);
                h
            })
            .collect()
    }

    fn read_nodes(f: &File, off: u64, n: u64) -> std::io::Result<Vec<H>> {
        let mut b = vec![0u8; (n * NODE) as usize];
        f.read_exact_at(&mut b, off)?;
        Ok(unflatten(&b))
    }

    fn hex(h: &H) -> String {
        h.iter().map(|b| format!("{b:02x}")).collect()
    }

    // ===== cluster context =====

    struct Ctx {
        world: SimpleCommunicator,
        rank: u64,
        size: u64,
        nodes: u64,
        /// First rank on each node: the one that evicts that node's cache.
        node_leader: bool,
        threads: usize,
    }

    impl Ctx {
        fn barrier(&self) {
            self.world.barrier();
        }
        fn root(&self) -> bool {
            self.rank == 0
        }
        /// Gather one f64 per rank onto rank 0.
        fn gather_f64(&self, v: f64) -> Vec<f64> {
            let root = self.world.process_at_rank(0);
            if self.root() {
                let mut all = vec![0.0f64; self.size as usize];
                root.gather_into_root(&v, &mut all[..]);
                all
            } else {
                root.gather_into(&v);
                Vec::new()
            }
        }
        fn gather_vec_f64(&self, v: &[f64]) -> Vec<f64> {
            let root = self.world.process_at_rank(0);
            if self.root() {
                let mut all = vec![0.0f64; v.len() * self.size as usize];
                root.gather_into_root(v, &mut all[..]);
                all
            } else {
                root.gather_into(v);
                Vec::new()
            }
        }
        fn sum_u64(&self, v: u64) -> u64 {
            let mut s = 0u64;
            self.world.all_reduce_into(&v, &mut s, SystemOperation::sum());
            s
        }
        /// Drop this file's pages on every node before a cold trial. Spawning
        /// `lfs ladvise` from every rank would be its own load test, so one
        /// rank per node does it for the node.
        /// Returns whether every node's eviction calls reported success.
        /// On Lustre this includes `lfs ladvise -a dontneed`, which asks the
        /// *servers* to drop the file's pages -- the cache O_DIRECT does not
        /// bypass, and the one a repeated read of the same prefix otherwise
        /// measures instead of the disks.
        fn evict(&self, paths: &[&Path]) -> bool {
            let mut ok = true;
            if self.node_leader {
                for p in paths {
                    if p.exists() {
                        ok &= evict_cache(p);
                    }
                }
            }
            let mut all = 0u64;
            self.world.all_reduce_into(&u64::from(!ok), &mut all, SystemOperation::sum());
            all == 0
        }
    }

    fn max(v: &[f64]) -> f64 {
        v.iter().copied().fold(f64::NAN, f64::max)
    }
    fn mean(v: &[f64]) -> f64 {
        if v.is_empty() { f64::NAN } else { v.iter().sum::<f64>() / v.len() as f64 }
    }
    fn pct(v: &mut [f64], p: f64) -> f64 {
        let v: Vec<f64> = {
            let mut w: Vec<f64> = v.iter().copied().filter(|x| x.is_finite()).collect();
            w.sort_by(|a, b| a.partial_cmp(b).unwrap());
            w
        };
        if v.is_empty() {
            return f64::NAN;
        }
        let i = ((v.len() - 1) as f64 * p).round() as usize;
        v[i]
    }

    /// Small deterministic generator: synthetic data and selection choices
    /// must be reproducible across reruns and identical across hash runs.
    struct XorShift(u64);
    impl XorShift {
        fn new(seed: u64) -> Self {
            Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
        }
        fn next(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }
        fn below(&mut self, n: u64) -> u64 {
            if n == 0 { 0 } else { self.next() % n }
        }
        fn fill(&mut self, buf: &mut [u8]) {
            for c in buf.chunks_mut(8) {
                let v = self.next().to_le_bytes();
                c.copy_from_slice(&v[..c.len()]);
            }
        }
    }

    // ===== provenance =====

    struct Prov {
        storage_class: String,
        stripe_count: String,
        stripe_size: String,
        /// The companion's own stripe count: E2's hotspot variable.
        comp_stripe: String,
        o_direct: bool,
        /// Per trial: "cold" if server caches were dropped before it,
        /// "warm" if not (--warm-server, or the drop failed).
        server_cache: &'static str,
    }

    fn provenance(data: &Path, companion: &Path, o_direct: bool) -> Prov {
        let lustre = is_lustre(data);
        let comp_stripe = if !companion.as_os_str().is_empty() && is_lustre(companion) {
            lfs_field(companion, "-c").unwrap_or_default()
        } else {
            String::new()
        };
        let env = |k: &str| std::env::var(k).ok().filter(|s| !s.trim().is_empty());
        Prov {
            storage_class: env("BENCH_STORAGE_CLASS")
                .unwrap_or_else(|| if lustre { "lustre".into() } else { "local".into() }),
            stripe_count: if lustre { lfs_field(data, "-c").unwrap_or_default() } else { String::new() },
            stripe_size: if lustre { lfs_field(data, "-S").unwrap_or_default() } else { String::new() },
            comp_stripe,
            o_direct,
            server_cache: "cold",
        }
    }

    fn print_header_comments(ctx: &Ctx, args: &Args, lay: Option<&Layout>, prov: &Prov) {
        let host = std::fs::read_to_string("/proc/sys/kernel/hostname").unwrap_or_default();
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        println!("# pfs_merkle_bench argv={}", args.0[1..].join(" "));
        println!(
            "# host0={} unix_time={now} nodes={} ranks={} threads_per_rank={}",
            host.trim(),
            ctx.nodes,
            ctx.size,
            ctx.threads
        );
        println!(
            "# clawhdf5={} s2d2={}",
            std::env::var("BENCH_CLAWHDF5_SHA").unwrap_or_else(|_| "unknown".into()),
            std::env::var("BENCH_S2D2_SHA").unwrap_or_else(|_| "unknown".into())
        );
        println!("# server_cache_requested={}", if args.has("--warm-server") { "warm" } else { "cold" });
        println!(
            "# storage_class={} stripe_count={} stripe_size={} companion_stripe_count={} o_direct={}",
            prov.storage_class,
            prov.stripe_count,
            prov.stripe_size,
            prov.comp_stripe,
            u8::from(prov.o_direct)
        );
        if let Some(l) = lay {
            println!(
                "# n_leaves={} padded={} depth={} subtree_leaves_per_rank={} companion_bytes={}",
                l.n_leaves,
                l.padded,
                l.d,
                l.s,
                l.companion_bytes()
            );
        }
    }

    const PREFIX_COLS: &str = "mode,strategy,alg,nodes,ranks,threads,file_bytes,leaf,n_leaves,\
                               o_direct,storage_class,stripe_count,stripe_size,companion_stripe_count,server_cache,trial";

    fn prefix(ctx: &Ctx, mode: &str, strategy: &str, alg: &str, lay: &Layout, prov: &Prov, t: usize) -> String {
        format!(
            "{mode},{strategy},{alg},{},{},{},{},{},{},{},{},{},{},{},{},{t}",
            ctx.nodes,
            ctx.size,
            ctx.threads,
            lay.file_bytes,
            lay.desc(),
            lay.n_leaves,
            u8::from(prov.o_direct),
            prov.storage_class,
            prov.stripe_count,
            prov.stripe_size,
            prov.comp_stripe,
            prov.server_cache
        )
    }

    // ===== entry =====

    pub fn main() {
        let args = Args(std::env::args().collect());
        let universe = mpi::initialize().unwrap_or_else(|| die("MPI initialisation failed"));
        let world = universe.world();
        let rank = world.rank() as u64;
        let size = world.size() as u64;
        let local = world.split_shared(world.rank());
        let node_leader = local.rank() == 0;
        let mut nodes = 0u64;
        world.all_reduce_into(&u64::from(node_leader), &mut nodes, SystemOperation::sum());

        let threads = args.num("--threads", "0") as usize;
        if threads > 0 {
            let _ = rayon::ThreadPoolBuilder::new().num_threads(threads).build_global();
        }
        let ctx = Ctx {
            world,
            rank,
            size,
            nodes,
            node_leader,
            threads: rayon::current_num_threads(),
        };

        let mode = args.get("--mode", "");
        let data = PathBuf::from(args.get("--data", ""));
        if data.as_os_str().is_empty() {
            die("--data <file> is required");
        }
        let res = match mode.as_str() {
            "gen" => run_gen(&ctx, &args, &data),
            "read" | "flat" | "build" => run_build(&ctx, &args, &data, &mode),
            "verify" => run_verify(&ctx, &args, &data),
            "update" => run_update(&ctx, &args, &data),
            _ => die("--mode must be gen, read, flat, build, verify or update"),
        };
        if let Err(e) = res {
            eprintln!("rank {}: {e}", ctx.rank);
            ctx.world.abort(1);
        }
    }

    fn layout_from_args(ctx: &Ctx, args: &Args, data: &Path) -> std::io::Result<Layout> {
        let file_bytes = std::fs::metadata(data)?.len();
        let geom = if args.has("--dims") {
            let dims: Vec<u64> = args
                .get("--dims", "")
                .split(['x', ','])
                .map(|s| s.parse().unwrap_or_else(|_| die("--dims: expected e.g. 1000x1000x1000")))
                .collect();
            let elem = args.num("--elem", "4") as u32;
            let target = args.num("--target", "1M");
            let need: u64 = dims.iter().product::<u64>() * u64::from(elem);
            if need > file_bytes {
                die(&format!("--dims needs {need} bytes but the file has {file_bytes}"));
            }
            let grid = verification_grid::verification_grid(&dims, elem, target)
                .unwrap_or_else(|| die("dataset too small for a verification grid"));
            Geom::Grid { grid, dims, elem }
        } else {
            Geom::Fixed(args.num("--leaf-bytes", "1M"))
        };
        let file_bytes = match &geom {
            Geom::Grid { dims, elem, .. } => dims.iter().product::<u64>() * u64::from(*elem),
            // `--bytes` treats a prefix of the file as the dataset, so a weak
            // scaling series can share one large striped file instead of
            // generating one per node count.
            Geom::Fixed(_) if args.has("--bytes") => {
                let b = args.num("--bytes", "0");
                if b == 0 || b > file_bytes {
                    die(&format!("--bytes {b} must be in 1..={file_bytes}"));
                }
                b
            }
            Geom::Fixed(_) => file_bytes,
        };
        let mut lay = Layout::new(file_bytes, geom, ctx.size).map_err(std::io::Error::other)?;
        let (offset, step) = region(args);
        let (w, n) = trials(args);
        let last = offset + step * (w + n).saturating_sub(1) as u64;
        let have = std::fs::metadata(data)?.len();
        if last + lay.file_bytes > have {
            die(&format!(
                "--offset {offset} + {} trials x --offset-step {step} + {} bytes exceeds the file ({have})",
                w + n,
                lay.file_bytes
            ));
        }
        lay.base = offset;
        Ok(lay)
    }

    /// `--offset` (first trial's region) and `--offset-step` (advance per
    /// trial, warm-ups included). Both are byte counts and must keep O_DIRECT
    /// alignment, so they are rounded to whole MiB.
    fn region(args: &Args) -> (u64, u64) {
        let mib = 1u64 << 20;
        let o = args.num("--offset", "0");
        let st = args.num("--offset-step", "0");
        if o % mib != 0 || st % mib != 0 {
            die("--offset and --offset-step must be whole MiB");
        }
        (o, st)
    }

    fn trials(args: &Args) -> (usize, usize) {
        (args.num("--warmup", "1") as usize, args.num("--trials", "5") as usize)
    }

    // ===== gen =====

    fn run_gen(ctx: &Ctx, args: &Args, data: &Path) -> std::io::Result<()> {
        let total = args.num("--size", "4G");
        let block = 64u64 << 20;
        if ctx.root() {
            // No truncate: a file pre-created with `lfs setstripe` keeps its
            // layout, which is fixed at creation and is the point of the run.
            let f = OpenOptions::new().create(true).write(true).truncate(false).open(data)?;
            f.set_len(total)?;
        }
        ctx.barrier();
        let f = OpenOptions::new().write(true).open(data)?;
        let per = total.div_ceil(ctx.size).div_ceil(1 << 20) * (1 << 20);
        let lo = (ctx.rank * per).min(total);
        let hi = ((ctx.rank + 1) * per).min(total);
        let mut buf = vec![0u8; block as usize];
        let t = Instant::now();
        let mut off = lo;
        while off < hi {
            let n = (hi - off).min(block) as usize;
            XorShift::new(off ^ 0x5eed).fill(&mut buf[..n]);
            f.write_all_at(&buf[..n], off)?;
            off += n as u64;
        }
        f.sync_all()?;
        let secs = t.elapsed().as_secs_f64();
        let all = ctx.gather_f64(secs);
        ctx.barrier();
        if ctx.root() {
            let wall = max(&all);
            eprintln!(
                "gen: {} bytes by {} ranks in {:.1}s ({:.0} MB/s aggregate)",
                total,
                ctx.size,
                wall,
                total as f64 / wall / 1e6
            );
        }
        Ok(())
    }

    // ===== E1: read / flat / build =====

    fn run_build(ctx: &Ctx, args: &Args, data: &Path, mode: &str) -> std::io::Result<()> {
        let o_direct = args.has("--o-direct");
        let alg = parse_alg(&args.get("--alg", "blake3"));
        let budget = args.num("--batch-bytes", "64M");
        let strategy = if mode == "build" { args.get("--strategy", "distributed") } else { "-".into() };
        if !matches!(strategy.as_str(), "none" | "gather" | "distributed" | "-") {
            die("--strategy for build must be none, gather or distributed");
        }
        let companion = PathBuf::from(args.get("--companion", ""));
        if matches!(strategy.as_str(), "gather" | "distributed") && companion.as_os_str().is_empty() {
            die("--companion <file> is required for --strategy gather|distributed");
        }
        if mode == "flat" && alg == HashAlg::K12 {
            die("flat mode supports sha256 and blake3");
        }
        let mut lay = layout_from_args(ctx, args, data)?;
        let (offset0, step) = region(args);
        let mut prov = if ctx.root() { provenance(data, &companion, o_direct) } else { provenance_stub(o_direct) };
        let warm_server = args.has("--warm-server");
        if ctx.root() {
            print_header_comments(ctx, args, Some(&lay), &prov);
            println!(
                "{PREFIX_COLS},wall_ms,throughput_mbps,leaf_max_ms,leaf_mean_ms,io_wait_max_ms,\
                 hash_max_ms,subtree_max_ms,reduce_ms,companion_ms,companion_writes,companion_bytes"
            );
        }
        // Rank 0 holds the gathered subtrees and the assembled array at once.
        // A Frontier node has 512 GB; 64 GiB leaves the node's other ranks
        // their share. Beyond that the naive coordinator is not a baseline
        // worth running -- which is itself the point of the comparison.
        if strategy == "gather" && lay.companion_bytes() * 2 > (64 << 30) {
            die("gather strategy would hold >64 GiB on rank 0; use distributed");
        }
        if ctx.root() && matches!(strategy.as_str(), "gather" | "distributed") {
            let f = OpenOptions::new().create(true).write(true).truncate(false).open(&companion)?;
            f.set_len(lay.companion_bytes())?;
        }
        ctx.barrier();

        let nulls = null_table(alg, lay.d);
        let (warm, n) = trials(args);
        let mut last_root: Option<H> = None;
        for t in 0..warm + n {
            // Client caches matter only for buffered reads, but server caches
            // matter for O_DIRECT too: drop both unless --warm-server.
            prov.server_cache = if !warm_server || !o_direct {
                if ctx.evict(&[data, &companion]) && !warm_server { "cold" } else { "warm" }
            } else {
                "warm"
            };
            // Fresh-region mode: each trial reads bytes no earlier trial read.
            lay.base = offset0 + t as u64 * step;
            if step > 0 {
                prov.server_cache = "fresh";
            }
            let file = open_maybe_direct(data, o_direct)?;
            let comp = if matches!(strategy.as_str(), "gather" | "distributed") {
                Some(OpenOptions::new().write(true).open(&companion)?)
            } else {
                None
            };
            ctx.barrier();
            let t0 = Instant::now();

            // --- phase A: read + hash this rank's leaves (flat: rank 0 streams everything)
            let (leaves, st) = if mode == "flat" {
                if ctx.root() {
                    let spans: Vec<Range<u64>> = (0..lay.file_bytes.div_ceil(budget))
                        .map(|i| lay.base + i * budget..lay.base + ((i + 1) * budget).min(lay.file_bytes))
                        .collect();
                    let mut sha = <sha2::Sha256 as sha2::Digest>::new();
                    let mut b3 = blake3::Hasher::new();
                    let st = stream_spans(&file, &spans, |_, bytes| match alg {
                        HashAlg::Sha256 => sha2::Digest::update(&mut sha, bytes),
                        _ => {
                            b3.update(bytes);
                        }
                    })?;
                    (Vec::new(), st)
                } else {
                    (Vec::new(), StreamStats { io_wait: 0.0, work: 0.0, bytes: 0 })
                }
            } else {
                hash_leaves(&file, &lay, lay.block(ctx.rank), budget, alg, mode == "build")?
            };
            let ta = t0.elapsed().as_secs_f64();
            ctx.barrier();
            let tb0 = t0.elapsed().as_secs_f64();

            // --- phase B: local subtree
            let mut sub_ms = 0.0;
            let mut levels: Vec<Vec<H>> = Vec::new();
            if mode == "build" {
                let tsb = Instant::now();
                levels = build_levels(leaves, lay.s, alg, &nulls);
                sub_ms = tsb.elapsed().as_secs_f64() * 1e3;
            }
            ctx.barrier();
            let tc0 = t0.elapsed().as_secs_f64();

            // --- phase C: reduce the P subtree roots on rank 0
            let mut top: Vec<Vec<H>> = Vec::new();
            let mut gathered: Vec<u8> = Vec::new();
            if mode == "build" {
                let root_proc = ctx.world.process_at_rank(0);
                let mine = *levels.last().unwrap().first().unwrap();
                if strategy == "gather" {
                    // Naive coordinator: every node of every subtree goes to
                    // rank 0, which writes the companion alone -- the shape of
                    // today's serial writer.
                    let flat: Vec<u8> = levels.iter().flat_map(|l| flatten(l)).collect();
                    if ctx.root() {
                        gathered = vec![0u8; flat.len() * ctx.size as usize];
                        root_proc.gather_into_root(&flat[..], &mut gathered[..]);
                    } else {
                        root_proc.gather_into(&flat[..]);
                    }
                }
                if ctx.root() {
                    let mut roots = vec![0u8; 32 * ctx.size as usize];
                    root_proc.gather_into_root(&mine[..], &mut roots[..]);
                    top = build_levels(unflatten(&roots), lay.ranks, alg, &nulls);
                    last_root = Some(top.last().unwrap()[0]);
                } else {
                    root_proc.gather_into(&mine[..]);
                }
            }
            let reduce_ms = (t0.elapsed().as_secs_f64() - tc0) * 1e3;
            ctx.barrier();
            let td0 = t0.elapsed().as_secs_f64();

            // --- phase D: companion write
            let mut writes = 0u64;
            let mut wbytes = 0u64;
            if let Some(c) = comp.as_ref() {
                if strategy == "distributed" {
                    for (h, lvl) in levels.iter().enumerate() {
                        let level = lay.d - h as u32;
                        let pos0 = ctx.rank * (lay.s >> h);
                        let b = flatten(lvl);
                        c.write_all_at(&b, node_off(level, pos0))?;
                        writes += 1;
                        wbytes += b.len() as u64;
                    }
                    if ctx.root() && lay.k > 0 {
                        // Levels 0..k-1 are contiguous at the head of the file.
                        let mut b = Vec::with_capacity(((1usize << lay.k) - 1) * 32);
                        for h in (1..=lay.k as usize).rev() {
                            b.extend(flatten(&top[h]));
                        }
                        c.write_all_at(&b, 0)?;
                        writes += 1;
                        wbytes += b.len() as u64;
                    }
                } else if ctx.root() {
                    let per = (2 * lay.s - 1) as usize * 32;
                    let mut all = vec![0u8; lay.companion_bytes() as usize];
                    for r in 0..ctx.size as usize {
                        let src = &gathered[r * per..(r + 1) * per];
                        let mut o = 0usize;
                        for h in 0..=lay.sub as usize {
                            let w = (lay.s >> h) as usize * 32;
                            let dst = node_off(lay.d - h as u32, r as u64 * (lay.s >> h)) as usize;
                            all[dst..dst + w].copy_from_slice(&src[o..o + w]);
                            o += w;
                        }
                    }
                    for h in 1..=lay.k as usize {
                        let b = flatten(&top[h]);
                        let dst = node_off(lay.k - h as u32, 0) as usize;
                        all[dst..dst + b.len()].copy_from_slice(&b);
                    }
                    c.write_all_at(&all, 0)?;
                    writes += 1;
                    wbytes += all.len() as u64;
                }
                if args.has("--fsync") {
                    c.sync_data()?;
                }
            }
            ctx.barrier();
            let wall = t0.elapsed().as_secs_f64();
            let comp_ms = (wall - td0) * 1e3;

            let leaf_all = ctx.gather_f64(ta * 1e3);
            let wait_all = ctx.gather_f64(st.io_wait * 1e3);
            let hash_all = ctx.gather_f64(st.work * 1e3);
            let sub_all = ctx.gather_f64(sub_ms);
            let writes = ctx.sum_u64(writes);
            let wbytes = ctx.sum_u64(wbytes);
            let bytes = ctx.sum_u64(st.bytes);
            let _ = tb0;
            if ctx.root() && t >= warm {
                println!(
                    "{},{:.1},{:.1},{:.1},{:.1},{:.1},{:.1},{:.2},{:.2},{:.1},{writes},{wbytes}",
                    prefix(ctx, mode, &strategy, alg_name(alg), &lay, &prov, t - warm),
                    wall * 1e3,
                    bytes as f64 / wall / 1e6,
                    max(&leaf_all),
                    mean(&leaf_all),
                    max(&wait_all),
                    max(&hash_all),
                    max(&sub_all),
                    reduce_ms,
                    comp_ms
                );
            }
        }
        if ctx.root()
            && let Some(r) = last_root {
                eprintln!("root={}", hex(&r));
            }
        if args.has("--check") && mode == "build" {
            let comp = (strategy != "none").then_some(companion.as_path());
            check_against_serial(ctx, &lay, data, comp, alg, last_root)?;
        }
        Ok(())
    }

    fn provenance_stub(o_direct: bool) -> Prov {
        Prov {
            storage_class: String::new(),
            stripe_count: String::new(),
            stripe_size: String::new(),
            comp_stripe: String::new(),
            o_direct,
            server_cache: "cold",
        }
    }

    /// Rebuild serially on rank 0 with the library constructor and compare
    /// root and companion bytes: the parallel path is only a result if it
    /// produces the tree the serial one does.
    fn check_against_serial(
        ctx: &Ctx,
        lay: &Layout,
        data: &Path,
        companion: Option<&Path>,
        alg: HashAlg,
        root: Option<H>,
    ) -> std::io::Result<()> {
        if ctx.root() {
            if lay.file_bytes > CHECK_LIMIT {
                eprintln!("check: skipped, file exceeds {CHECK_LIMIT} bytes");
            } else {
                let f = File::open(data)?;
                let mut leaves = Vec::with_capacity(lay.n_leaves as usize);
                for i in 0..lay.n_leaves {
                    let r = lay.leaf_range(i);
                    let mut b = vec![0u8; (r.end - r.start) as usize];
                    f.read_exact_at(&mut b, r.start)?;
                    leaves.push(alg.hash_leaf(&b));
                }
                let tree = MerkleTree::from_leaf_hashes(&leaves, alg);
                let mut ok = true;
                if let Some(r) = root
                    && &r != tree.root() {
                        eprintln!("check: ROOT MISMATCH parallel={} serial={}", hex(&r), hex(tree.root()));
                        ok = false;
                    }
                if let Some(c) = companion {
                    let got = std::fs::read(c)?;
                    let want = flatten(tree.nodes());
                    if got != want {
                        let first = got.iter().zip(&want).position(|(a, b)| a != b);
                        eprintln!(
                            "check: COMPANION MISMATCH ({} vs {} bytes, first diff at {:?})",
                            got.len(),
                            want.len(),
                            first
                        );
                        ok = false;
                    }
                }
                eprintln!("check: {}", if ok { "PASS" } else { "FAIL" });
                if !ok {
                    ctx.world.abort(3);
                }
            }
        }
        ctx.barrier();
        Ok(())
    }

    // ===== E2: concurrent partial verification =====

    fn run_verify(ctx: &Ctx, args: &Args, data: &Path) -> std::io::Result<()> {
        let o_direct = args.has("--o-direct");
        let alg = parse_alg(&args.get("--alg", "blake3"));
        let companion = PathBuf::from(args.get("--companion", ""));
        if companion.as_os_str().is_empty() {
            die("--companion <file> is required for verify");
        }
        let k = args.num("--k", "16").max(1);
        let ops = args.num("--ops", "64") as usize;
        let select = args.get("--select", "random");
        if !matches!(select.as_str(), "random" | "local") {
            die("--select must be random or local");
        }
        let cache_top = args.num("--cache-top", "0") as u32;
        let lay = layout_from_args(ctx, args, data)?;
        let (offset0, step) = region(args);
        let _ = offset0;
        if step > 0 {
            die("--offset-step applies to read/flat/build only: verify and update need the companion of one fixed region");
        }
        if k > lay.n_leaves {
            die("--k exceeds the leaf count");
        }
        let cache_top = cache_top.min(lay.d + 1);
        let mut prov = if ctx.root() { provenance(data, &companion, o_direct) } else { provenance_stub(o_direct) };
        let warm_server = args.has("--warm-server");
        if ctx.root() {
            print_header_comments(ctx, args, Some(&lay), &prov);
            println!(
                "{PREFIX_COLS},wall_ms,ops,ops_per_s,verified_mbps,lat_p50_ms,lat_p95_ms,lat_p99_ms,\
                 data_read_mean_ms,hash_mean_ms,proof_mean_ms,proof_reads_per_op,cache_top,k_leaves,\
                 select,failures"
            );
        }
        let strategy = format!("{select}-k{k}-top{cache_top}");
        let (warm, n) = trials(args);
        for t in 0..warm + n {
            // Client caches matter only for buffered reads, but server caches
            // matter for O_DIRECT too: drop both unless --warm-server.
            prov.server_cache = if !warm_server || !o_direct {
                if ctx.evict(&[data, &companion]) && !warm_server { "cold" } else { "warm" }
            } else {
                "warm"
            };
            let file = open_maybe_direct(data, o_direct)?;
            let comp = File::open(&companion)?;
            let mut root = [0u8; 32];
            if ctx.root() {
                comp.read_exact_at(&mut root, 0)?;
            }
            ctx.world.process_at_rank(0).broadcast_into(&mut root[..]);
            let max_span = (k * lay.leaf_bytes_max()) as usize + 2 * ALIGN;
            let mut buf = AlignedBuf::new(max_span);
            let mut rng = XorShift::new((t as u64) << 32 | ctx.rank);

            ctx.barrier();
            let t0 = Instant::now();
            // Levels above `cache_top` are fetched once per rank, as a client
            // that keeps the top of the tree resident would. They sit
            // contiguously at the head of the companion.
            let top: Vec<H> = if cache_top > 0 {
                read_nodes(&comp, 0, (1u64 << cache_top) - 1)?
            } else {
                Vec::new()
            };
            let fetch = |level: u32, pos: u64, reads: &mut u64| -> std::io::Result<H> {
                if level < cache_top {
                    Ok(top[((1u64 << level) - 1 + pos) as usize])
                } else {
                    *reads += 1;
                    let mut h = [0u8; 32];
                    comp.read_exact_at(&mut h, node_off(level, pos))?;
                    Ok(h)
                }
            };

            let block = lay.block(ctx.rank);
            let mut lat = Vec::with_capacity(ops);
            let (mut rd, mut hs, mut pf) = (0.0f64, 0.0f64, 0.0f64);
            let mut reads = 0u64;
            let mut done = 0u64;
            let mut useful = 0u64;
            let mut failures = 0u64;
            for _ in 0..ops {
                let a = match select.as_str() {
                    "local" => {
                        let len = block.end - block.start;
                        if len < k {
                            lat.push(f64::NAN);
                            continue;
                        }
                        block.start + rng.below(len - k + 1)
                    }
                    _ => rng.below(lay.n_leaves - k + 1),
                };
                let s0 = Instant::now();
                let span = lay.leaf_range(a).start..lay.leaf_range(a + k - 1).end;
                let al = align_down(span.start);
                let len = (align_up(span.end) - al) as usize;
                let got = pread_upto(&file, &mut buf.as_mut()[..len], al)?;
                if (got as u64) < span.end - al {
                    return Err(std::io::Error::new(std::io::ErrorKind::UnexpectedEof, "short data read"));
                }
                let s1 = Instant::now();
                let bytes = &buf.raw[buf.off..buf.off + len];
                let slices: Vec<&[u8]> = (a..a + k)
                    .map(|i| {
                        let r = lay.leaf_range(i);
                        &bytes[(r.start - al) as usize..(r.end - al) as usize]
                    })
                    .collect();
                let mut cur: Vec<H> = if k >= 8 {
                    slices.par_iter().map(|s| alg.hash_leaf(s)).collect()
                } else {
                    slices.iter().map(|s| alg.hash_leaf(s)).collect()
                };
                let s2 = Instant::now();
                // Walk up with the pruned witness set: a contiguous range
                // needs at most one sibling on each side per level.
                let (mut lo, mut hi) = (a, a + k);
                let mut level = lay.d;
                while level > 0 {
                    if lo % 2 == 1 {
                        cur.insert(0, fetch(level, lo - 1, &mut reads)?);
                        lo -= 1;
                    }
                    if hi % 2 == 1 {
                        cur.push(fetch(level, hi, &mut reads)?);
                        hi += 1;
                    }
                    cur = hash_level(&cur, alg);
                    lo /= 2;
                    hi /= 2;
                    level -= 1;
                }
                if cur[0] != root {
                    failures += 1;
                }
                let s3 = Instant::now();
                rd += (s1 - s0).as_secs_f64();
                hs += (s2 - s1).as_secs_f64();
                pf += (s3 - s2).as_secs_f64();
                lat.push((s3 - s0).as_secs_f64() * 1e3);
                done += 1;
                useful += span.end - span.start;
            }
            ctx.barrier();
            let wall = t0.elapsed().as_secs_f64();

            let mut all_lat = ctx.gather_vec_f64(&lat);
            let d = done.max(1) as f64;
            let rd_all = ctx.gather_f64(if done > 0 { rd / d * 1e3 } else { f64::NAN });
            let hs_all = ctx.gather_f64(if done > 0 { hs / d * 1e3 } else { f64::NAN });
            let pf_all = ctx.gather_f64(if done > 0 { pf / d * 1e3 } else { f64::NAN });
            let done = ctx.sum_u64(done);
            let useful = ctx.sum_u64(useful);
            let reads = ctx.sum_u64(reads);
            let failures = ctx.sum_u64(failures);
            let fin = |v: &[f64]| mean(&v.iter().copied().filter(|x| x.is_finite()).collect::<Vec<_>>());
            if ctx.root() && t >= warm {
                println!(
                    "{},{:.1},{done},{:.1},{:.1},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.2},{cache_top},{k},{select},{failures}",
                    prefix(ctx, "verify", &strategy, alg_name(alg), &lay, &prov, t - warm),
                    wall * 1e3,
                    done as f64 / wall,
                    useful as f64 / wall / 1e6,
                    pct(&mut all_lat, 0.50),
                    pct(&mut all_lat, 0.95),
                    pct(&mut all_lat, 0.99),
                    fin(&rd_all),
                    fin(&hs_all),
                    fin(&pf_all),
                    reads as f64 / done.max(1) as f64
                );
            }
            if failures > 0 && ctx.root() {
                eprintln!("verify: {failures} selections FAILED against the root");
            }
        }
        Ok(())
    }

    // ===== E3: parallel writers =====

    fn run_update(ctx: &Ctx, args: &Args, data: &Path) -> std::io::Result<()> {
        let o_direct = args.has("--o-direct");
        let alg = parse_alg(&args.get("--alg", "blake3"));
        let companion = PathBuf::from(args.get("--companion", ""));
        if companion.as_os_str().is_empty() {
            die("--companion <file> is required for update (build it first)");
        }
        let strategy = args.get("--strategy", "lazy");
        if !matches!(strategy.as_str(), "eager" | "lazy") {
            die("--strategy for update must be eager or lazy");
        }
        let updates = args.num("--updates", "64") as usize;
        let fsync = args.has("--fsync");
        let lay = layout_from_args(ctx, args, data)?;
        let (offset0, step) = region(args);
        let _ = offset0;
        if step > 0 {
            die("--offset-step applies to read/flat/build only: verify and update need the companion of one fixed region");
        }
        if o_direct {
            let aligned = matches!(lay.geom, Geom::Fixed(l) if l % ALIGN as u64 == 0)
                && lay.file_bytes % ALIGN as u64 == 0;
            if !aligned {
                die("--o-direct writes need page-aligned leaves (--leaf-bytes multiple of 4096)");
            }
        }
        if std::fs::metadata(&companion)?.len() != lay.companion_bytes() {
            die("companion size does not match this layout; rebuild it with --mode build");
        }
        let mut prov = if ctx.root() { provenance(data, &companion, o_direct) } else { provenance_stub(o_direct) };
        let warm_server = args.has("--warm-server");
        if ctx.root() {
            print_header_comments(ctx, args, Some(&lay), &prov);
            println!(
                "{PREFIX_COLS},wall_ms,updates,updates_per_s,upd_mean_ms,upd_p95_ms,coord_tail_ms,\
                 data_bytes,companion_writes,companion_bytes,wa_ops_per_update,wa_bytes_ratio,fsync"
            );
        }
        let mut dopts = OpenOptions::new();
        dopts.read(true).write(true);
        if o_direct {
            dopts.custom_flags(libc::O_DIRECT);
        }

        let (warm, n) = trials(args);
        for t in 0..warm + n {
            // Client caches matter only for buffered reads, but server caches
            // matter for O_DIRECT too: drop both unless --warm-server.
            prov.server_cache = if !warm_server || !o_direct {
                if ctx.evict(&[data, &companion]) && !warm_server { "cold" } else { "warm" }
            } else {
                "warm"
            };
            let dfile = dopts.open(data)?;
            let comp = OpenOptions::new().read(true).write(true).open(&companion)?;

            // Load this rank's subtree slices, and on rank 0 the shared top
            // (levels 0..=k, contiguous at the head of the file). Untimed:
            // a long-running writer holds these already.
            let mut levels: Vec<Vec<H>> = (0..=lay.sub as usize)
                .map(|h| read_nodes(&comp, node_off(lay.d - h as u32, ctx.rank * (lay.s >> h)), lay.s >> h))
                .collect::<std::io::Result<_>>()?;
            let mut top: Vec<Vec<H>> = Vec::new();
            if ctx.root() {
                let head = read_nodes(&comp, 0, (1u64 << (lay.k + 1)) - 1)?;
                top = (0..=lay.k as usize)
                    .map(|h| {
                        let level = lay.k as usize - h;
                        head[(1usize << level) - 1..(1usize << (level + 1)) - 1].to_vec()
                    })
                    .collect();
            }

            let block = lay.block(ctx.rank);
            let my_updates = if block.is_empty() { 0 } else { updates };
            let mut counts = vec![0u64; ctx.size as usize];
            ctx.world.all_gather_into(&(my_updates as u64), &mut counts[..]);
            let expected: u64 = counts[1..].iter().sum();

            let mut buf = AlignedBuf::new(lay.leaf_bytes_max() as usize + ALIGN);
            let mut rng = XorShift::new(0xD1CE ^ ((t as u64) << 32) ^ ctx.rank);
            let mut writes = 0u64;
            let mut wbytes = 0u64;
            let mut dbytes = 0u64;
            let mut lat = Vec::with_capacity(my_updates);
            let mut received = 0u64;

            // Rank 0's half of eager: fold a worker's new subtree root into
            // the shared top and write that path (k nodes).
            let apply_top = |top: &mut Vec<Vec<H>>, msg: &[u8; 40], writes: &mut u64, wbytes: &mut u64| -> std::io::Result<()> {
                let r = u64::from_le_bytes(msg[..8].try_into().unwrap());
                let mut h = [0u8; 32];
                h.copy_from_slice(&msg[8..]);
                top[0][r as usize] = h;
                let mut pos = r;
                for lv in 1..=lay.k as usize {
                    pos /= 2;
                    let p = pos as usize;
                    top[lv][p] = alg.hash_pair(&top[lv - 1][2 * p], &top[lv - 1][2 * p + 1]);
                    comp.write_all_at(&top[lv][p], node_off(lay.k - lv as u32, pos))?;
                    *writes += 1;
                    *wbytes += NODE;
                }
                Ok(())
            };

            ctx.barrier();
            let t0 = Instant::now();
            for u in 0..my_updates {
                let s0 = Instant::now();
                let leaf = block.start + rng.below(block.end - block.start);
                let r = lay.leaf_range(leaf);
                let len = (r.end - r.start) as usize;
                // New content for the chunk: the writer has it in memory, so
                // no read precedes the write.
                XorShift::new(leaf ^ ((t as u64) << 40) ^ ((u as u64) << 20)).fill(&mut buf.as_mut()[..len]);
                dfile.write_all_at(&buf.as_mut()[..len], r.start)?;
                dbytes += len as u64;
                let lh = alg.hash_leaf(&buf.raw[buf.off..buf.off + len]);
                let mut idx = (leaf - ctx.rank * lay.s) as usize;
                levels[0][idx] = lh;
                if strategy == "eager" {
                    comp.write_all_at(&lh, node_off(lay.d, leaf))?;
                    writes += 1;
                    wbytes += NODE;
                }
                for h in 1..=lay.sub as usize {
                    idx /= 2;
                    levels[h][idx] = alg.hash_pair(&levels[h - 1][2 * idx], &levels[h - 1][2 * idx + 1]);
                    if strategy == "eager" {
                        let pos = ctx.rank * (lay.s >> h) + idx as u64;
                        comp.write_all_at(&levels[h][idx], node_off(lay.d - h as u32, pos))?;
                        writes += 1;
                        wbytes += NODE;
                    }
                }
                if strategy == "eager" {
                    if fsync {
                        dfile.sync_data()?;
                        comp.sync_data()?;
                    }
                    let mut msg = [0u8; 40];
                    msg[..8].copy_from_slice(&ctx.rank.to_le_bytes());
                    msg[8..].copy_from_slice(&levels[lay.sub as usize][0]);
                    if ctx.root() {
                        apply_top(&mut top, &msg, &mut writes, &mut wbytes)?;
                        // Drain whatever workers have sent so far, so the
                        // coordinator's own updates do not starve them.
                        while let Some((m, _)) = ctx.world.any_process().immediate_matched_probe_with_tag(TAG_EAGER) {
                            let mut b = [0u8; 40];
                            m.matched_receive_into(&mut b[..]);
                            apply_top(&mut top, &b, &mut writes, &mut wbytes)?;
                            received += 1;
                        }
                    } else {
                        ctx.world.process_at_rank(0).send_with_tag(&msg[..], TAG_EAGER);
                    }
                }
                lat.push(s0.elapsed().as_secs_f64() * 1e3);
            }
            let own_done = t0.elapsed().as_secs_f64();
            let mut tail_ms = 0.0;
            if strategy == "eager" && ctx.root() {
                let ts = Instant::now();
                while received < expected {
                    let mut b = [0u8; 40];
                    ctx.world.any_process().receive_into_with_tag(&mut b[..], TAG_EAGER);
                    apply_top(&mut top, &b, &mut writes, &mut wbytes)?;
                    received += 1;
                }
                tail_ms = ts.elapsed().as_secs_f64() * 1e3;
            }
            if strategy == "lazy" {
                // One pass per epoch: every slice of this rank's subtree, then
                // the P roots to rank 0, which rewrites the shared head.
                for (h, lvl) in levels.iter().enumerate() {
                    let b = flatten(lvl);
                    comp.write_all_at(&b, node_off(lay.d - h as u32, ctx.rank * (lay.s >> h)))?;
                    writes += 1;
                    wbytes += b.len() as u64;
                }
                let mine = levels[lay.sub as usize][0];
                let root_proc = ctx.world.process_at_rank(0);
                if ctx.root() {
                    let mut roots = vec![0u8; 32 * ctx.size as usize];
                    root_proc.gather_into_root(&mine[..], &mut roots[..]);
                    let nulls = null_table(alg, lay.k);
                    let tl = build_levels(unflatten(&roots), lay.ranks, alg, &nulls);
                    if lay.k > 0 {
                        let mut b = Vec::new();
                        for h in (1..=lay.k as usize).rev() {
                            b.extend(flatten(&tl[h]));
                        }
                        comp.write_all_at(&b, 0)?;
                        writes += 1;
                        wbytes += b.len() as u64;
                    }
                } else {
                    root_proc.gather_into(&mine[..]);
                }
                if fsync {
                    dfile.sync_data()?;
                    comp.sync_data()?;
                }
            } else if fsync && ctx.root() {
                comp.sync_data()?;
            }
            ctx.barrier();
            let wall = t0.elapsed().as_secs_f64();
            let _ = own_done;

            let mut all_lat = ctx.gather_vec_f64(&{
                let mut l = lat.clone();
                l.resize(updates, f64::NAN);
                l
            });
            let total = ctx.sum_u64(my_updates as u64);
            let writes = ctx.sum_u64(writes);
            let wbytes = ctx.sum_u64(wbytes);
            let dbytes = ctx.sum_u64(dbytes);
            if ctx.root() && t >= warm {
                let finite: Vec<f64> = all_lat.iter().copied().filter(|x| x.is_finite()).collect();
                println!(
                    "{},{:.1},{total},{:.1},{:.3},{:.3},{:.2},{dbytes},{writes},{wbytes},{:.3},{:.5},{}",
                    prefix(ctx, "update", &strategy, alg_name(alg), &lay, &prov, t - warm),
                    wall * 1e3,
                    total as f64 / wall,
                    mean(&finite),
                    pct(&mut all_lat, 0.95),
                    tail_ms,
                    writes as f64 / total.max(1) as f64,
                    wbytes as f64 / dbytes.max(1) as f64,
                    u8::from(fsync)
                );
            }
        }
        if args.has("--check") {
            let root = if ctx.root() {
                let c = File::open(&companion)?;
                let mut r = [0u8; 32];
                c.read_exact_at(&mut r, 0)?;
                Some(r)
            } else {
                None
            };
            check_against_serial(ctx, &lay, data, Some(&companion), alg, root)?;
        }
        Ok(())
    }
}
