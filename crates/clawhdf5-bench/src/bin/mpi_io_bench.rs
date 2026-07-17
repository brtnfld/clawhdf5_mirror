//! h5bench-equivalent MPI-IO performance benchmark.
//!
//! Usage: mpirun -np N cargo run -p clawhdf5-bench --features mpi-io --bin mpi_io_bench -- --size <N>
//!
//! Measures collective write and read throughput in MB/s for f64 arrays.

#[cfg(feature = "mpi-io")]
fn main() {
    use clawhdf5_io::mpi_vol::MpiVol;
    use clawhdf5_io::vol::VirtualObjectLayer;
    use mpi::traits::*;
    use std::time::Instant;

    let args: Vec<String> = std::env::args().collect();
    let n_elements: usize = args
        .iter()
        .position(|a| a == "--size")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(100_000);

    let mut vol = MpiVol::new_world().expect("MPI init failed");
    let world = vol.universe.world();
    let rank = world.rank() as usize;
    let size = world.size() as usize;

    let path = format!("/tmp/clawhdf5_mpiio_bench_{n_elements}.h5");
    vol.open(&path).unwrap();

    // Each rank contributes n_elements/size f64 values
    let per_rank = n_elements / size;
    let shard: Vec<f64> = (0..per_rank)
        .map(|i| (rank * per_rank + i) as f64)
        .collect();
    let shard_bytes: Vec<u8> = shard.iter().flat_map(|v| v.to_le_bytes()).collect();

    // Collective write
    world.barrier();
    let t0 = Instant::now();
    vol.write_dataset("data", &shard_bytes, &[n_elements as u64], "f64")
        .unwrap();
    world.barrier();
    let write_elapsed = t0.elapsed().as_secs_f64();

    // Collective read
    let t1 = Instant::now();
    let _data = vol.read_dataset("data").unwrap();
    world.barrier();
    let read_elapsed = t1.elapsed().as_secs_f64();

    if rank == 0 {
        let total_mb = (n_elements * 8) as f64 / 1e6;
        println!("=== clawhdf5 MPI-IO Benchmark ===");
        println!("Elements : {n_elements}");
        println!("Ranks    : {size}");
        println!("Total    : {total_mb:.1} MB");
        println!("Write    : {:.1} MB/s", total_mb / write_elapsed);
        println!("Read     : {:.1} MB/s", total_mb / read_elapsed);
    }
}

#[cfg(not(feature = "mpi-io"))]
fn main() {
    eprintln!("mpi_io_bench requires the `mpi-io` feature.");
    eprintln!("Run: mpirun -np N cargo run -p clawhdf5-bench --features mpi-io --bin mpi_io_bench");
    std::process::exit(1);
}
