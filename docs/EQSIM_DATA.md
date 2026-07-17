# Obtaining EQSIM Test Data

This guide explains how to obtain EQSIM earthquake-simulation data for the
P2.4 attack harness (`crates/clawhdf5-attack-harness`, `--eqsim-file`).

## What EQSIM is

EQSIM is a DOE/Lawrence Berkeley National Laboratory earthquake simulation
framework (McCallen et al.); its output populates the **Simulated Ground
Motion Database (SGMD)**, representative of large-scale HPC checkpoint/
restart-style write patterns (S2-D2-Yr2 §7.4, "Datasets").

- Portal: <https://sgmd.peer.berkeley.edu>
- DOE announcement: <https://www.energy.gov/ceser/articles/doe-lbnl-release-earthquake-database-strengthen-disaster-plans>
- LBNL announcement: <https://cs.lbl.gov/news-and-events/news/2023/the-most-advanced-bay-area-earthquake-simulations-will-be-publicly-available/>
- Paper: McCallen et al., *"An open-access simulated earthquake ground-motion
  database for an M7 Hayward Fault earthquake in the San Francisco Bay
  Region"* — <https://journals.sagepub.com/doi/10.1177/87552930251340960>

## Access

The portal requires a free account (Signup/Login) before downloading. Its
[Documentation section](https://sgmd.peer.berkeley.edu/documentation/) offers
a `Simulated_Ground_Motion_Database_User_Guide.pdf` and a `Glossary.pdf`;
those, not this file, are the authoritative source for the exact download
procedure (web UI vs. bulk/API) and directory layout, since **this repo's
maintainers have not independently downloaded a file to verify the
mechanics** — do not assume anything below beyond what the portal's own docs
say. For questions the portal's own docs don't answer, contact
`peer_center@berkeley.edu`.

According to the DOE/LBNL announcements above, the finer-resolution (6.25 m
spacing) simulations are stored as HDF5, transferred in ~2 GB partitions.
The coarser (2 km spacing) series are plain ASCII text, not HDF5, and are not
useful for this harness.

## Finding the dataset name

Unlike NOAA GOES-18 (which has a well-known `"Rad"` variable — see
`docs/NOAA_DATA.md`), EQSIM/SGMD HDF5 files have no single conventional
chunked-dataset name this repo can assume. Once you have a real `.h5` file,
find its dataset name(s) yourself, e.g.:

```bash
h5dump -n /path/to/eqsim_output.h5
```

or, without the HDF5 command-line tools installed, using this repo's own
Python bindings:

```bash
python3 -c "import clawhdf5; f = clawhdf5.File('/path/to/eqsim_output.h5'); print(f.keys())"
```

Then run the harness with both flags:

```bash
cargo run -p clawhdf5-attack-harness --release -- \
    --eqsim-file /path/to/eqsim_output.h5 --eqsim-dataset <name-from-above>
```

`--eqsim-file` **requires** `--eqsim-dataset` alongside it; the harness
refuses to run rather than guess (see `main.rs::load_eqsim_dataset`) —
guessing a plausible-looking default and silently falling back to the
synthetic fixture on failure would make a broken invocation look like a
successful real-data run.

## No file committed to this repo

Per the same reasoning as `docs/NOAA_DATA.md`: no real EQSIM/SGMD data is
committed here. The default `cargo run` (no `--eqsim-file`) uses
`fixture::synthetic_eqsim_dataset()`, an in-process stand-in shaped like an
HPC checkpoint write (few, large chunks) rather than downloaded data, so the
committed `attack-results/matrix.csv` is always reproducible without any
account or download.
