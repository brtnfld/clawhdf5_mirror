# Obtaining NOAA GOES-18 Test Data

This guide explains how to obtain NOAA GOES-18 satellite data for testing the chunk iteration and Merkle tree verification examples.

## Quick Start

### Option 1: AWS S3 (Fastest)

NOAA publishes GOES-18 data to a public S3 bucket. No AWS account required.

```bash
# Install AWS CLI if needed
# Ubuntu/Debian: sudo apt install awscli
# macOS: brew install awscli

# Download a single file (~50 MB)
aws s3 cp \
  s3://noaa-goes18/ABI-L1b-RadC/2024/001/00/OR_ABI-L1b-RadC-M6C01_G18_s20240010001172_e20240010003545_c20240010003583.nc \
  goes18_sample.nc \
  --no-sign-request

# Or download an hour of data (~1 GB)
aws s3 cp \
  s3://noaa-goes18/ABI-L1b-RadC/2024/001/00/ \
  ./goes18_data/ \
  --recursive \
  --no-sign-request
```

### Option 2: NOAA Data Portal

1. Visit https://www.ncei.noaa.gov/access/satellite-data/
2. Select "GOES-R Series" → "GOES-18"
3. Choose "ABI L1b Radiances" (these are chunked HDF5/NetCDF4 files)
4. Select date range and download

### Option 3: Google Cloud Storage

```bash
# GOES-18 is also mirrored on GCS
gsutil cp \
  gs://gcp-public-data-goes-18/ABI-L1b-RadC/2024/001/00/OR_ABI-L1b-RadC-M6C01_G18_s20240010001172_e20240010003545_c20240010003583.nc \
  goes18_sample.nc
```

## File Structure

GOES-18 ABI L1b files are HDF5/NetCDF4 with:

- **Chunked datasets**: `Rad` (radiance data), `DQF` (quality flags)
- **Compression**: Typically DEFLATE (gzip) compressed
- **Chunk sizes**: Usually 226×226 or similar for 2D imagery

## Running the Tests

### Chunk Iteration Test

```bash
cd /path/to/clawhdf5_mirror

# Test on downloaded GOES-18 file
cargo run --example goes18_chunk_test --release -- ./goes18_sample.nc

# Or test on synthetic data (no download needed)
cargo run --example generate_10gb --release 1
cargo run --example goes18_chunk_test --release -- ./synthetic_1gb.h5
```

### Expected Output

```
=== HDF5 Chunk Modification Test ===

File: goes18_sample.nc

--- Step 1: Exploring file structure ---
Root datasets: ["Rad", "DQF", "t", "y", "x", ...]
Root groups: []

Testing dataset: Rad
  Shape: [5424, 5424]
  Dtype: I16
  Layout: Chunked v4
  Chunk dimensions: [226, 226]
  Filters: 1 filters applied
    - Filter ID 1: deflate

--- Step 2: Chunk Information ---
Number of chunks: 576
Elements per chunk: 51076
...
```

## Dataset Characteristics

| Dataset | Typical Size | Chunks | Compression |
|---------|-------------|--------|-------------|
| Full Disk (RadF) | ~2 GB | ~10,000 | DEFLATE |
| CONUS (RadC) | ~50 MB | ~500 | DEFLATE |
| Mesoscale (RadM) | ~5 MB | ~50 | DEFLATE |

## Troubleshooting

### "Dataset is compressed"

The test will skip write modifications on compressed data (to avoid corruption) but will still iterate and checksum all chunks.

### "No DataLayout"

The file may not contain chunked datasets. Try a different product or use the synthetic generator instead.

## References

- [NOAA GOES-R Series Data](https://www.goes-r.gov/users/hrit.html)
- [AWS Open Data - NOAA GOES](https://registry.opendata.aws/noaa-goes/)
- [GOES-18 Product Guide](https://www.ospo.noaa.gov/Products/Suites/GOES-18.html)
