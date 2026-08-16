# OpenIO and MinIO comparison sample

This directory contains historical SBK CSV inputs and generated Excel reports for comparing OpenIO and MinIO read/write workloads. It is an example dataset, not a deployment guide for current OpenIO or MinIO releases.

## Files

| File | Meaning |
|---|---|
| `OpenIO_read_benchmarks.csv` | OpenIO read benchmark input |
| `OpenIO_write_benchmarks.csv` | OpenIO write benchmark input |
| `minio_read_benchmarks.csv` | MinIO read benchmark input |
| `minio_write_benchmarks.csv` | MinIO write benchmark input |
| `openio_minio_reader_results.xlsx` | Generated read comparison workbook |
| `openio_minio_writer_results.xlsx` | Generated write comparison workbook |

## Recreate the workbooks

From the sbk-charts repository root:

```bash
./sbk-charts \
  -i samples/charts/openio-minio/OpenIO_read_benchmarks.csv,samples/charts/openio-minio/minio_read_benchmarks.csv \
  -o samples/charts/openio-minio/openio_minio_reader_results.xlsx
```

```bash
./sbk-charts \
  -i samples/charts/openio-minio/OpenIO_write_benchmarks.csv,samples/charts/openio-minio/minio_write_benchmarks.csv \
  -o samples/charts/openio-minio/openio_minio_writer_results.xlsx
```

The order of CSV paths determines the R/T numbering and series order in the workbook.

## Historical benchmark commands

The sample CSV files were created with commands similar to these SBK invocations:

```bash
# MinIO write
./build/install/sbk/bin/sbk -class minio -writers 1 -size 100 \
  -seconds 60 -csvfile minio_write_benchmarks.csv

# MinIO read
./build/install/sbk/bin/sbk -class minio -readers 1 -size 100 \
  -seconds 60 -csvfile minio_read_benchmarks.csv
```

```bash
# OpenIO write against a reachable endpoint
./build/install/sbk/bin/sbk -class openio -url http://127.0.0.1:6007 \
  -writers 1 -size 100 -seconds 60 -csvfile OpenIO_write_benchmarks.csv

# OpenIO read
./build/install/sbk/bin/sbk -class openio -url http://127.0.0.1:6007 \
  -readers 1 -size 100 -seconds 60 -csvfile OpenIO_read_benchmarks.csv
```

Storage services, container images, ports, SBK flags, and security recommendations change over time. Consult current SBK, MinIO, and OpenIO documentation before reproducing the benchmark environment. Do not expose an unauthenticated storage endpoint merely to match this historical sample.
