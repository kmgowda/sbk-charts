<!--
Copyright (c) KMG. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0
-->

# Build and verify sbk-charts

Use this skill for packaging, release preparation, launcher changes, or a full regression check.

## Standard verification

```bash
venv-sbk-charts/bin/python -m unittest discover -s tests -v
./sbk-charts -h
./sbk-charts -i samples/charts/sbk-file-read.csv \
  -o /tmp/sbk-charts-verify.xlsx
venv-sbk-charts/bin/python -c \
  "import openpyxl; w=openpyxl.load_workbook('/tmp/sbk-charts-verify.xlsx'); print(w.sheetnames)"
venv-sbk-charts/bin/python -m flake8 . \
  --count --select=E9,F63,F7,F82 --show-source --statistics
git diff --check
```

## Package verification

```bash
venv-sbk-charts/bin/python -m build
unzip -l dist/sbk_charts-<version>-py3-none-any.whl
tar -tzf dist/sbk_charts-<version>.tar.gz
```

Confirm the banner, logo, policy file, source launchers, scripts, documentation, and developer skills appear in the artifact type where `setup.py` and `MANIFEST.in` intend them.

For release confidence, install the wheel into a fresh temporary virtual environment and run it from outside the repository. For portable changes, install `requirements-portable.txt`, run `scripts/build_portable.py` on a native target, verify the checksum and manifest, extract the whole bundle, and run the sample.

For source-launcher changes, also verify:

- first-run managed creation with a fresh `SBK_CHARTS_RUNTIME_ROOT`;
- second-run reuse with network access disabled;
- fallback past a version-compatible Python that cannot create a working venv;
- cleanup of failed `.tool.*`, venv-probe, and unpublished `.env-*` paths;
- runtime-state kind, prefix, profile, and fingerprint;
- PowerShell and CMD behavior on Windows through CI or a real Windows host.

The managed checks can download a pinned runtime and dependencies. Use an isolated temporary runtime root and do not remove a user's existing `.sbk-runtime/` directory.

Do not delete or overwrite unrelated existing artifacts. Do not claim an operating system, provider, or visual check that was not performed.
