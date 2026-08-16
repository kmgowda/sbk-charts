<!--
Copyright (c) KMG. All Rights Reserved.
Licensed under the Apache License, Version 2.0.
-->

# Runtime and artifact policy

`sbk-charts.ini` is the single configuration file for metadata that several delivery systems need. It prevents the Bash launcher, PowerShell launcher, package builder, CI workflows, and portable builder from keeping different copies of the same value.

This policy controls how sbk-charts starts and how release artifacts are named and assembled. It does not control benchmark calculations or workbook appearance.

## Consumers

```mermaid
flowchart TD
    P[sbk-charts.ini] --> B[Bash launcher]
    P --> W[PowerShell launcher]
    P --> H[project_policy.py]
    H --> S[setup.py]
    H --> C[GitHub Actions matrix]
    H --> A[Portable archive builder]
    H --> T[Policy and portable tests]
```

Bash and PowerShell use small native INI readers because the launcher cannot assume Python is already installed. After an interpreter is available, `scripts/project_policy.py` provides typed and validated access.

## Sections and owners

### `[application]`

Defines the command and distribution identity, Python module and entry point, description, project URL, license label, author metadata, version-file path, and runtime-requirements path.

Consumers include `setup.py`, launchers, and portable builds.

### `[package_data]`

Maps Python packages to non-Python files that must be included, such as the banner and logo.

### `[runtime]`

Defines:

- minimum supported Python;
- default Conda environment name;
- remembered-environment state filename;
- project virtual-environment names;
- Unix interpreter search order;
- Windows interpreter launcher search order.

### `[portable]`

Defines native target names, build Python, manifest and checksum names, bundled documentation/metadata paths, entry script, and modules PyInstaller must collect.

The related mapping sections connect targets to operating systems, processor names, archive formats, and GitHub-hosted runners.

## Python policy helper

`scripts/project_policy.py` converts INI values into frozen dataclasses and validates relationships between them. Important helpers include:

| Helper | Purpose |
|---|---|
| `load_policy()` | Read and validate the complete policy. |
| `application_version()` | Read one module-level string version assignment with the Python AST, without importing application code. |
| `load_requirements()` | Ignore comments while preserving URL fragments such as `#sha256=...`. |
| `environment_matches_policy()` | Check installed distribution version and application import. |
| `runtime_details()` | Produce consistent OS, Python, and environment messages. |
| `remember_environment()` | Atomically save a successful venv or Conda selection. |
| `load_remembered_environment()` | Read a valid prior selection. |
| `github_matrix()` | Generate the portable native-runner matrix. |

Useful commands:

```bash
python scripts/project_policy.py --minimum-python
python scripts/project_policy.py --github-matrix
```

## What belongs in policy

Put a value in `sbk-charts.ini` when all of these are true:

1. it describes runtime selection, package identity, or release artifacts;
2. two or more delivery systems need it;
3. changing it should update those consumers together;
4. it can be represented safely as configuration rather than executable logic.

Examples are the Python minimum, environment candidates, package entry point, portable target names, archive formats, and bundle paths.

## What stays in code

Some values are deliberately not global policy:

| Value | Owner | Reason |
|---|---|---|
| Exact SBK CSV headers | `src/charts/constants.py` | Input-schema contract |
| R/T prefixes and Total marker | `src/sheets/constants.py` | Workbook addressing contract |
| Chart dimensions, fonts, fills, and colors | `src/charts/` | Presentation policy |
| AI model, endpoint, token, and request defaults | Each plugin | Backend-specific behavior exposed by CLI flags |
| AI total timeout | `src/ai/sbk_ai.py` | Analysis scheduling behavior |
| Retrieval scoring and tags | `src/rag/sbk_rag.py` | Algorithm behavior |
| Action commit SHAs | Workflow YAML | Security review must see the exact pinned action |
| Current version value | `src/version/sbk_version.py` | One canonical release declaration |
| Build-tool versions | `requirements-portable.txt` | Dependency management |

Centralization is not the same as putting every constant into one file. A value should stay close to the algorithm or domain that owns it.

## Version and environment freshness

The policy points to `src/version/sbk_version.py`; it does not repeat the version. Launchers compare the installed distribution version with that source declaration. They also import the configured application module and run a dependency check. A mismatch or failed check causes repair or selection of another environment.

After runtime validation and immediately before starting the application process, the launcher writes only environment kind and prefix to the state file:

```text
kind=conda
prefix=/path/to/conda/environment
```

The write is atomic. The remembered environment is a preference, not proof that the previous workbook operation completed and not an unconditional trust decision. It is validated again before reuse.

## How to change policy safely

1. Identify every consumer with `rg`.
2. Edit `sbk-charts.ini` and keep key names stable when possible.
3. Update `ProjectPolicy` dataclasses and validation when adding a key.
4. Update native Bash and PowerShell readers when they need the new key.
5. Add or change unit tests in `tests/test_portable.py`.
6. Update this document and any affected launcher or portable guide.
7. Run:

```bash
python scripts/project_policy.py --minimum-python
python scripts/project_policy.py --github-matrix
python -m unittest discover -s tests -v
bash -n sbk-charts
./sbk-charts -h
python -m build
```

8. Test PowerShell and batch launchers on Windows if runtime selection changed.
9. Build a native portable archive if portable metadata changed.

Do not add a target name without matching platform, architecture, archive-format, and runner mappings. `load_policy()` intentionally rejects incomplete target policy.
