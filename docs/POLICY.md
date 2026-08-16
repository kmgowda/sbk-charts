# Runtime Policy and Artifact Metadata

`sbk-charts.ini` is the canonical source for values shared by launchers,
packaging, CI, and portable releases. Change a value there when modifying:

- application and distribution identity;
- the Python minimum and interpreter search order;
- default Conda and project virtual-environment names;
- package data, version file, or runtime requirements file;
- portable targets, native runners, archive formats, and build Python;
- portable manifest, checksum, bundled paths, entry script, or collected
  plugin modules.

`scripts/project_policy.py` provides typed, validated access for Python and
emits the GitHub Actions runtime version and build matrix. Bash and PowerShell
use small native INI readers because the self-bootstrap path cannot assume that
Python is already installed. Portable-only build-tool versions are kept in
`requirements-portable.txt`; application dependencies remain in
`requirements.txt`.

## Audit boundaries

The codebase-wide hardcoded-value review intentionally leaves these values in
their owning modules:

- exact SBK CSV headers and R/T worksheet names, which are input-schema
  contracts already centralized in `src/charts/constants.py` and
  `src/sheets/constants.py`;
- chart dimensions, fonts, colors, and Excel layout coordinates, which are
  presentation policy already grouped at the top of the chart implementation;
- AI model names, endpoints, token limits, and request timeouts, which are
  backend-specific defaults exposed through each plugin's command-line flags;
- RAG scoring weights, which belong to the retrieval algorithm;
- GitHub Action commit pins and bootstrap runner selection, which must remain
  directly visible in workflow YAML for security and startup purposes;
- documentation examples and tests that assert the public contract.

This separation keeps deployment policy centralized without turning unrelated
domain behavior into a global configuration file.
