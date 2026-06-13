# Build and Verify sbk-charts

## Overview
This skill provides guidance for building and verifying the sbk-charts package, ensuring all assets are properly bundled and the application works end-to-end.

## When to use this skill
Use this skill when:
- Building the package for distribution (wheel/tarball)
- Verifying that assets (logo, banner.txt) are bundled
- Testing end-to-end functionality after build
- Preparing for a release

## Build Process

### 1. Clean previous builds
```bash
rm -rf dist/ build/ sbk_charts.egg-info/
```

### 2. Build the package
```bash
python -m build
```
This creates:
- `dist/sbk_charts-<version>-py3-none-any.whl`
- `dist/sbk_charts-<version>.tar.gz`

### 3. Verify bundled assets
```bash
# Check wheel contents
unzip -l dist/sbk_charts-<version>-py3-none-any.whl | grep -E "banner|sbk-logo"

# Check tarball contents
tar -tzf dist/sbk_charts-<version>.tar.gz | grep -E "banner|sbk-logo"
```

Expected output:
```
src/images/sbk-logo.png
src/main/banner.txt
```

## End-to-End Verification

### Test with a fresh virtual environment
```bash
# Create fresh venv
python3 -m venv /tmp/venv-test
/tmp/venv-test/bin/pip install dist/sbk_charts-<version>-py3-none-any.whl

# Run from different directory
cd /tmp
/tmp/venv-test/bin/sbk-charts -i <path-to-sample-csv> -o /tmp/test-output.xlsx
```

### Verify output
- Banner should display correctly
- Logo should be inserted in the SBK sheet
- Output xlsx should be created successfully
- Exit code should be 0

## Common Issues

### Logo or banner not found
- Check that `src/images/__init__.py` exists
- Verify `setup.py` has correct `package_data` configuration
- Ensure `pyproject.toml` is present with build backend specified

### Build fails with import error
- This is likely the version import issue in setup.py
- The fix is to read version from file instead of importing
- Verify setup.py uses the `get_version()` function

### Files missing in wheel
- Check `MANIFEST.in` includes the files
- Verify `include_package_data=True` in setup.py
- Ensure package keys in `package_data` match actual package names (e.g., `src.main` not `main`)

## Release Checklist
- [ ] Version updated in `src/version/sbk_version.py`
- [ ] Changes committed to git
- [ ] Clean build performed
- [ ] Assets verified in wheel and tarball
- [ ] End-to-end test passed with fresh install
- [ ] Git tag created
- [ ] Release published on GitHub
