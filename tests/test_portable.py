import hashlib
import json
import re
import subprocess
import sys
import tarfile
import tempfile
import unittest
import zipfile
from dataclasses import replace
from pathlib import Path, PureWindowsPath
from unittest.mock import patch

from scripts import build_portable, sbk_charts_portable_entry
from scripts.project_policy import (
    POLICY_FILE,
    application_version,
    environment_matches_policy,
    github_matrix,
    load_remembered_environment,
    load_policy,
    load_requirements,
    remember_environment,
    runtime_details,
)
from src.ai.registry import BACKENDS


BACKEND_GUIDES = frozenset(
    {
        "src/custom_ai/README.md",
        "src/custom_ai/anthropic/README.md",
        "src/custom_ai/gemini/README.md",
        "src/custom_ai/hugging_face/README.md",
        "src/custom_ai/lm_studio/README.md",
        "src/custom_ai/no_ai/README.md",
        "src/custom_ai/ollama/README.md",
        "src/custom_ai/pytorch_llm/README.md",
    }
)


class PortableReleaseTest(unittest.TestCase):
    def setUp(self) -> None:
        self.commands: list[list[str]] = []
        self.policy = load_policy()

    def fake_run(self, command: list[str], **_kwargs: object) -> subprocess.CompletedProcess:
        self.commands.append(command)
        if "PyInstaller" in command:
            application_name = self.policy.application.name
            dist = Path(command[command.index("--distpath") + 1]) / application_name
            dist.mkdir(parents=True)
            executable_name = (
                f"{application_name}.exe"
                if build_portable.current_platform().startswith("windows-")
                else application_name
            )
            executable = dist / executable_name
            executable.write_bytes(b"portable executable")
            executable.chmod(0o755)
            (dist / "_internal").mkdir()
            (dist / "_internal" / "runtime.dat").write_bytes(b"runtime")
        return subprocess.CompletedProcess(command, 0)

    def test_portable_builder_runs_without_installed_project_paths(self):
        repository_root = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [
                sys.executable,
                "-I",
                "-S",
                str(repository_root / "scripts" / "build_portable.py"),
                "--help",
            ],
            cwd=repository_root,
            capture_output=True,
            check=False,
            text=True,
        )

        self.assertEqual(0, result.returncode, result.stderr)
        self.assertIn("Build a self-contained", result.stdout)

    def test_portable_entry_forwards_arguments_to_the_application(self):
        captured: list[str] = []

        def application():
            captured.extend(sbk_charts_portable_entry.sys.argv)

        with (
            patch.object(sbk_charts_portable_entry.sys, "argv", ["original-command"]),
            patch.object(sbk_charts_portable_entry.sys, "executable", "/bundle/sbk-charts"),
            patch("src.main.sbk_charts.sbk_charts", side_effect=application),
        ):
            self.assertEqual(0, sbk_charts_portable_entry.main(["-i", "input.csv", "-o", "output.xlsx"]))

        self.assertEqual(
            [self.policy.application.name, "-i", "input.csv", "-o", "output.xlsx"],
            captured,
        )

    def test_builder_creates_manifested_checksummed_archive(self):
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch.object(build_portable, "application_version", return_value="1.2.3.4"),
            patch.object(build_portable, "current_platform", return_value="linux-amd64"),
            patch.object(build_portable.subprocess, "run", side_effect=self.fake_run),
        ):
            archive = build_portable.build_bundle(Path(temporary))
            bundle_name = f"{self.policy.application.name}-1.2.3.4-linux-amd64"
            self.assertEqual(f"{bundle_name}.tar.gz", archive.name)
            checksum_path = archive.with_suffix(archive.suffix + self.policy.portable.checksum_suffix)
            expected = hashlib.sha256(archive.read_bytes()).hexdigest()
            self.assertEqual(f"{expected}  {archive.name}\n", checksum_path.read_text(encoding="utf-8"))
            with tarfile.open(archive, "r:gz") as source:
                archive_members = set(source.getnames())
                manifest = json.load(
                    source.extractfile(f"{bundle_name}/{self.policy.portable.manifest_name}")
                )
            self.assertEqual(self.policy.application.name, manifest["application"])
            self.assertEqual("tar.gz", manifest["archive_format"])
            self.assertEqual(self.policy.portable.hash_algorithm, manifest["hash_algorithm"])
            self.assertEqual("1.2.3.4", manifest["version"])
            self.assertEqual("linux-amd64", manifest["platform"])
            self.assertIn(self.policy.application.name, manifest["files"])
            self.assertIn(POLICY_FILE.name, manifest["files"])
            self.assertIn("docs/PORTABLE.md", manifest["files"])
            for guide in BACKEND_GUIDES:
                self.assertIn(guide, manifest["files"])
                self.assertIn(f"{bundle_name}/{guide}", archive_members)
            self.assertIn("--help", self.commands[-1])

    def test_windows_target_creates_zip_archive(self):
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch.object(build_portable, "application_version", return_value="1.2.3.4"),
            patch.object(build_portable, "current_platform", return_value="windows-amd64"),
            patch.object(build_portable.subprocess, "run", side_effect=self.fake_run),
        ):
            archive = build_portable.build_bundle(Path(temporary))
            bundle_name = f"{self.policy.application.name}-1.2.3.4-windows-amd64"
            self.assertEqual(f"{bundle_name}.zip", archive.name)
            with zipfile.ZipFile(archive) as source:
                self.assertIn(
                    f"{bundle_name}/{self.policy.portable.manifest_name}",
                    source.namelist(),
                )

    def test_windows_zip_member_names_use_posix_separators(self):
        member = build_portable.zip_member_name(
            "sbk-charts-version-windows-amd64",
            PureWindowsPath("docs") / "POLICY.md",
        )
        self.assertEqual("sbk-charts-version-windows-amd64/docs/POLICY.md", member)
        self.assertNotIn("\\", member)

    def test_central_policy_defines_runtime_and_artifact_metadata(self):
        self.assertEqual("3.10", self.policy.runtime.minimum_python)
        self.assertEqual(".sbk-charts-runtime", self.policy.runtime.runtime_state_file)
        self.assertEqual("venv-sbk-charts", self.policy.runtime.virtual_environment_names[0])
        self.assertEqual(
            {"linux-amd64", "macos-arm64", "windows-amd64"},
            set(self.policy.portable.targets),
        )
        self.assertEqual(set(self.policy.portable.targets), set(self.policy.portable.archive_formats))
        self.assertEqual(set(self.policy.portable.targets), set(self.policy.portable.runners))
        self.assertTrue(BACKEND_GUIDES.issubset(self.policy.portable.bundle_paths))
        self.assertTrue(POLICY_FILE.is_file())
        self.assertEqual("3.12.10", self.policy.runtime.managed_python)
        self.assertEqual(".sbk-runtime", self.policy.runtime.managed_runtime_directory)
        self.assertEqual("requirements-lock", self.policy.runtime.lock_directory)
        self.assertEqual(300, self.policy.runtime.bootstrap_lock_timeout_seconds)
        self.assertEqual("uv", self.policy.bootstrap.manager)
        self.assertRegex(self.policy.bootstrap.manager_version, r"^\d+\.\d+\.\d+$")
        self.assertTrue(set(self.policy.portable.targets).issubset(self.policy.bootstrap.archives))
        self.assertTrue(all(re.fullmatch(r"[0-9a-f]{64}", value)
                            for value in self.policy.bootstrap.checksums.values()))
        self.assertEqual(set(BACKENDS) - {"noai"}, set(self.policy.ai_requirements))

        matrix = github_matrix(self.policy)
        self.assertEqual(len(self.policy.portable.targets), len(matrix["include"]))
        self.assertEqual(set(self.policy.portable.targets), {item["target"] for item in matrix["include"]})
        self.assertTrue(all(item["python"] == self.policy.portable.build_python for item in matrix["include"]))

    def test_version_resolution_uses_the_selected_policy_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            version_path = root / "custom" / "version.py"
            version_path.parent.mkdir()
            version_path.write_text('__sbk_version__ = "9.8.7.6"\n', encoding="utf-8")
            application = replace(self.policy.application, version_file="custom/version.py")
            selected_policy = replace(self.policy, application=application)

            self.assertEqual("9.8.7.6", application_version(selected_policy, root))

            version_path.write_text("VERSION_MISSING = True\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, re.escape(str(version_path))):
                application_version(selected_policy, root)

    def test_policy_rejects_invalid_bootstrap_checksum(self):
        with tempfile.TemporaryDirectory() as temporary:
            policy_file = Path(temporary) / "sbk-charts.ini"
            original = POLICY_FILE.read_text(encoding="utf-8")
            source, replaced = re.subn(
                r"linux-amd64-sha256 = [0-9a-f]{64}",
                "linux-amd64-sha256 = not-a-checksum",
                original,
            )
            self.assertEqual(1, replaced, "linux-amd64 checksum entry not found")
            policy_file.write_text(source, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "invalid SHA-256"):
                load_policy(policy_file)

    def test_policy_rejects_nonpositive_bootstrap_lock_timeout(self):
        with tempfile.TemporaryDirectory() as temporary:
            policy_file = Path(temporary) / "sbk-charts.ini"
            original = POLICY_FILE.read_text(encoding="utf-8")
            source, replaced = re.subn(
                r"bootstrap_lock_timeout_seconds = \d+",
                "bootstrap_lock_timeout_seconds = 0",
                original,
            )
            self.assertEqual(1, replaced, "bootstrap lock timeout entry not found")
            policy_file.write_text(source, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "at least one second"):
                load_policy(policy_file)

    def test_version_resolution_ignores_non_module_assignments_and_text(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            version_path = root / "version.py"
            application = replace(self.policy.application, version_file=version_path.name)
            selected_policy = replace(self.policy, application=application)
            invalid_sources = (
                '# __sbk_version__ = "1.0.0"\n',
                '"""__sbk_version__ = "1.0.0"""\n',
                'if False:\n    __sbk_version__ = "1.0.0"\n',
            )

            for source in invalid_sources:
                with self.subTest(source=source):
                    version_path.write_text(source, encoding="utf-8")
                    with self.assertRaisesRegex(RuntimeError, "found 0"):
                        application_version(selected_policy, root)

    def test_version_resolution_rejects_multiple_assignments(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            version_path = root / "version.py"
            version_path.write_text(
                '__sbk_version__ = "1.0.0"\n__sbk_version__ = "2.0.0"\n',
                encoding="utf-8",
            )
            application = replace(self.policy.application, version_file=version_path.name)
            selected_policy = replace(self.policy, application=application)

            with self.assertRaisesRegex(RuntimeError, "found 2"):
                application_version(selected_policy, root)

    def test_version_resolution_rejects_non_literal_assignment(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            version_path = root / "version.py"
            version_path.write_text('__sbk_version__ = str("1.0.0")\n', encoding="utf-8")
            application = replace(self.policy.application, version_file=version_path.name)
            selected_policy = replace(self.policy, application=application)

            with self.assertRaisesRegex(RuntimeError, "string literal"):
                application_version(selected_policy, root)

    def test_requirements_strip_comments_but_preserve_url_fragments(self):
        with tempfile.TemporaryDirectory() as temporary:
            requirements_file = Path(temporary) / "requirements.txt"
            requirements_file.write_text(
                "# comment\n"
                "package>=1.0  # explanation\n"
                "archive @ https://example.test/archive.whl#sha256=abc123\n",
                encoding="utf-8",
            )
            self.assertEqual(
                [
                    "package>=1.0",
                    "archive @ https://example.test/archive.whl#sha256=abc123",
                ],
                load_requirements(requirements_file),
            )

    def test_environment_policy_rejects_stale_installed_metadata(self):
        with (
            patch("scripts.project_policy.application_version", return_value="2.0.0"),
            patch("scripts.project_policy.distribution_version", return_value="1.0.0"),
            patch("scripts.project_policy.importlib.import_module") as import_module,
        ):
            self.assertFalse(environment_matches_policy(self.policy))
        import_module.assert_not_called()

    def test_environment_policy_accepts_matching_application(self):
        with (
            patch("scripts.project_policy.application_version", return_value="2.0.0"),
            patch("scripts.project_policy.distribution_version", return_value="2.0.0"),
            patch("scripts.project_policy.importlib.import_module") as import_module,
        ):
            self.assertTrue(environment_matches_policy(self.policy))
        import_module.assert_called_once_with(self.policy.application.module)

    def test_runtime_details_report_platform_python_and_environment(self):
        with (
            patch("scripts.project_policy.platform.platform", return_value="macOS-15.6-arm64"),
            patch("scripts.project_policy.platform.python_version", return_value="3.12.11"),
            patch("scripts.project_policy.sys.executable", "/project/.venv/bin/python"),
        ):
            details = runtime_details(self.policy, "venv", "/project/.venv")

        self.assertEqual(
            (
                "sbk-charts: Operating system: macOS-15.6-arm64",
                "sbk-charts: Python: 3.12.11 (/project/.venv/bin/python)",
                "sbk-charts: Environment: venv (/project/.venv)",
            ),
            details,
        )

    def test_successful_environment_is_remembered_atomically(self):
        with tempfile.TemporaryDirectory() as temporary:
            state_file = Path(temporary) / self.policy.runtime.runtime_state_file

            remember_environment("conda", "/opt/conda envs/sbk-charts", state_file)
            self.assertEqual(
                ("conda", "/opt/conda envs/sbk-charts"),
                load_remembered_environment(state_file),
            )
            remember_environment("venv", "/project/.venv", state_file)
            self.assertEqual(("venv", "/project/.venv"), load_remembered_environment(state_file))
            remember_environment(
                "managed", "/project/.sbk-runtime/envs/abc", state_file, "abc", "gemini"
            )
            self.assertEqual(
                ("managed", "/project/.sbk-runtime/envs/abc"),
                load_remembered_environment(state_file),
            )
            state = state_file.read_text(encoding="utf-8")
            self.assertIn("fingerprint=abc\n", state)
            self.assertIn("profile=gemini\n", state)
            self.assertEqual([], list(state_file.parent.glob(".*.tmp")))

    def test_core_and_backend_dependency_profiles_are_locked(self):
        root = build_portable.ROOT
        core_input = (root / "requirements.txt").read_text(encoding="utf-8").lower()
        for optional_package in (
            "anthropic", "google-genai", "huggingface_hub", "lmstudio", "torch", "transformers"
        ):
            self.assertNotIn(optional_package, core_input)

        for profile in ("core", *self.policy.ai_requirements):
            lock = root / self.policy.runtime.lock_directory / f"{profile}.txt"
            self.assertTrue(lock.is_file(), profile)
            contents = lock.read_text(encoding="utf-8")
            self.assertIn("==", contents)
            self.assertIn("--hash=sha256:", contents)

    def test_backend_registry_is_lazy_and_complete(self):
        descriptor_modules_before = {
            descriptor.module for descriptor in BACKENDS.values() if descriptor.module in sys.modules
        }
        self.assertEqual(
            {"anthropic", "gemini", "huggingface", "lmstudio", "noai", "ollama", "pytorchllm"},
            set(BACKENDS),
        )
        descriptor_modules_after = {
            descriptor.module for descriptor in BACKENDS.values() if descriptor.module in sys.modules
        }
        self.assertEqual(descriptor_modules_before, descriptor_modules_after)

    def test_invalid_remembered_environment_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            state_file = Path(temporary) / "runtime-state"
            with self.assertRaisesRegex(ValueError, "Unsupported environment kind"):
                remember_environment("system", "/usr/bin", state_file)
            self.assertIsNone(load_remembered_environment(state_file))

    def test_policy_cli_remembers_profile_without_fingerprint(self):
        with tempfile.TemporaryDirectory() as temporary:
            state_file = Path(temporary) / "runtime-state"
            result = subprocess.run(
                [
                    sys.executable,
                    str(build_portable.ROOT / "scripts" / "project_policy.py"),
                    "--remember-environment",
                    "venv",
                    "/project/venv",
                    str(state_file),
                    "ollama",
                ],
                cwd=build_portable.ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(0, result.returncode, result.stderr)
            remembered = load_remembered_environment(state_file)
            self.assertEqual(("venv", "/project/venv"), remembered)
            state_values = dict(
                line.split("=", 1)
                for line in state_file.read_text(encoding="utf-8").splitlines()
            )
            self.assertEqual("", state_values["fingerprint"])
            self.assertEqual("ollama", state_values["profile"])

    def test_launchers_and_ci_consume_runtime_policy(self):
        bash_launcher = (build_portable.ROOT / "sbk-charts").read_text(encoding="utf-8")
        powershell_launcher = (build_portable.ROOT / "sbk-charts.ps1").read_text(encoding="utf-8")
        workflow = (build_portable.ROOT / ".github" / "workflows" / "python-app.yml").read_text(
            encoding="utf-8"
        )
        self.assertIn("policy_value runtime minimum_python", bash_launcher)
        self.assertIn("scripts/project_policy.py\" --environment-ready", bash_launcher)
        self.assertIn('--runtime-details "$environment_kind" "$environment_prefix"', bash_launcher)
        self.assertIn("policy_value runtime runtime_state_file", bash_launcher)
        self.assertIn("policy_value runtime managed_python", bash_launcher)
        self.assertIn("policy_value runtime bootstrap_lock_timeout_seconds", bash_launcher)
        self.assertIn("--require-hashes --requirement", bash_launcher)
        self.assertIn("venv --relocatable --managed-python", bash_launcher)
        self.assertIn("python_can_create_venv", bash_launcher)
        self.assertIn("cannot create a working venv; trying the next candidate", bash_launcher)
        self.assertIn('rm -rf "$temporary_dir"', bash_launcher)
        self.assertNotIn("\n    \\\n        \"$1\" -m pip check", bash_launcher)
        self.assertIn("Trying remembered managed environment", bash_launcher)
        self.assertIn('--remember-environment "$environment_kind" "$environment_prefix"', bash_launcher)
        self.assertIn(
            'if [ -z "${SBK_CHARTS_VENV:-}" ] && [ -n "$EXPECTED_FINGERPRINT" ]; then',
            bash_launcher,
        )
        self.assertNotIn(
            'acquire_bootstrap_lock\n    try_managed_environment',
            bash_launcher,
        )
        self.assertLess(
            bash_launcher.index("Trying remembered Conda environment"),
            bash_launcher.index('try_virtual_environment "${VIRTUAL_ENV:-}"'),
        )
        self.assertIn('"runtime.minimum_python"', powershell_launcher)
        self.assertIn("$PolicyReader --environment-ready", powershell_launcher)
        self.assertIn("--runtime-details $EnvironmentKind $EnvironmentPrefix", powershell_launcher)
        self.assertIn('"runtime.runtime_state_file"', powershell_launcher)
        self.assertIn('"runtime.managed_python"', powershell_launcher)
        self.assertIn('"runtime.bootstrap_lock_timeout_seconds"', powershell_launcher)
        self.assertIn("--require-hashes --requirement", powershell_launcher)
        self.assertIn("venv --relocatable --managed-python", powershell_launcher)
        self.assertIn("Test-PythonLauncherVenv", powershell_launcher)
        self.assertIn("[int]::TryParse", powershell_launcher)
        self.assertIn(
            "if ($TemporaryEnvironment -and (Test-Path -LiteralPath $TemporaryEnvironment))",
            powershell_launcher,
        )
        self.assertIn(
            "if ($TemporaryDirectory -and (Test-Path -LiteralPath $TemporaryDirectory))",
            powershell_launcher,
        )
        self.assertIn("Trying remembered managed environment", powershell_launcher)
        self.assertIn("--remember-environment $EnvironmentKind", powershell_launcher)
        self.assertIn("if ($EnvironmentFingerprint)", powershell_launcher)
        self.assertIn(
            "if (-not $env:SBK_CHARTS_VENV -and $ExpectedFingerprint)",
            powershell_launcher,
        )
        environment_calls = re.findall(
            r"^\s+Use-EnvironmentPrefix .+$", powershell_launcher, flags=re.MULTILINE
        )
        self.assertTrue(environment_calls)
        self.assertTrue(all("-EnvironmentPrefix" in call for call in environment_calls))
        self.assertLess(
            powershell_launcher.index("Trying remembered Conda environment"),
            powershell_launcher.index("$EnvironmentCandidates ="),
        )
        self.assertNotIn('MINIMUM_PYTHON="3.10"', bash_launcher)
        self.assertNotIn('$MinimumPython = "3.10"', powershell_launcher)
        self.assertIn("scripts/project_policy.py --minimum-python", workflow)
        self.assertIn("needs.policy.outputs.minimum_python", workflow)
        self.assertIn("managed-bootstrap-unix:", workflow)
        self.assertIn("managed-bootstrap-windows:", workflow)
        self.assertIn("Skip a Python interpreter with broken venv support", workflow)
        self.assertIn("Skip a Python launcher with broken venv support", workflow)
        self.assertIn("SBK_BROKEN_PY_MARKER", workflow)
        self.assertIn("Clean an unpublished managed environment after failure", workflow)
        self.assertIn('UV_OFFLINE: "true"', workflow)
        self.assertIn("python -m build --wheel --sdist", workflow)
        self.assertNotIn("actions/checkout@v", workflow)
        self.assertNotIn("actions/setup-python@v", workflow)
        self.assertEqual(workflow.count("actions/checkout@"), workflow.count("persist-credentials: false"))

    def test_packaging_consumes_application_metadata(self):
        setup_source = (build_portable.ROOT / "setup.py").read_text(encoding="utf-8")
        self.assertIn("project_policy.application.distribution_name", setup_source)
        self.assertIn("project_policy.application.entry_point", setup_source)
        self.assertIn("project_policy.runtime.minimum_python", setup_source)
        self.assertIn("extras_required['all-ai']", setup_source)
        self.assertNotIn("Fallback to hardcoded requirements", setup_source)

    def test_release_workflow_uses_native_pinned_builds(self):
        workflow = (build_portable.ROOT / ".github" / "workflows" / "portable.yml").read_text(
            encoding="utf-8"
        )
        self.assertIn("scripts/project_policy.py --github-matrix", workflow)
        self.assertIn("fromJSON(needs.policy.outputs.matrix)", workflow)
        self.assertIn("-r requirements-portable.txt", workflow)
        portable_requirements = (build_portable.ROOT / "requirements-portable.txt").read_text(
            encoding="utf-8"
        )
        self.assertIn("pyinstaller==6.22.0", portable_requirements)
        self.assertIn("pyinstaller-hooks-contrib==2026.6", portable_requirements)
        self.assertIn("https://download.pytorch.org/whl/cpu", workflow)
        self.assertIn("gh release upload", workflow)
        self.assertEqual(workflow.count("actions/checkout@"), workflow.count("persist-credentials: false"))
        self.assertIn("cache-dependency-path: |", workflow)
        self.assertIn("requirements-ai/*.txt", workflow)
        self.assertIn("requirements-bootstrap.txt", workflow)
        self.assertIn('python -m pip install ".[all-ai]"', workflow)


if __name__ == "__main__":
    unittest.main()
