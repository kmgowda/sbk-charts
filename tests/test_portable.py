import hashlib
import json
import subprocess
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from scripts import build_portable, sbk_charts_portable_entry
from scripts.project_policy import POLICY_FILE, github_matrix, load_policy


class PortableReleaseTest(unittest.TestCase):
    def setUp(self):
        self.commands: list[list[str]] = []
        self.policy = load_policy()

    def fake_run(self, command, **_kwargs):
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

    def test_central_policy_defines_runtime_and_artifact_metadata(self):
        self.assertEqual("3.10", self.policy.runtime.minimum_python)
        self.assertEqual("venv-sbk-charts", self.policy.runtime.virtual_environment_names[0])
        self.assertEqual(
            {"linux-amd64", "macos-arm64", "windows-amd64"},
            set(self.policy.portable.targets),
        )
        self.assertEqual(set(self.policy.portable.targets), set(self.policy.portable.archive_formats))
        self.assertEqual(set(self.policy.portable.targets), set(self.policy.portable.runners))
        self.assertTrue(POLICY_FILE.is_file())

        matrix = github_matrix(self.policy)
        self.assertEqual(len(self.policy.portable.targets), len(matrix["include"]))
        self.assertEqual(set(self.policy.portable.targets), {item["target"] for item in matrix["include"]})
        self.assertTrue(all(item["python"] == self.policy.portable.build_python for item in matrix["include"]))

    def test_launchers_and_ci_consume_runtime_policy(self):
        bash_launcher = (build_portable.ROOT / "sbk-charts").read_text(encoding="utf-8")
        powershell_launcher = (build_portable.ROOT / "sbk-charts.ps1").read_text(encoding="utf-8")
        workflow = (build_portable.ROOT / ".github" / "workflows" / "python-app.yml").read_text(
            encoding="utf-8"
        )
        self.assertIn("policy_value runtime minimum_python", bash_launcher)
        self.assertIn('"runtime.minimum_python"', powershell_launcher)
        self.assertNotIn('MINIMUM_PYTHON="3.10"', bash_launcher)
        self.assertNotIn('$MinimumPython = "3.10"', powershell_launcher)
        self.assertIn("scripts/project_policy.py --minimum-python", workflow)
        self.assertIn("needs.policy.outputs.minimum_python", workflow)

    def test_packaging_consumes_application_metadata(self):
        setup_source = (build_portable.ROOT / "setup.py").read_text(encoding="utf-8")
        self.assertIn("project_policy.application.distribution_name", setup_source)
        self.assertIn("project_policy.application.entry_point", setup_source)
        self.assertIn("project_policy.runtime.minimum_python", setup_source)
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


if __name__ == "__main__":
    unittest.main()
