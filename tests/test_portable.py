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


class PortableReleaseTest(unittest.TestCase):
    def setUp(self):
        self.commands: list[list[str]] = []

    def fake_run(self, command, **_kwargs):
        self.commands.append(command)
        if "PyInstaller" in command:
            dist = Path(command[command.index("--distpath") + 1]) / "sbk-charts"
            dist.mkdir(parents=True)
            executable_name = (
                "sbk-charts.exe" if build_portable.current_platform().startswith("windows-") else "sbk-charts"
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
            patch("src.main.sbk_charts.sbk_charts", side_effect=application),
        ):
            self.assertEqual(0, sbk_charts_portable_entry.main(["-i", "input.csv", "-o", "output.xlsx"]))

        self.assertEqual(["sbk-charts", "-i", "input.csv", "-o", "output.xlsx"], captured)

    def test_builder_creates_manifested_checksummed_archive(self):
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch.object(build_portable, "application_version", return_value="1.2.3.4"),
            patch.object(build_portable, "current_platform", return_value="linux-amd64"),
            patch.object(build_portable.subprocess, "run", side_effect=self.fake_run),
        ):
            archive = build_portable.build_bundle(Path(temporary))
            self.assertEqual("sbk-charts-1.2.3.4-linux-amd64.tar.gz", archive.name)
            checksum_path = archive.with_suffix(archive.suffix + ".sha256")
            expected = hashlib.sha256(archive.read_bytes()).hexdigest()
            self.assertEqual(f"{expected}  {archive.name}\n", checksum_path.read_text(encoding="utf-8"))
            with tarfile.open(archive, "r:gz") as source:
                root = "sbk-charts-1.2.3.4-linux-amd64"
                manifest = json.load(source.extractfile(f"{root}/manifest.json"))
            self.assertEqual("1.2.3.4", manifest["version"])
            self.assertEqual("linux-amd64", manifest["platform"])
            self.assertIn("sbk-charts", manifest["files"])
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
            self.assertEqual("sbk-charts-1.2.3.4-windows-amd64.zip", archive.name)
            with zipfile.ZipFile(archive) as source:
                self.assertIn(
                    "sbk-charts-1.2.3.4-windows-amd64/manifest.json",
                    source.namelist(),
                )

    def test_release_workflow_uses_native_pinned_builds(self):
        workflow = (build_portable.ROOT / ".github" / "workflows" / "portable.yml").read_text(
            encoding="utf-8"
        )
        self.assertIn("ubuntu-24.04", workflow)
        self.assertIn("macos-15", workflow)
        self.assertIn("windows-2022", workflow)
        self.assertIn('"pyinstaller==6.22.0"', workflow)
        self.assertIn('"pyinstaller-hooks-contrib==2026.6"', workflow)
        self.assertIn("https://download.pytorch.org/whl/cpu", workflow)
        self.assertIn("gh release upload", workflow)


if __name__ == "__main__":
    unittest.main()
