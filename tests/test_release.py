#!/usr/bin/env python3
# Copyright (c) KMG. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
##

"""Tests for the GitHub release coordinator."""

from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import create_github_release
from scripts.project_policy import application_version, load_policy


class GitHubReleaseTest(unittest.TestCase):
    """Verify release naming, notes, checksums, and completion checks."""

    def setUp(self) -> None:
        self.policy = load_policy()
        self.version = application_version(self.policy)

    def test_repository_and_asset_names_come_from_policy(self):
        self.assertEqual("kmgowda/sbk-charts", create_github_release.repository_slug(self.policy))
        self.assertEqual(
            "kmgowda/sbk-charts",
            create_github_release.remote_repository_slug(
                "git@github.com:kmgowda/sbk-charts.git"
            ),
        )
        self.assertEqual(
            "kmgowda/sbk-charts",
            create_github_release.remote_repository_slug(
                "https://github.com/kmgowda/sbk-charts.git"
            ),
        )
        wheel, source = create_github_release.package_asset_names(
            self.policy,
            self.version,
        )
        self.assertEqual(f"sbk_charts-{self.version}-py3-none-any.whl", wheel)
        self.assertEqual(f"sbk_charts-{self.version}.tar.gz", source)
        portable = create_github_release.expected_portable_asset_names(
            self.policy,
            self.version,
        )
        self.assertEqual(len(self.policy.portable.targets) * 2, len(portable))
        for target in self.policy.portable.targets:
            extension = self.policy.portable.self_extracting_extensions[target]
            name = f"sbk-charts-{self.version}-{target}.{extension}"
            self.assertIn(name, portable)
            self.assertIn(name + self.policy.portable.checksum_suffix, portable)

    def test_generated_and_checkout_only_files_are_rejected(self):
        forbidden = create_github_release.forbidden_tracked_files(
            [
                "README.md",
                "samples/data.csv",
                "samples/.DS_Store",
                ".idea/workspace.xml",
                "dist/sbk_charts-1.0.0.tar.gz",
                "src/sbk_charts.egg-info/SOURCES.txt",
                "src/__pycache__/module.pyc",
                "sbk-charts-1.0.0-linux-amd64.run",
                ".sbk-charts-runtime",
            ]
        )
        self.assertEqual(
            (
                ".idea/workspace.xml",
                ".sbk-charts-runtime",
                "dist/sbk_charts-1.0.0.tar.gz",
                "samples/.DS_Store",
                "sbk-charts-1.0.0-linux-amd64.run",
                "src/__pycache__/module.pyc",
                "src/sbk_charts.egg-info/SOURCES.txt",
            ),
            forbidden,
        )

    def test_current_tracked_sources_pass_the_release_audit(self):
        create_github_release.verify_tracked_release_sources()

    def test_package_build_writes_reproducible_checksum_manifest(self):
        wheel_name, source_name = create_github_release.package_asset_names(
            self.policy,
            self.version,
        )

        def fake_run(arguments, **_kwargs):
            output = Path(arguments[arguments.index("--outdir") + 1])
            (output / wheel_name).write_bytes(b"wheel")
            (output / source_name).write_bytes(b"source")

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "release"
            with patch.object(create_github_release, "run", side_effect=fake_run):
                assets = create_github_release.build_package_assets(
                    self.policy,
                    self.version,
                    output,
                    "python",
                )
            checksum_lines = assets.checksums.read_text(encoding="utf-8").splitlines()
            self.assertEqual(2, len(checksum_lines))
            self.assertTrue(checksum_lines[0].endswith(f"  {wheel_name}"))
            self.assertTrue(checksum_lines[1].endswith(f"  {source_name}"))
            self.assertTrue(assets.expected_names.issuperset({wheel_name, source_name, "SHA256SUMS"}))

    def test_portable_release_directory_requires_exact_checksummed_assets(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            expected_names = create_github_release.expected_portable_asset_names(
                self.policy,
                self.version,
            )
            checksum_suffix = self.policy.portable.checksum_suffix
            for name in expected_names:
                if name.endswith(checksum_suffix):
                    continue
                application = directory / name
                application.write_bytes(name.encode("utf-8"))
                (directory / (name + checksum_suffix)).write_text(
                    f"{create_github_release.sha256(application)}  {name}\r\n",
                    encoding="utf-8",
                )
            validated = create_github_release.verify_portable_asset_directory(
                self.policy,
                self.version,
                directory,
            )
            self.assertEqual(expected_names, frozenset(path.name for path in validated))

    def test_portable_release_directory_rejects_missing_or_modified_assets(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            expected_names = create_github_release.expected_portable_asset_names(
                self.policy,
                self.version,
            )
            checksum_suffix = self.policy.portable.checksum_suffix
            for name in expected_names:
                if name.endswith(checksum_suffix):
                    continue
                application = directory / name
                application.write_bytes(b"verified")
                (directory / (name + checksum_suffix)).write_text(
                    f"{create_github_release.sha256(application)}  {name}\n",
                    encoding="utf-8",
                )
            missing = directory / sorted(expected_names)[0]
            missing.unlink()
            with self.assertRaisesRegex(ValueError, "missing"):
                create_github_release.verify_portable_asset_directory(
                    self.policy,
                    self.version,
                    directory,
                )

            missing.write_text("unexpected replacement", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Invalid checksum file"):
                create_github_release.verify_portable_asset_directory(
                    self.policy,
                    self.version,
                    directory,
                )

    def test_portable_release_directory_rejects_a_different_version(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "canonical application version"):
                create_github_release.verify_portable_asset_directory(
                    self.policy,
                    "9.9.9.9",
                    Path(temporary),
                )

    def test_generated_notes_describe_every_delivery(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            assets = create_github_release.ReleaseAssets(
                wheel=root / f"sbk_charts-{self.version}-py3-none-any.whl",
                source_distribution=root / f"sbk_charts-{self.version}.tar.gz",
                checksums=root / "SHA256SUMS",
                expected_names=frozenset(),
            )
            with (
                patch.object(create_github_release, "previous_tag", return_value="4.26.8.1"),
                patch.object(create_github_release, "change_summary", return_value=["Improve releases"]),
            ):
                notes = create_github_release.generated_release_notes(
                    self.policy,
                    self.version,
                    "kmgowda/sbk-charts",
                    "a" * 40,
                    assets,
                )
        self.assertIn("Improve releases", notes)
        self.assertIn("SHA256SUMS", notes)
        self.assertIn("compare/4.26.8.1", notes)
        for name in create_github_release.expected_portable_asset_names(
            self.policy,
            self.version,
        ):
            if not name.endswith(self.policy.portable.checksum_suffix):
                self.assertIn(name, notes)

    def test_wait_for_assets_requires_the_complete_expected_set(self):
        releases = iter(
            (
                {"assets": [{"name": "wheel"}]},
                {"assets": [{"name": "wheel"}, {"name": "native"}]},
            )
        )
        with (
            patch.object(create_github_release, "release_information", side_effect=lambda *_: next(releases)),
            patch.object(create_github_release.time, "sleep"),
        ):
            create_github_release.wait_for_assets(
                "kmgowda/sbk-charts",
                self.version,
                frozenset({"wheel", "native"}),
                timeout_seconds=10,
                poll_seconds=1,
            )

    def test_release_asset_names_rejects_malformed_api_data(self):
        self.assertEqual(
            frozenset({"wheel"}),
            create_github_release.release_asset_names(
                {"assets": [{"name": "wheel"}]}
            ),
        )
        with self.assertRaisesRegex(ValueError, "must be a list"):
            create_github_release.release_asset_names({"assets": "wheel"})

    def test_remote_tag_lookup_accepts_only_the_exact_version(self):
        exact_commit = "a" * 40
        similar_commit = "b" * 40
        remote_tags = "\n".join(
            (
                f"{similar_commit}\trefs/tags/{self.version}.1",
                f"{similar_commit}\trefs/tags/{self.version}-candidate",
                f"{similar_commit}\trefs/tags/{self.version}",
                f"{exact_commit}\trefs/tags/{self.version}^{{}}",
            )
        )
        self.assertEqual(
            exact_commit,
            create_github_release.remote_tag_commit(remote_tags, self.version),
        )
        self.assertIsNone(
            create_github_release.remote_tag_commit(
                f"{similar_commit}\trefs/tags/{self.version}.1",
                self.version,
            )
        )

    def test_ensure_tag_queries_exact_remote_refs(self):
        head = "a" * 40
        with (
            patch.object(
                create_github_release,
                "run",
                return_value=create_github_release.subprocess.CompletedProcess(
                    args=[], returncode=0
                ),
            ),
            patch.object(
                create_github_release,
                "captured",
                side_effect=(head, f"{head}\trefs/tags/{self.version}^{{}}"),
            ) as mocked_captured,
        ):
            create_github_release.ensure_tag(self.version, head, "origin")
        remote_command = mocked_captured.call_args_list[1].args[0]
        self.assertEqual(f"refs/tags/{self.version}", remote_command[-2])
        self.assertEqual(f"refs/tags/{self.version}^{{}}", remote_command[-1])
        self.assertNotIn("*", " ".join(remote_command))

    def test_portable_workflow_supports_release_recovery(self):
        workflow = (
            create_github_release.ROOT / ".github" / "workflows" / "portable.yml"
        ).read_text(encoding="utf-8")
        self.assertIn("release_tag:", workflow)
        self.assertIn("inputs.release_tag != ''", workflow)
        build_workflow, publish_workflow = workflow.split("\n  publish:\n", 1)
        self.assertNotIn("gh release upload", build_workflow)
        self.assertEqual(1, publish_workflow.count("gh release upload"))
        self.assertIn("needs: [policy, build]", publish_workflow)
        self.assertIn("runs-on: ubuntu-latest", publish_workflow)
        self.assertEqual(
            workflow.count("actions/checkout@"),
            workflow.count("ref: ${{ github.event.release.tag_name || inputs.release_tag || github.ref }}"),
        )
        self.assertIn(
            "actions/upload-artifact@bbbca2ddaa5d8feaa63e36b76fdaad77386f024f",
            build_workflow,
        )
        self.assertIn("actions/download-artifact@37930b1c2abaa49bbe596cd826c3c89aef350131", publish_workflow)
        self.assertIn("pattern: portable-*", publish_workflow)
        self.assertIn("merge-multiple: true", publish_workflow)
        self.assertIn("scripts/verify_portable_release_assets.py", publish_workflow)
        self.assertIn('gh release view "$RELEASE_TAG" --repo "$GH_REPO"', publish_workflow)
        self.assertIn('--clobber --repo "$GH_REPO"', publish_workflow)

    def test_resume_restarts_native_build_for_the_immutable_tag(self):
        with patch.object(create_github_release, "run") as mocked_run:
            create_github_release.restart_portable_workflow(
                "kmgowda/sbk-charts",
                self.version,
            )
        command = mocked_run.call_args.args[0]
        self.assertIn("workflow", command)
        self.assertIn("portable.yml", command)
        self.assertIn(f"release_tag={self.version}", command)

    def test_resume_requires_publish(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                create_github_release.parse_args(["--resume"])


if __name__ == "__main__":
    unittest.main()
