from __future__ import annotations

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).with_name("collect_rocm_env.py")
spec = importlib.util.spec_from_file_location("collector", SCRIPT)
collector = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(collector)


class CollectorTests(unittest.TestCase):
    def test_package_filter(self):
        for name in ("torch", "_rocm_sdk_core", "rocm-sdk", "amd-torch-device-gfx1201",
                     "pytorch-triton-rocm", "triton-windows", "apache_tvm_ffi"):
            self.assertTrue(collector.relevant_package(name), name)
        self.assertFalse(collector.relevant_package("unrelated-private-package"))

    def test_home_redaction_is_recursive(self):
        data = {"path": str(Path.home() / "workspace"), "nested": [str(Path.home() / "env")]}
        actual = collector.redact_home(data)
        self.assertEqual(actual["path"], "<HOME>/workspace")
        self.assertEqual(actual["nested"], ["<HOME>/env"])

    def test_missing_git(self):
        with patch.object(collector.shutil, "which", return_value=None):
            self.assertEqual(collector.git_inventory(Path.cwd()), {"available": False})

    def test_git_status_does_not_include_file_names(self):
        replies = [{"returncode": 0, "stdout": "a" * 40},
                   {"returncode": 0, "stdout": "work"},
                   {"returncode": 0, "stdout": " M private-file.py\n?? hidden.txt"}]
        with patch.object(collector.shutil, "which", return_value="git"), \
             patch.object(collector, "run_command", side_effect=replies):
            report = collector.git_inventory(Path.cwd())
        self.assertTrue(report["dirty"])
        self.assertEqual(report["status_entry_count"], 2)
        self.assertNotIn("private-file", json.dumps(report))

    def test_probe_is_opt_in(self):
        with patch.object(collector, "torch_inventory") as probe:
            report = collector.collect(Path.cwd(), "test", False)
        probe.assert_not_called()
        self.assertEqual(report["torch_probe"], {"not_run": True})

    def test_probe_reads_its_json_marker(self):
        result = {"returncode": 0, "stdout": 'other message\nFREETOKEN_ENV_JSON={"hip_build_version":"test"}'}
        with patch.object(collector, "run_command", return_value=result):
            report = collector.torch_inventory()
        self.assertEqual(report["hip_build_version"], "test")

    def test_output_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "keep.json"
            path.write_text("original", encoding="utf-8")
            with patch.object(collector.sys, "argv", [str(SCRIPT), "--out", str(path)]), \
                 contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as exit_info:
                    collector.main()
            self.assertEqual(exit_info.exception.code, 2)
            self.assertEqual(path.read_text(), "original")

    def test_writes_only_requested_new_json(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "report.json"
            with patch.object(collector.sys, "argv", [str(SCRIPT), "--out", str(path)]), \
                 patch.object(collector, "collect", return_value={"ok": True}), \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(collector.main(), 0)
            self.assertEqual(json.loads(path.read_text()), {"ok": True})


if __name__ == "__main__":
    unittest.main()
