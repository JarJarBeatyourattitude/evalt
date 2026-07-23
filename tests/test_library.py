from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from io import StringIO
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

from evalt import EvidenceLibrary, Suite, resolve_evidence_reference
from evalt.cli import main as cli_main


PNG_DATA_URL = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
    "+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


def suite_payload(*, image: bool = False, private_marker: str = "PRIVATE PROMPT"):
    examples = []
    for index in range(6):
        input_value = f"ticket {index}"
        if image:
            input_value = [
                {"type": "text", "text": f"receipt {index}"},
                {
                    "type": "image_url",
                    "image_url": {"url": PNG_DATA_URL, "detail": "low"},
                },
            ]
        examples.append(
            {
                "id": f"case-{index + 1}",
                "input": input_value,
                "approved_output": "billing" if index % 2 == 0 else "account",
            }
        )
    return {
        "schema": "evalt-suite-v2",
        "name": "private-support-suite",
        "prompt": private_marker,
        "examples": examples,
        "models": ["fixture/model-a", "fixture/model-b"],
        "optimizer_model": "fixture/optimizer",
        "evaluator_model": "fixture/evaluator",
        "evaluator": {"type": "exact_text"},
        "quality_threshold": 0.8,
        "max_optimization_cost_usd": 0.1,
        "rounds": 1,
        "optimize_prompt": not image,
        "holdout_repeats": 1,
        "target_max_tokens": 16,
    }


def result_payload(
    *,
    model: str = "fixture/model-a",
    quality: float = 1.0,
    schema: str | None = None,
):
    cases = [
        {
            "example_id": f"case-{index + 1}",
            "split": "holdout",
            "passed": True,
            "score": 1.0,
            "reason": "fixture",
            "output": "PRIVATE OUTPUT",
            "approved_output": "billing",
        }
        for index in range(3)
    ]
    winner = {
        "model": model,
        "selected_prompt": "PRIVATE SELECTED PROMPT",
        "holdout_pass_rate": quality,
        "estimated_cost_per_successful_call_usd": 0.0002,
        "target_latency_p90_ms": 150,
        "cases": cases,
    }
    value = {
        "objective": "lowest_cost_at_accuracy",
        "quality_threshold": 0.8,
        "exploratory": False,
        "winner_scope": "all completed configurations",
        "winner": winner,
        "models": [winner],
        "total_provider_spend_usd": 0.01,
        "regression_suite": {"suite_hash": "a" * 64},
    }
    if schema is not None:
        value.update(
            {
                "schema": schema,
                "monitor_status": "HEALTHY",
                "passed": True,
            }
        )
    return value


def write_json(path: Path, value, *, compact: bool = False) -> bytes:
    data = (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        if compact
        else json.dumps(value, ensure_ascii=False, indent=2)
    ).encode("utf-8") + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return data


def run_cli(*args: str):
    stdout = StringIO()
    stderr = StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        code = cli_main(list(args))
    return code, stdout.getvalue(), stderr.getvalue()


class EvidenceLibraryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.base = Path(self.temporary.name)
        self.root = self.base / "private library"
        self.library = EvidenceLibrary(self.root)
        self.suite_path = self.base / "project evidence" / "approved suite.json"
        self.suite_bytes = write_json(self.suite_path, suite_payload())
        self.result_path = self.base / "project evidence" / "frozen result.json"
        self.result_bytes = write_json(self.result_path, result_payload(), compact=True)

    def tearDown(self):
        self.temporary.cleanup()

    def add_suite(self, name: str = "support-v1"):
        return self.library.add(
            self.suite_path,
            name=name,
            tags=["production", "support"],
        )

    def add_result(self, name: str = "support-result-v1"):
        return self.library.add(
            self.result_path,
            name=name,
            tags=["baseline", "production"],
        )

    def test_add_preserves_exact_bytes_and_exposes_only_bounded_metadata(self):
        entry = self.add_suite()
        object_path = self.library.resolve(entry.name, expected_kind="suite")
        self.assertEqual(object_path.read_bytes(), self.suite_bytes)
        self.assertEqual(entry.kind, "suite")
        self.assertEqual(entry.tags, ("production", "support"))
        self.assertEqual(entry.summary["examples"], 6)
        self.assertEqual(entry.summary["models"], 2)
        self.assertFalse(entry.summary["has_images"])

        index_text = self.library.index_path.read_text(encoding="utf-8")
        self.assertNotIn("PRIVATE PROMPT", index_text)
        self.assertNotIn(str(self.suite_path), index_text)
        self.assertNotIn("fixture/model-a", index_text)
        self.assertNotIn(PNG_DATA_URL, index_text)
        self.assertIn("local-only; never synchronized", index_text)

    def test_image_suite_is_validated_and_labeled_as_multimodal(self):
        image_path = self.base / "image suite.json"
        write_json(image_path, suite_payload(image=True))
        entry = self.library.add(
            image_path, name="receipts-vision-v1", tags=["image"], kind="suite"
        )
        self.assertTrue(entry.summary["has_images"])
        loaded = self.library.read(entry.name, expected_kind="suite")
        self.assertEqual(loaded["examples"][0]["input"][1]["type"], "image_url")
        Suite.from_dict(loaded)

    def test_optimization_and_monitor_results_are_supported(self):
        optimization = self.add_result()
        self.assertEqual(optimization.kind, "result")
        self.assertEqual(
            optimization.summary["selected_model"], "fixture/model-a"
        )
        monitor_path = self.base / "monitor.json"
        write_json(
            monitor_path,
            result_payload(schema="evalt-monitor-result-v1"),
        )
        monitor = self.library.add(
            monitor_path, name="support-monitor-2026-07-23", kind="result"
        )
        self.assertEqual(monitor.summary["monitor_status"], "HEALTHY")

    def test_v1_suite_and_result_remain_compatible(self):
        legacy_suite = suite_payload()
        legacy_suite["schema"] = "evalt-suite-v1"
        legacy_suite_path = self.base / "legacy-suite.json"
        write_json(legacy_suite_path, legacy_suite)
        suite_entry = self.library.add(
            legacy_suite_path, name="legacy-suite-v1", kind="suite"
        )
        self.assertEqual(suite_entry.kind, "suite")

        legacy_result = result_payload()
        legacy_result["schema"] = "evalt-result-v1"
        legacy_result.pop("models")
        legacy_result_path = self.base / "legacy-result.json"
        write_json(legacy_result_path, legacy_result)
        result_entry = self.library.add(
            legacy_result_path, name="legacy-result-v1", kind="result"
        )
        self.assertEqual(result_entry.kind, "result")

    def test_list_filters_by_kind_tag_and_casefolded_query(self):
        self.add_suite()
        self.add_result()
        self.assertEqual(len(self.library.list()), 2)
        self.assertEqual(
            [entry.name for entry in self.library.list(kind="suite")],
            ["support-v1"],
        )
        self.assertEqual(
            [entry.name for entry in self.library.list(tag="PRODUCTION")],
            ["support-result-v1", "support-v1"],
        )
        self.assertEqual(
            [entry.name for entry in self.library.list(query="BASE")],
            ["support-result-v1"],
        )

    def test_same_add_is_idempotent_but_names_are_immutable(self):
        first = self.add_suite()
        second = self.add_suite()
        self.assertEqual(first, second)
        changed_path = self.base / "changed.json"
        write_json(changed_path, suite_payload(private_marker="changed prompt"))
        with self.assertRaisesRegex(ValueError, "already identifies different"):
            self.library.add(
                changed_path,
                name=first.name,
                tags=["production", "support"],
            )
        with self.assertRaisesRegex(ValueError, "already identifies different"):
            self.library.add(
                self.suite_path,
                name=first.name,
                tags=["different"],
            )

    def test_name_tag_and_kind_validation_rejects_ambiguous_inputs(self):
        invalid_names = ("", "@suite", "../suite", "bad name", "a/" "b", "a" * 65)
        for name in invalid_names:
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.library.add(self.suite_path, name=name)
        with self.assertRaises(ValueError):
            self.library.add(self.suite_path, name="valid", tags=["bad tag"])
        with self.assertRaisesRegex(ValueError, "at most 12"):
            self.library.add(
                self.suite_path,
                name="valid",
                tags=[f"tag-{index}" for index in range(13)],
            )
        with self.assertRaisesRegex(ValueError, "not the requested result"):
            self.library.add(self.suite_path, name="valid", kind="result")

    def test_malformed_non_utf8_non_object_and_unsupported_results_are_rejected(self):
        fixtures = {
            "empty.json": b"",
            "invalid.json": b"{not-json",
            "binary.json": b"\xff\xfe",
            "array.json": b"[]",
            "duplicate.json": b'{"schema":"evalt-suite-v2","schema":"other"}',
            "nonfinite.json": b'{"winner":{"model":"x","holdout_pass_rate":NaN}}',
        }
        for filename, data in fixtures.items():
            path = self.base / filename
            path.write_bytes(data)
            with self.subTest(filename=filename), self.assertRaises(ValueError):
                self.library.add(path, name=f"fixture-{filename[:-5]}")

        unknown = self.base / "unknown.json"
        write_json(unknown, {"hello": "world"})
        with self.assertRaisesRegex(ValueError, "exported Evalt"):
            self.library.add(unknown, name="unknown")
        unsupported = self.base / "unsupported.json"
        write_json(
            unsupported,
            result_payload(schema="evalt-future-result-v99"),
        )
        with self.assertRaisesRegex(ValueError, "supported Evalt result schema"):
            self.library.add(unsupported, name="unsupported")

        oversized = self.base / "oversized.json"
        with oversized.open("wb") as handle:
            handle.seek(128 * 1024 * 1024)
            handle.write(b"}")
        with self.assertRaisesRegex(ValueError, "at most 128 MiB"):
            self.library.add(oversized, name="oversized")

    def test_index_and_objects_are_integrity_checked_on_every_resolution(self):
        entry = self.add_suite()
        index = json.loads(self.library.index_path.read_text(encoding="utf-8"))
        index["entries"][0]["summary"]["examples"] = 999
        self.library.index_path.write_text(json.dumps(index), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "modified outside Evalt"):
            self.library.entry(entry.name)

        # Restore a valid catalog, then prove the object check is independent.
        self.root.replace(self.base / "corrupt-index")
        self.library = EvidenceLibrary(self.root)
        entry = self.add_suite()
        self.library.resolve(entry.name).write_bytes(b"{}")
        with self.assertRaisesRegex(ValueError, "SHA-256 integrity"):
            self.library.resolve(entry.name)

    def test_missing_object_is_reported_clearly(self):
        entry = self.add_suite()
        self.library.resolve(entry.name).unlink()
        with self.assertRaisesRegex(ValueError, "is missing"):
            self.library.read(entry.name)

    def test_export_is_exact_atomic_and_never_overwrites_by_default(self):
        entry = self.add_result()
        output = self.base / "exports" / "baseline.json"
        exported = self.library.export(entry.name, output)
        self.assertEqual(exported.read_bytes(), self.result_bytes)
        self.assertEqual(self.library.export(entry.name, output), output)
        output.write_text("different", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            self.library.export(entry.name, output)
        self.library.export(entry.name, output, force=True)
        self.assertEqual(output.read_bytes(), self.result_bytes)
        with self.assertRaises(IsADirectoryError):
            self.library.export(entry.name, self.base, force=True)
        self.assertFalse(list(output.parent.glob(".*.tmp")))

    def test_environment_default_and_reference_resolution_are_explicit(self):
        with mock.patch.dict(
            os.environ, {"EVALT_LIBRARY_HOME": str(self.root)}, clear=False
        ):
            library = EvidenceLibrary()
            library.add(self.suite_path, name="support-v1")
            resolved = resolve_evidence_reference(
                "@support-v1", expected_kind="suite"
            )
        self.assertEqual(resolved.read_bytes(), self.suite_bytes)
        normal = resolve_evidence_reference("relative/suite.json", root=self.root)
        self.assertEqual(normal, Path("relative/suite.json"))
        with self.assertRaisesRegex(ValueError, "name after @"):
            resolve_evidence_reference("@", root=self.root)
        with self.assertRaisesRegex(ValueError, "not a result"):
            resolve_evidence_reference(
                "@support-v1", root=self.root, expected_kind="result"
            )

    def test_concurrent_writers_preserve_every_entry(self):
        def add(index: int):
            return EvidenceLibrary(self.root).add(
                self.suite_path,
                name=f"suite-{index:02d}",
                tags=["concurrent"],
            )

        with ThreadPoolExecutor(max_workers=8) as pool:
            entries = list(pool.map(add, range(20)))
        self.assertEqual(len(entries), 20)
        self.assertEqual(len(self.library.list()), 20)
        self.assertEqual(
            {entry.name for entry in self.library.list()},
            {f"suite-{index:02d}" for index in range(20)},
        )
        json.loads(self.library.index_path.read_text(encoding="utf-8"))


class EvidenceLibraryCliTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.base = Path(self.temporary.name)
        self.root = self.base / "local library"
        self.suite_path = self.base / "suite.json"
        self.result_path = self.base / "result.json"
        self.suite_bytes = write_json(self.suite_path, suite_payload())
        self.result_bytes = write_json(self.result_path, result_payload())

    def tearDown(self):
        self.temporary.cleanup()

    def test_cli_add_list_show_resolve_and_export_are_private_and_offline(self):
        code, output, error = run_cli(
            "library",
            "add",
            str(self.suite_path),
            "--name",
            "support-v1",
            "--tag",
            "production",
            "--root",
            str(self.root),
        )
        self.assertEqual((code, error), (0, ""))
        payload = json.loads(output)
        self.assertFalse(payload["provider_call_started"])
        self.assertFalse(payload["dashboard_sync_started"])
        self.assertNotIn("PRIVATE PROMPT", output)
        self.assertNotIn("ticket 0", output)

        for command in ("list", "show"):
            args = ["library", command]
            if command == "show":
                args.append("support-v1")
            args.extend(["--root", str(self.root)])
            code, output, error = run_cli(*args)
            self.assertEqual((code, error), (0, ""))
            self.assertNotIn("PRIVATE PROMPT", output)
            self.assertNotIn("ticket 0", output)
            payload = json.loads(output)
            self.assertFalse(payload["dashboard_sync_started"])

        code, output, error = run_cli(
            "library", "resolve", "support-v1", "--root", str(self.root)
        )
        self.assertEqual((code, error), (0, ""))
        self.assertEqual(Path(output.strip()).read_bytes(), self.suite_bytes)

        export_path = self.base / "portable" / "suite.json"
        code, output, error = run_cli(
            "library",
            "export",
            "support-v1",
            "--output",
            str(export_path),
            "--root",
            str(self.root),
        )
        self.assertEqual((code, error), (0, ""))
        self.assertEqual(export_path.read_bytes(), self.suite_bytes)
        self.assertFalse(json.loads(output)["dashboard_sync_started"])

    def test_at_name_integrates_with_validate_check_compare_and_report(self):
        library = EvidenceLibrary(self.root)
        library.add(self.suite_path, name="support-v1")
        library.add(self.result_path, name="baseline-v1")
        candidate_path = self.base / "candidate.json"
        candidate = result_payload(model="fixture/model-b")
        write_json(candidate_path, candidate)
        library.add(candidate_path, name="candidate-v2")

        code, output, error = run_cli(
            "validate",
            "@support-v1",
            "--library-root",
            str(self.root),
        )
        self.assertEqual((code, error), (0, ""))
        self.assertTrue(json.loads(output)["valid"])

        code, output, error = run_cli(
            "check",
            "@baseline-v1",
            "--library-root",
            str(self.root),
            "--min-pass-rate",
            "0.8",
            "--json",
        )
        self.assertEqual((code, error), (0, ""))
        self.assertTrue(json.loads(output)["passed"])

        comparison_path = self.base / "comparison.json"
        code, output, error = run_cli(
            "compare",
            "@baseline-v1",
            "@candidate-v2",
            "--library-root",
            str(self.root),
            "--output",
            str(comparison_path),
        )
        self.assertEqual((code, error), (0, ""))
        self.assertTrue(json.loads(output)["comparison"]["comparable_contract"])
        self.assertTrue(comparison_path.exists())

        html_path = self.base / "report.html"
        code, output, error = run_cli(
            "report",
            "@baseline-v1",
            "--library-root",
            str(self.root),
            "--html",
            str(html_path),
        )
        self.assertEqual((code, error), (0, ""))
        self.assertTrue(html_path.exists())
        self.assertIn("fixture/model-a", html_path.read_text(encoding="utf-8"))

    def test_cli_failures_are_unambiguous_and_do_not_create_a_catalog(self):
        code, output, error = run_cli(
            "validate",
            "@missing",
            "--library-root",
            str(self.root),
        )
        self.assertEqual(code, 2)
        self.assertEqual(output, "")
        self.assertIn("no evidence named", error)
        self.assertFalse(self.root.exists())


if __name__ == "__main__":
    unittest.main()
