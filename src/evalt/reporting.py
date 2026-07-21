"""Portable, offline reports for Evalt optimization results."""

from __future__ import annotations

from collections.abc import Mapping
from html import escape
import json
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET


def _number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _winner(result: Mapping[str, Any]) -> Mapping[str, Any]:
    winner = result.get("winner") or result.get("selected") or result.get("report") or result
    if isinstance(winner, Mapping) and isinstance(winner.get("selected"), Mapping):
        return winner["selected"]
    return winner if isinstance(winner, Mapping) else {}


def _models(result: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    models = result.get("models") or result.get("frontier", {}).get("entries") or []
    return [item for item in models if isinstance(item, Mapping)]


def _final_cases(winner: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {
        str(item.get("example_id") or f"case-{index + 1}"): item
        for index, item in enumerate(winner.get("cases") or [])
        if isinstance(item, Mapping)
        and str(item.get("split")) in {"holdout", "final_test", "test"}
    }


def compare_results(
    baseline: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, Any]:
    """Compare two saved runs case by case without making a provider call."""

    baseline_winner = _winner(baseline)
    candidate_winner = _winner(candidate)
    baseline_cases = _final_cases(baseline_winner)
    candidate_cases = _final_cases(candidate_winner)
    baseline_hash = str((baseline.get("regression_suite") or {}).get("suite_hash") or "")
    candidate_hash = str((candidate.get("regression_suite") or {}).get("suite_hash") or "")
    comparable_contract = bool(baseline_hash and candidate_hash and baseline_hash == candidate_hash)
    case_rows: list[dict[str, Any]] = []
    summary = {
        "improvements": 0,
        "regressions": 0,
        "unchanged_passes": 0,
        "unchanged_failures": 0,
        "added": 0,
        "missing": 0,
    }
    for example_id in sorted(set(baseline_cases) | set(candidate_cases)):
        before = baseline_cases.get(example_id)
        after = candidate_cases.get(example_id)
        if before is None:
            status = "added"
        elif after is None:
            status = "missing"
        elif bool(before.get("passed")) and not bool(after.get("passed")):
            status = "regression"
        elif not bool(before.get("passed")) and bool(after.get("passed")):
            status = "improvement"
        elif bool(after.get("passed")):
            status = "unchanged_pass"
        else:
            status = "unchanged_failure"
        summary_key = {
            "added": "added",
            "missing": "missing",
            "regression": "regressions",
            "improvement": "improvements",
            "unchanged_pass": "unchanged_passes",
            "unchanged_failure": "unchanged_failures",
        }[status]
        summary[summary_key] += 1
        case_rows.append({
            "example_id": example_id,
            "status": status,
            "baseline_passed": None if before is None else bool(before.get("passed")),
            "candidate_passed": None if after is None else bool(after.get("passed")),
            "baseline_score": None if before is None else _number(before.get("score")),
            "candidate_score": None if after is None else _number(after.get("score")),
            "baseline_output": None if before is None else before.get("output"),
            "candidate_output": None if after is None else after.get("output"),
            "approved_output": (after or before or {}).get("approved_output"),
        })

    baseline_quality = _number(
        baseline_winner.get("holdout_pass_rate") or baseline_winner.get("pass_rate")
    )
    candidate_quality = _number(
        candidate_winner.get("holdout_pass_rate") or candidate_winner.get("pass_rate")
    )
    baseline_cost = baseline_winner.get("estimated_cost_per_successful_call_usd")
    candidate_cost = candidate_winner.get("estimated_cost_per_successful_call_usd")
    baseline_p90 = _number(baseline_winner.get("target_latency_p90_ms"))
    candidate_p90 = _number(candidate_winner.get("target_latency_p90_ms"))
    return {
        "schema": "evalt-comparison-v1",
        "comparable_contract": comparable_contract,
        "contract": {
            "baseline_suite_hash": baseline_hash or None,
            "candidate_suite_hash": candidate_hash or None,
            "warning": None if comparable_contract else (
                "The suite hashes differ or are missing. Case deltas are descriptive and must not be used as a promotion gate."
            ),
        },
        "baseline": {
            "model": baseline_winner.get("model"),
            "prompt": baseline_winner.get("selected_prompt"),
            "quality": baseline_quality,
            "cost_per_1k_successful_calls_usd": None if baseline_cost is None else _number(baseline_cost) * 1000,
            "p90_latency_ms": baseline_p90,
        },
        "candidate": {
            "model": candidate_winner.get("model"),
            "prompt": candidate_winner.get("selected_prompt"),
            "quality": candidate_quality,
            "cost_per_1k_successful_calls_usd": None if candidate_cost is None else _number(candidate_cost) * 1000,
            "p90_latency_ms": candidate_p90,
        },
        "delta": {
            "quality_percentage_points": (candidate_quality - baseline_quality) * 100,
            "cost_per_1k_successful_calls_usd": None if baseline_cost is None or candidate_cost is None else (_number(candidate_cost) - _number(baseline_cost)) * 1000,
            "p90_latency_ms": candidate_p90 - baseline_p90,
            "prompt_changed": baseline_winner.get("selected_prompt") != candidate_winner.get("selected_prompt"),
            "model_changed": baseline_winner.get("model") != candidate_winner.get("model"),
        },
        "case_summary": summary,
        "cases": case_rows,
    }


def render_comparison_html(
    comparison: Mapping[str, Any], *, title: str = "Evalt comparison"
) -> str:
    """Render a self-contained case-level comparison report."""

    before = comparison.get("baseline") or {}
    after = comparison.get("candidate") or {}
    delta = comparison.get("delta") or {}
    summary = comparison.get("case_summary") or {}
    rows = []
    for item in comparison.get("cases") or []:
        if not isinstance(item, Mapping):
            continue
        status = str(item.get("status") or "unknown")
        rows.append(
            "<tr class='{}'><td><strong>{}</strong></td><td>{}</td><td>{}</td><td>{}</td></tr>".format(
                escape(status), escape(str(item.get("example_id") or "case")),
                escape("—" if item.get("baseline_passed") is None else "Pass" if item.get("baseline_passed") else "Fail"),
                escape("—" if item.get("candidate_passed") is None else "Pass" if item.get("candidate_passed") else "Fail"),
                escape(status.replace("_", " ").title()),
            )
        )
    warning = str((comparison.get("contract") or {}).get("warning") or "")
    cost_delta = delta.get("cost_per_1k_successful_calls_usd")
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{escape(title)}</title><style>
:root{{--ink:#18151e;--muted:#706a76;--line:#ddd8d0;--paper:#f5f2ec;--card:#fff;--violet:#6756d2;--green:#23885b;--red:#b65546}}*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);font:14px/1.5 Inter,ui-sans-serif,system-ui,sans-serif}}main{{width:min(1040px,calc(100% - 32px));margin:auto;padding:56px 0 80px}}.kicker{{color:var(--violet);font-size:10px;font-weight:800;letter-spacing:.12em;text-transform:uppercase}}h1{{margin:8px 0 24px;font-size:clamp(36px,6vw,64px);letter-spacing:-.055em;line-height:1}}.routes,.metrics{{display:grid;gap:12px}}.routes{{grid-template-columns:1fr 1fr}}.route,.metric,.warning{{padding:18px;border:1px solid var(--line);border-radius:12px;background:var(--card)}}.route small,.metric small{{display:block;color:var(--muted);font-size:9px;font-weight:800;letter-spacing:.08em;text-transform:uppercase}}.route strong{{display:block;margin:5px 0;font-size:18px}}.metrics{{grid-template-columns:repeat(4,1fr);margin:12px 0 30px}}.metric strong{{display:block;margin-top:6px;font-size:17px}}table{{width:100%;border-collapse:collapse;background:var(--card);border:1px solid var(--line)}}th,td{{padding:12px 14px;border-bottom:1px solid var(--line);text-align:left}}th{{color:var(--muted);font-size:9px;text-transform:uppercase;letter-spacing:.08em}}tr.regression td:last-child{{color:var(--red);font-weight:800}}tr.improvement td:last-child{{color:var(--green);font-weight:800}}.warning{{margin-bottom:18px;border-color:#dfb783;background:#fff8ed}}@media(max-width:720px){{.routes,.metrics{{grid-template-columns:1fr 1fr}}th:nth-child(2),td:nth-child(2){{display:none}}}}
</style></head><body><main><span class="kicker">Frozen-result diff</span><h1>{escape(title)}</h1>{f'<div class="warning">{escape(warning)}</div>' if warning else ''}<div class="routes"><article class="route"><small>Baseline</small><strong>{escape(str(before.get('model') or 'Unknown'))}</strong><span>{_number(before.get('quality')):.1%} final-test quality</span></article><article class="route"><small>Candidate</small><strong>{escape(str(after.get('model') or 'Unknown'))}</strong><span>{_number(after.get('quality')):.1%} final-test quality</span></article></div><div class="metrics"><div class="metric"><small>Quality delta</small><strong>{_number(delta.get('quality_percentage_points')):+.1f} pp</strong></div><div class="metric"><small>Cost / 1K delta</small><strong>{'—' if cost_delta is None else f'${_number(cost_delta):+.4f}'}</strong></div><div class="metric"><small>Improvements</small><strong>{int(summary.get('improvements') or 0)}</strong></div><div class="metric"><small>Regressions</small><strong>{int(summary.get('regressions') or 0)}</strong></div></div><table><thead><tr><th>Final-test case</th><th>Baseline</th><th>Candidate</th><th>Change</th></tr></thead><tbody>{''.join(rows) or '<tr><td colspan="4">No comparable final-test cases were stored.</td></tr>'}</tbody></table></main></body></html>"""


def render_html_report(result: Mapping[str, Any], *, title: str = "Evalt evaluation report") -> str:
    """Render one self-contained, escaped HTML report without provider calls."""

    winner = _winner(result)
    quality = _number(winner.get("holdout_pass_rate") or winner.get("pass_rate"))
    threshold = _number(result.get("quality_threshold"), 0.95)
    cost = winner.get("estimated_cost_per_successful_call_usd")
    model = str(winner.get("model") or winner.get("label") or "No selected route")
    status = str(result.get("quality_gate_status") or ("QUALIFIED" if quality >= threshold else "BELOW_GATE"))
    spend = _number(result.get("total_provider_spend_usd") or result.get("provider_spend_usd"))
    scope = str(result.get("winner_scope") or "")
    rows = []
    for item in sorted(_models(result), key=lambda entry: _number(entry.get("estimated_cost_per_successful_call_usd"), float("inf"))):
        item_quality = _number(item.get("holdout_pass_rate") or item.get("pass_rate"))
        item_cost = item.get("estimated_cost_per_successful_call_usd")
        item_model = str(item.get("model") or item.get("label") or "Unknown")
        decision = "Selected" if item_model == model else "Passed" if item_quality >= threshold else "Below gate"
        rows.append(
            "<tr><td><strong>{}</strong></td><td>{:.1%}</td><td>{}</td><td><span class='decision {}'>{}</span></td></tr>".format(
                escape(item_model), item_quality,
                "—" if item_cost is None else f"${_number(item_cost) * 1000:.4f}",
                "pass" if decision in {"Selected", "Passed"} else "fail", decision,
            )
        )
    cases = []
    for case in winner.get("cases") or []:
        if not isinstance(case, Mapping) or str(case.get("split")) not in {"holdout", "final_test", "test"}:
            continue
        passed = bool(case.get("passed"))
        cases.append(
            "<article class='case {}'><header><strong>{}</strong><span>{}</span></header><p>{}</p><details><summary>Output and expected result</summary><div><small>Output</small><pre>{}</pre><small>Expected</small><pre>{}</pre></div></details></article>".format(
                "pass" if passed else "fail", escape(str(case.get("example_id") or "case")),
                "Passed" if passed else "Failed", escape(str(case.get("reason") or "No reason recorded")),
                escape(str(case.get("output") or "")), escape(str(case.get("approved_output") or "")),
            )
        )
    model_table = "".join(rows) or "<tr><td colspan='4'>No model comparison was stored in this result.</td></tr>"
    case_cards = "".join(cases) or "<p class='empty'>No final-test case rows were stored in this result.</p>"
    cost_label = "—" if cost is None else f"${_number(cost) * 1000:.4f} / 1K successful calls"
    warnings = "".join(f"<li>{escape(str(item))}</li>" for item in result.get("warnings") or [])
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{escape(title)}</title><style>
:root{{--ink:#18151e;--muted:#706a76;--line:#ddd8d0;--paper:#f5f2ec;--card:#fff;--violet:#6756d2;--green:#23885b;--red:#b65546}}*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);font:14px/1.55 Inter,ui-sans-serif,system-ui,sans-serif}}main{{width:min(1040px,calc(100% - 32px));margin:0 auto;padding:56px 0 80px}}.kicker{{color:var(--violet);font-size:10px;font-weight:800;letter-spacing:.12em;text-transform:uppercase}}h1{{margin:8px 0 10px;font-size:clamp(36px,6vw,68px);letter-spacing:-.06em;line-height:.98}}.lead{{max-width:720px;color:var(--muted)}}.metrics{{margin:34px 0 20px;display:grid;grid-template-columns:repeat(4,1fr);overflow:hidden;border:1px solid var(--line);border-radius:14px;background:var(--card)}}.metrics div{{padding:20px;border-right:1px solid var(--line)}}.metrics div:last-child{{border:0}}small{{display:block;color:var(--muted);font-size:9px;font-weight:750;letter-spacing:.08em;text-transform:uppercase}}.metrics strong{{display:block;margin-top:7px;font-size:16px}}section{{margin-top:28px}}h2{{font-size:24px;letter-spacing:-.035em}}table{{width:100%;border-collapse:collapse;overflow:hidden;border:1px solid var(--line);border-radius:12px;background:var(--card)}}th,td{{padding:13px 16px;border-bottom:1px solid var(--line);text-align:left}}th{{color:var(--muted);font-size:9px;letter-spacing:.08em;text-transform:uppercase}}tr:last-child td{{border:0}}.decision{{font-size:10px;font-weight:800}}.decision.pass{{color:var(--green)}}.decision.fail{{color:var(--red)}}.case{{margin:9px 0;padding:16px;border:1px solid var(--line);border-left:4px solid var(--green);border-radius:10px;background:var(--card)}}.case.fail{{border-left-color:var(--red)}}.case header{{display:flex;justify-content:space-between}}.case header span{{font-size:10px;font-weight:800;color:var(--green)}}.case.fail header span{{color:var(--red)}}.case p,.empty{{color:var(--muted)}}details summary{{cursor:pointer;color:var(--violet);font-size:11px;font-weight:750}}pre{{overflow:auto;padding:12px;border-radius:8px;background:#1d1a22;color:#f3f0f6;font:11px/1.5 ui-monospace,monospace;white-space:pre-wrap}}.warnings{{padding:16px 20px;border:1px solid #e3c9a7;border-radius:10px;background:#fff9f0}}footer{{margin-top:36px;padding-top:16px;border-top:1px solid var(--line);color:var(--muted);font-size:10px}}@media(max-width:700px){{.metrics{{grid-template-columns:repeat(2,1fr)}}.metrics div:nth-child(2){{border-right:0}}th:nth-child(3),td:nth-child(3){{display:none}}}}
</style></head><body><main><span class="kicker">Portable Evalt result</span><h1>{escape(model)}</h1><p class="lead">{escape(status.replace('_', ' ').title())}. This report describes one frozen evaluation result; it is not a general claim about the model.</p>
<div class="metrics"><div><small>Final-test quality</small><strong>{quality:.1%}</strong></div><div><small>Required</small><strong>{threshold:.1%}</strong></div><div><small>Production cost</small><strong>{escape(cost_label)}</strong></div><div><small>Test spend</small><strong>${spend:.4f}</strong></div></div>
<section><span class="kicker">Measured frontier</span><h2>Configurations</h2><table><thead><tr><th>Configuration</th><th>Final test</th><th>Cost / 1K</th><th>Decision</th></tr></thead><tbody>{model_table}</tbody></table></section>
<section><span class="kicker">Frozen evidence</span><h2>Final-test cases</h2>{case_cards}</section>
{f'<section class="warnings"><strong>Warnings</strong><ul>{warnings}</ul></section>' if warnings else ''}
<footer>{escape(scope)} · Generated offline from the saved Evalt JSON result.</footer></main></body></html>"""


def render_junit_report(result: Mapping[str, Any], *, suite_name: str = "evalt") -> str:
    """Render final-test cases as portable JUnit XML for CI systems."""

    winner = _winner(result)
    final_cases = [
        item for item in winner.get("cases") or []
        if isinstance(item, Mapping) and str(item.get("split")) in {"holdout", "final_test", "test"}
    ]
    failures = sum(1 for item in final_cases if not bool(item.get("passed")))
    testsuite = ET.Element("testsuite", {
        "name": suite_name,
        "tests": str(len(final_cases) or 1),
        "failures": str(failures + (0 if final_cases else 1)),
        "errors": "0",
        "time": str(_number(result.get("elapsed_seconds"))),
    })
    properties = ET.SubElement(testsuite, "properties")
    for name, value in {
        "winner_model": winner.get("model", ""),
        "quality_threshold": result.get("quality_threshold", ""),
        "winner_scope": result.get("winner_scope", ""),
        "provider_spend_usd": result.get("total_provider_spend_usd", ""),
        "suite_hash": (result.get("regression_suite") or {}).get("suite_hash", ""),
    }.items():
        ET.SubElement(properties, "property", {"name": name, "value": str(value)})
    if not final_cases:
        test = ET.SubElement(testsuite, "testcase", {"name": "final-test-evidence", "classname": suite_name})
        ET.SubElement(test, "failure", {"message": "No final-test case rows were stored"}).text = "The result cannot provide case-level CI evidence."
    for item in final_cases:
        test = ET.SubElement(testsuite, "testcase", {
            "name": str(item.get("example_id") or "case"),
            "classname": f"{suite_name}.{item.get('difficulty') or 'typical'}",
            "time": f"{_number(item.get('target_latency_ms')) / 1000:.6f}",
        })
        if not bool(item.get("passed")):
            failure = ET.SubElement(test, "failure", {"message": str(item.get("reason") or "Evaluation failed")})
            failure.text = json.dumps({"output": item.get("output"), "expected": item.get("approved_output")}, ensure_ascii=False)
        ET.SubElement(test, "system-out").text = str(item.get("output") or "")
    ET.indent(testsuite, space="  ")
    return ET.tostring(testsuite, encoding="unicode", xml_declaration=True) + "\n"


def write_reports(result: Mapping[str, Any], *, html_path: str | None = None, junit_path: str | None = None, title: str = "Evalt evaluation report") -> dict[str, str]:
    """Write requested reports and return their paths."""

    written: dict[str, str] = {}
    if html_path:
        target = Path(html_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(render_html_report(result, title=title), encoding="utf-8")
        written["html"] = str(target)
    if junit_path:
        target = Path(junit_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(render_junit_report(result, suite_name=title), encoding="utf-8")
        written["junit"] = str(target)
    if not written:
        raise ValueError("Choose at least one report path: html_path or junit_path.")
    return written


__all__ = [
    "compare_results",
    "render_comparison_html",
    "render_html_report",
    "render_junit_report",
    "write_reports",
]
