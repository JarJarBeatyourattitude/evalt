"""Compatibility command-line entry point for the pre-release lgp command."""

from __future__ import annotations

import argparse
import json
import sys

from .core import BudgetExceeded, Client, ProviderError


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="lgp", description="Test prompts against approved answers.")
    commands = root.add_subparsers(dest="command", required=True)
    draft = commands.add_parser("draft", help="Generate the first answer for approval or correction.")
    draft.add_argument("--task", required=True)
    draft.add_argument("--input", required=True)
    draft.add_argument("--model", default="openai/gpt-5-mini")
    draft.add_argument("--max-cost", type=float, default=0.10)
    optimize = commands.add_parser("optimize", help="Run a bounded prompt/model tournament.")
    optimize.add_argument("job")
    optimize.add_argument("--output", default="last-good-prompt-result.json")
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        client = Client()
        if args.command == "draft":
            draft = client.draft_answer(
                task=args.task,
                input=args.input,
                model=args.model,
                max_cost_usd=args.max_cost,
            )
            print(json.dumps({
                "task": draft.task,
                "input": draft.input,
                "answer": draft.answer,
                "model": draft.model,
                "provider_cost_usd": draft.provider_cost_usd,
                "next": "Approve this answer or replace it with the answer you wanted.",
            }, indent=2, ensure_ascii=False))
            return 0
        with open(args.job, encoding="utf-8") as handle:
            job = json.load(handle)
        result = client.optimize(
            prompt=job["prompt"],
            examples=job["examples"],
            models=job["models"],
            optimizer_model=job.get("optimizer_model", "openai/gpt-5.6-luna"),
            evaluator_model=job.get("evaluator_model", "openai/gpt-5.6-luna"),
            objective=job.get("objective", "cheapest_passing"),
            quality_threshold=float(job.get("quality_threshold", 0.95)),
            max_optimization_cost_usd=float(job.get("max_optimization_cost_usd", 2.00)),
            rounds=int(job.get("rounds", 3)),
            minimum_meaningful_quality_gain=float(job.get("minimum_meaningful_quality_gain", 0.03)),
        )
        result.save(args.output)
        print(json.dumps({
            "winner_model": result.winner.model,
            "winner_prompt": result.winner.selected_prompt,
            "holdout_pass_rate": result.winner.holdout_pass_rate,
            "estimated_cost_per_successful_call_usd": result.winner.estimated_cost_per_successful_call_usd,
            "optimization_spend_usd": result.total_provider_spend_usd,
            "exploratory": result.exploratory,
            "quality_frontier": result.quality_frontier,
            "diminishing_returns": result.diminishing_returns,
            "unavailable_models": result.unavailable_models,
            "result": args.output,
        }, indent=2, ensure_ascii=False))
        return 0
    except (BudgetExceeded, ProviderError, ValueError, KeyError, OSError, json.JSONDecodeError) as error:
        print(f"lgp (Evalt compatibility command): {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
