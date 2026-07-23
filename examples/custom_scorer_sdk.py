"""Run a reviewed suite with an explicitly registered local command scorer."""

from __future__ import annotations

from pathlib import Path
import sys

from evalt import CommandScorer, Evalt, Suite


HERE = Path(__file__).resolve().parent
suite = Suite.load(HERE / "custom-scorer-suite.json")
scorer = CommandScorer(
    "casefold-exact",
    "1.0",
    [sys.executable, str(HERE / "custom_scorer.py")],
    timeout_seconds=5,
)
result = Evalt(custom_scorers={scorer.scorer_id: scorer}).run(suite)
result.save("evalt-result.json")
print(result.winner.model, result.winner.holdout_pass_rate)
