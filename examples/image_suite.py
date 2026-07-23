"""Run a two-model image evaluation from a human-labeled JSON manifest.

Manifest format:
[
  {"image": "./photos/package-01.png", "label": "intact"},
  {"image": "./photos/package-02.png", "label": "damaged"}
]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evalt import Evalt, Example, ImageInput, Suite, multimodal_input


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path, help="JSON array of image paths and approved labels")
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    records = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(records, list) or len(records) < 2:
        raise ValueError("The manifest must contain at least two human-labeled images.")

    instruction = "Return exactly one label: damaged or intact."
    examples = []
    for index, record in enumerate(records, start=1):
        image_path = (manifest_path.parent / record["image"]).resolve()
        label = str(record["label"]).strip().lower()
        if label not in {"damaged", "intact"}:
            raise ValueError(f"Case {index} has an unsupported label: {label!r}")
        examples.append(
            Example(
                multimodal_input(instruction, ImageInput.from_path(image_path)),
                approved_output=label,
                id=f"package-{index}",
            )
        )

    suite = Suite(
        name="package-condition",
        prompt="Inspect the package image. Return exactly one label: damaged or intact.",
        examples=tuple(examples),
        models=("google/gemini-2.5-flash-lite", "openai/gpt-5.4-nano"),
        evaluator={"type": "exact_text"},
        optimize_prompt=False,
        max_optimization_cost_usd=0.25,
    )
    result = Evalt(show_progress=True).run(suite)
    print(json.dumps({"model": result.winner.model, "pass_rate": result.winner.holdout_pass_rate}))


if __name__ == "__main__":
    main()
