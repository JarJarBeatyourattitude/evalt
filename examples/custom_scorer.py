"""Example Evalt command scorer: case-insensitive exact text."""

from __future__ import annotations

import json
import sys


request = json.load(sys.stdin)
actual = str(request["actual_output"]).strip().casefold()
approved = str(request["approved_output"]).strip().casefold()
passed = actual == approved
json.dump(
    {
        "passed": passed,
        "score": 1.0 if passed else 0.0,
        "reason": (
            "Case-insensitive text matched."
            if passed
            else "Case-insensitive text differed."
        ),
    },
    sys.stdout,
    separators=(",", ":"),
)
