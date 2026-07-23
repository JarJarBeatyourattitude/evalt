"""Example command scorer implementing case-insensitive exact text."""

from __future__ import annotations

import json
import sys


def main() -> int:
    raw_request = sys.stdin.read()
    if not raw_request.strip():
        print(
            "This is an Evalt command scorer. Register it with CommandScorer or "
            "pass one evalt-custom-score-request-v1 JSON object on stdin.",
            file=sys.stderr,
        )
        return 2
    try:
        request = json.loads(raw_request)
        if (
            not isinstance(request, dict)
            or request.get("schema") != "evalt-custom-score-request-v1"
        ):
            raise ValueError("unsupported request schema")
        actual = str(request["actual_output"]).strip().casefold()
        approved = str(request["approved_output"]).strip().casefold()
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        print(
            "Invalid Evalt custom-score request; expected one "
            "evalt-custom-score-request-v1 JSON object.",
            file=sys.stderr,
        )
        return 2
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
