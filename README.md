# Evalt Python SDK

[![PyPI](https://img.shields.io/pypi/v/evalt?label=PyPI&logo=pypi&logoColor=white)](https://pypi.org/project/evalt/)
[![Python](https://img.shields.io/pypi/pyversions/evalt?logo=python&logoColor=white)](https://pypi.org/project/evalt/)
[![CI](https://github.com/JarJarBeatyourattitude/evalt/actions/workflows/ci.yml/badge.svg)](https://github.com/JarJarBeatyourattitude/evalt/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-2f7a65.svg)](LICENSE)

Evalt is a durable runtime router. Your application gives it a prompt, an input, and a
stable route name. Evalt finds the lowest-cost prompt/model/reasoning configuration that
clears the approved validation target (95% by default), records the decision in SQLite,
and uses a separate capped test budget when traffic, a new model, or a provider-price
change makes retesting worthwhile.

## Install

```bash
python -m pip install evalt
```

Migrating from OpenAI Evals result JSONL? Evalt can recover only the reviewable
input/reference pairs, offline, and reports everything it cannot honestly reconstruct:

```bash
evalt import-openai-results results.jsonl --prompt-file system-prompt.txt --output evalt.json
```

Read the [OpenAI Evals migration guide](docs/openai-evals-migration.md) before running
an imported suite. Historical candidate outputs are never treated as approved answers.

For an offline or pinned artifact install, use the verified versioned wheel from this
repository checkout:

```bash
python -m venv .venv
python -m pip install dist/evalt-0.8.13-py3-none-any.whl
evalt --version
```

`evalt` is the primary import and command. `modelsieve`, `last_good_prompt`, and `lgp`
remain compatibility aliases for existing integrations.

The SDK is MIT licensed. Hosted Evalt Pro is a separate managed service; using the free
package never creates a platform charge.

- [Package on PyPI](https://pypi.org/project/evalt/)
- [Source and issues on GitHub](https://github.com/JarJarBeatyourattitude/evalt)
- [Hosted documentation](https://evalt.dev/docs/)

## Production API

```python
from evalt import Evalt

evalt = Evalt(api_key=OPENROUTER_API_KEY)
answer = evalt.run(
    "Classify this support request as billing, account, or technical.",
    ticket,
    route="support-routing",
    test_budget_usd="auto",
)

send(answer.content)
answer.accept()                 # or answer.correct("billing")
print(evalt.route_status("support-routing"))
```

Use one stable route name per production task. A single `Evalt` instance can manage any
number of independent routes; each route keeps its own prompt, approved or corrected
examples, selected model, price ceiling, test budget, and maintenance history:

```python
support = evalt.run(support_prompt, ticket, route="support-routing")
summary = evalt.run(summary_prompt, transcript, route="call-summary")
fraud = evalt.run(risk_prompt, transaction, route="fraud-review")
```

Feedback on `support-routing` cannot enter the evaluation set for `call-summary` or
`fraud-review`. Reusing a route name means “this is the same repeated task”; using a new
name creates a separate optimization track in the same local state database.

The first call uses the selected bootstrap route within the production price ceiling.
Explicit feedback becomes the route-specific evaluation set. On a later call, Evalt can
launch a background maintenance run after the configured evidence/traffic threshold is
met. Automatic test spend is bounded by `max_test_budget_usd` (USD 1 by default); a
retest never spends from an unlimited hidden allowance.

The focused default is `objective="lowest_cost_at_accuracy"` with
`target_accuracy=0.95`: Evalt promotes the cheapest tested configuration that clears that
approved bar. A price-first frontier and an incumbent-preservation migration mode remain
available:

```python
answer = evalt.run(
    prompt,
    input,
    route="support-routing",
    price_usd=0.05,
    target_accuracy=0.97,
    objective="lowest_cost_at_accuracy",  # or "best_within_price"
    test_budget_usd=0.75,
)
```

`incumbent_model` is optional. Use it with
`objective="match_baseline_at_lowest_cost"` when migrating an existing workflow and you
specifically want the incumbent's measured validation quality to be the bar. New
workflows need no comparison model: their approved validation target is the bar.

Reasoning effort is tested as part of the model configuration only when the current ZDR
endpoint supports it. The adaptive search first runs one configuration across a broad
price/intelligence frontier, then spends the remaining test budget on effort variants for
models in the observed task-capability band. The first request receives large,
reasoning-aware completion headroom: 32,768 tokens without reasoning, 65,536 at low,
98,304 at medium, and up to 131,072 at high effort, always clamped to the provider's
natural output limit and the remaining context. An empty or explicitly truncated
response is retried only when a genuinely larger valid ceiling exists. Production cost
uses measured 90th-percentile successful calls rather than pricing the entire safety
ceiling. The default quality floor is 95%, and
held-out cases are repeated twice before promotion. A measured 100% means every repeated
approved final-test scenario passed on every configured repeat; it is not a guarantee
about every future input. Reports show distinct scenario count and execution count
separately.

Speed can be part of the explicit production contract. Set
`max_p90_latency_seconds` to require a measured tail-latency ceiling, or use the advanced
`latency_value_usd_per_second` value-of-time term. Evalt reports measured p50 and p90 for
every completed route. OpenRouter's provider price/latency/throughput preferences help
search, but cannot replace those frozen-run measurements. The default deadline
is 600 seconds per response, and complex or long-context suites can raise it explicitly
up to 7200 seconds with `request_timeout_seconds` in the suite, with
`Evalt(request_timeout_seconds=...)`, or with `evalt optimize --request-timeout ...`.
The deadline protects against a genuinely hung provider request; the provider spend cap
remains the economic stop condition.

Model roles are selected separately:

- the test designer / prompt improver uses the cheapest catalog model near the top of the
  current intelligence range;
- the judge may use a lower-cost model only after route-specific verdict calibration;
- production targets come from the price/intelligence frontier, then win or lose on the
  route's frozen human-approved cases.

Higher maintenance budgets tighten the intelligence floors and broaden the target field.
Catalog benchmarks shortlist contenders only; they never promote a route without the
task-specific holdout.

## Explicit optimization and CI

```bash
evalt init evalt.json
evalt validate evalt.json
export OPENROUTER_API_KEY="..."
evalt optimize evalt.json --output evalt-result.json
evalt check evalt-result.json --min-pass-rate 0.95
```

`init`, `validate`, and `check` are offline and make no provider calls. `optimize` uses the
single `max_optimization_cost_usd` value in the suite as a hard cap across optimization,
target runs, judging, and every selected model. The command reports partial coverage
instead of calling an unfinished tournament globally best. In a terminal it shows a
live elapsed/active/settled heartbeat and prints score, p90 latency, and spend as each
route finishes. When piped it emits the same progress as JSONL on stderr, while the final
JSON stays on stdout and at the requested output path. Model/scenario concurrency and the
wall-clock timeout for one provider response can be overridden per run.

Use stricter CI gates when needed:

```bash
evalt check evalt-result.json \
  --min-pass-rate 0.95 \
  --max-cost-per-success 0.002 \
  --require-complete-coverage
```

The command exits `0` on pass, `1` when the measured result fails the gate, and `2` for an
invalid file or runtime error.

To inspect the CI contract without a provider call:

```bash
evalt check examples/passing-result.json --min-pass-rate 0.95 --require-complete-coverage
```

## Explicit suite API

```python
from evalt import Evalt, Suite

suite = Suite.load("evalt.json")  # validates without a provider call
result = Evalt().run(suite)

print(result.winner.model)
print(result.winner.selected_prompt)
print(result.winner.holdout_pass_rate)
print(result.winner.estimated_cost_per_successful_call_usd)

result.save("evalt-result.json")
result.save_regression_suite("evalt-regression.json")
```

For lower-level integrations, `Client.optimize(...)` remains available. `Suite` is the
recommended surface because the full evaluation contract stays inspectable, serializable,
and offline-validatable before spend.

## Suite shape

```json
{
  "schema": "evalt-suite-v1",
  "name": "support-routing",
  "prompt": "Classify the support message. Return one route label.",
  "examples": [
    {"id": "billing-1", "input": "I was charged twice", "approved_output": "billing"},
    {"id": "account-1", "input": "My reset link expired", "approved_output": "account"},
    {"id": "technical-1", "input": "The app freezes", "approved_output": "technical"}
  ],
  "models": ["qwen/qwen3.5-9b", "google/gemini-3-flash-preview"],
  "optimizer_model": "openai/gpt-5.6-luna",
  "evaluator_model": "openai/gpt-5.6-luna",
  "evaluator": {"type": "semantic"},
  "quality_threshold": 0.95,
  "max_optimization_cost_usd": 2.0,
  "rounds": 3,
  "max_parallel_models": 16,
  "max_parallel_scenarios": 32,
  "request_timeout_seconds": 600,
  "allow_few_shot": true,
  "max_few_shot_examples": 3
}
```

With the default 20% final-test split, use at least 25 approved scenarios to obtain the
minimum five distinct final-test scenarios for a non-exploratory result. Repeats measure
consistency; they never inflate the distinct scenario count. A scenario may contain
a `turns` array for multi-turn behavior; Evalt keeps the whole conversation in one split
and replays prior assistant context. Few-shot examples can come only from the training
split and are removed while evaluating their own scenario.

For exact outputs, replace the semantic evaluator with a deterministic contract:

```json
"evaluator": {
  "type": "exact_json",
  "required_keys": ["x", "y"],
  "allow_additional_properties": false,
  "normalize_rational_strings": true
}
```

This makes no evaluator-model call. `exact_text` is also available for strict labels.
Independent model lanes run concurrently (eight by default) and each lane evaluates up
to sixteen independent case executions concurrently. Model lanes are configurable up to
16 and case execution concurrency up to 64. Repeated executions are parallel work units;
turns inside one multi-turn scenario remain ordered. Every in-flight estimate is reserved
against the one hard suite budget. A prompt that already scores 100% on the validation
split skips training replay and rewriting and moves directly to the frozen final test.

## Provider and data contract

- The API key is read from `OPENROUTER_API_KEY`; it is never written to a suite or result.
- Every OpenRouter request requires Zero Data Retention and denies provider data collection.
- Current provider pricing is refreshed at least hourly by default before calls; a changed
  price changes the catalog revision and makes the durable route due for a bounded re-test.
  An unpriced or unbounded route fails closed. Set `catalog_ttl_seconds` on
  `OpenRouterTransport` when a different refresh interval is required.
- Exact provider-reported cost is accumulated in the result.
- The SDK adds no platform or usage fee. Hosted BYOK is also free during early access.
  A future paid hosted control plane for shared history, CI, scheduled Model Watch,
  permissions, managed credentials, and support is an evidence-gated hypothesis rather
  than a current subscription offer.

## Compatibility

Evalt follows semantic versioning for the primary `evalt` Python API, CLI command names,
suite schema, result schema, and exit-code contract. Deprecations remain available for
at least one minor release and are documented before removal. The legacy
`modelsieve`/`last_good_prompt` imports and `lgp` command are compatibility shims, not the
recommended surface for new integrations.

An exported result is evidence about its frozen examples and named model versions—not a
general intelligence ranking or a promise about unseen production traffic.
