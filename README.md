# Evalt Python SDK

[![PyPI](https://img.shields.io/pypi/v/evalt?label=PyPI&logo=pypi&logoColor=white)](https://pypi.org/project/evalt/)
[![Python](https://img.shields.io/pypi/pyversions/evalt?logo=python&logoColor=white)](https://pypi.org/project/evalt/)
[![CI](https://github.com/JarJarBeatyourattitude/evalt/actions/workflows/ci.yml/badge.svg)](https://github.com/JarJarBeatyourattitude/evalt/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-2f7a65.svg)](LICENSE)

Evalt turns a recurring AI task into a tested production route. On a route's first
call, it can design and calibrate the test, compare prompt/model/reasoning/few-shot
configurations under one hard budget, remember the lowest-cost passing package, and
then answer the real input through that package. Later calls reuse it immediately.

## Install

```bash
pip install evalt
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
python -m pip install dist/evalt-0.9.3-py3-none-any.whl
evalt --version
```

`evalt` is the primary import and command. `modelsieve`, `last_good_prompt`, and `lgp`
remain compatibility aliases for existing integrations.

The SDK is MIT licensed. Hosted Evalt Pro is a separate managed service; using the free
package never creates a platform charge.

- [Package on PyPI](https://pypi.org/project/evalt/)
- [Source and issues on GitHub](https://github.com/JarJarBeatyourattitude/evalt)
- [Hosted documentation](https://evalt.dev/docs/)

## One-call production route

```python
from evalt import Evalt

evalt = Evalt(show_progress=True)
answer = evalt.run(
    "Classify this request. Return exactly one lowercase label: billing, account, or technical.",
    "Please help—the website won't load.",
    task="Route recurring support tickets to billing, account, or technical.",
    route="support-routing",
    target_accuracy=0.95,
    test_budget_usd="auto",
)
print(answer.content)
```

For a new route, that one call visibly:

1. uses a smart designer to create 25 routine, boundary, adversarial, format, and
   multi-turn cases where relevant;
2. calibrates the proposed deterministic or semantic judge on separate labeled checks;
3. searches the original prompt, prompt rewrites, training-only few-shot packages,
   current models, providers, and supported reasoning efforts in parallel;
4. promotes only a non-exploratory configuration clearing the frozen final test;
5. stores the entire prompt/model/reasoning/few-shot package in the local route database;
6. answers the real input through that selected package.

The first production input is shown to the designer only as an unlabeled example of
realistic domain, shape, and length. It is not copied into the suite or treated as a
correct answer.

AI-generated cases and AI judging are labeled `AI_GENERATED_AI_JUDGED`. They are useful
initial evidence, not disguised human ground truth. `answer.accept()` and
`answer.correct(expected)` add real production labels that calibrate and strengthen
future retests. Pass `first_run="bootstrap"` only when you explicitly want one untested
provider call with no tournament. The automatic first test and later retests never
exceed `max_test_budget_usd` (USD 1 by default).

## Optional: review the AI-drafted test first

For high-stakes or tightly specified work, return an unapproved draft before any
tournament runs:

```python
from evalt import Evalt

evalt = Evalt()
draft = evalt.optimize_task(
    task="Route recurring support tickets to billing, account, or technical.",
    prompt="Return exactly one lowercase label: billing, account, or technical.",
    route="support-routing",
    case_control="review",
    workflow_budget_usd=1.00,  # test design + tournament share this cap
)

draft.save("support-routing-draft.json")
for case in draft.examples:
    print(case.id, case.conversation())
if input("Type APPROVE after reviewing every expected output: ").strip() != "APPROVE":
    raise SystemExit("Draft saved; no tournament ran.")

# This call is the explicit approval boundary. Pass edited examples when needed.
suite = draft.approve()
result = evalt.run(suite)
print(result.winner.model, result.winner.holdout_pass_rate)
```

The designer covers routine, complex, adversarial, format, boundary, and multi-turn
cases where relevant, and recommends deterministic or semantic judging. Evalt then
splits the frozen contract, varies prompts and approved few-shot examples, tests current
models and supported reasoning levels in parallel, and promotes only on the untouched
final test. Use 25 or more distinct cases to obtain at least five final-test cases.

For a fast directional result, set `case_control="autopilot"`. That runs the draft and
tournament immediately, but the result is permanently labeled
`AI_GENERATED_AI_JUDGED`; it never masquerades as a human-verified regression contract.
Bring your own examples with `Suite`, or replace `draft.examples` when calling
`draft.approve(...)`, for the hands-on end of the spectrum.

## Feedback and route maintenance

```python
from evalt import Evalt

ticket = "Please help—the website won't load."
expected = "technical"
evalt = Evalt(show_progress=True)
answer = evalt.run(
    "Classify this request. Return exactly one lowercase label: billing, account, or technical.",
    ticket,
    task="Route recurring support tickets to billing, account, or technical.",
    route="support-routing",
    test_budget_usd="auto",
)

print(answer.content)
if answer.content.strip().lower() == expected:
    answer.accept()
else:
    answer.correct(expected)
print(evalt.route_status("support-routing"))
```

`Evalt()` reads `OPENROUTER_API_KEY` automatically from the process environment or a
`.env` file in the current working directory. An explicit `api_key=` takes precedence.
In an interactive terminal it also prints compact route, actual provider cost, automatic
request ceiling, and bounded maintenance progress to `stderr`. Use
`Evalt(show_progress=False)` for a silent service process, or pass
`progress_callback=...` to receive structured event dictionaries without parsing text.

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

Automatic first-route requests give the one-time AI suite designer 300 seconds, then
use a 120-second per-provider deadline for candidate and production responses. Set
`designer_request_timeout_seconds=` or `test_request_timeout_seconds=` on
`Evalt.run(...)` when the workload needs different limits. A candidate effort that
times out cannot earn a higher reasoning rung. Explicit `Suite` workflows keep their
independently configurable 600-second default. During the broad screen, interactive
progress reports each settled model configuration, validation rate, latency, spend,
and total elapsed time. The same stream names every designer model and attempt. A
malformed structured draft is rejected and retried once inside the same workflow
budget before Evalt falls back to another cost-qualified designer role or fails closed.
For semantic judges, Evalt also replaces AI-authored positive calibration guesses with
identity controls: each known-pass candidate is exactly its approved answer. This keeps
a designer from calling a different score or factual claim a known pass.

The first call now performs the bounded AI-designed tournament by default. It fails
closed without serving or promoting a route when no configuration clears the requested
accuracy, the test is exploratory, judge calibration fails, or the shared test budget
cannot complete a valid comparison. `answer.accept()` records the returned output as
correct; `answer.correct(expected)` records the desired output when it was wrong. Once
enough real labels accumulate, Evalt calibrates the semantic judge against known passes
and corrected failures before a feedback-based retest. Already-launched bounded
maintenance is not abandoned when a short script exits; call
`evalt.wait_for_maintenance()` when an application needs an explicit synchronization
point. Changing the source prompt requires a new first-route tournament while preserving
the old version in the audit history.

The focused default is `objective="lowest_cost_at_accuracy"` with
`target_accuracy=0.95`: Evalt promotes the cheapest tested configuration that clears that
approved bar. Omitting `price_usd` uses a request-sized automatic safety ceiling for the
selected route; it does not impose a hidden $0.02 limit. Set `price_usd` only when the
production call itself has a hard monetary ceiling. This is independent from
`test_budget_usd`, which caps evaluation and maintenance. A price-first frontier and an incumbent-preservation migration mode remain
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
models in the observed task-capability band. Ordinary supported efforts may run in the
same parallel wave. Extra-high and maximum effort are staged: validation must still be
within one case of the quality target, the prior rung must have completed without a
timeout, and its measured p90 latency must satisfy the route's production-speed ceiling.
Final-test luck never unlocks a higher rung. Reasoning hone lanes reuse the same frozen
prompt and few-shot package, so the comparison measures reasoning rather than silently
re-running prompt search. The first request receives large,
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

### Represent the real workload, including its hard tail

An overall score can hide a model that is excellent on frequent easy inputs and unsafe
on rarer difficult ones. Evalt therefore supports a production-weighted, stratified
suite. Give related variants the same `group`, label their `difficulty`, and optionally
set a `weight` that reflects expected traffic. The splitter puts examples from every
group into training, validation, and final test. `difficulty_thresholds` then makes the
hard tail a separate promotion gate instead of letting routine volume average it away.

The excerpt below shows the added fields; a valid suite includes at least five cases in
each declared group.

```json
{
  "examples": [
    {
      "id": "routine-01",
      "group": "ordinary-refund",
      "difficulty": "routine",
      "weight": 6,
      "input": "Unopened item, receipt, 12 days after delivery.",
      "approved_output": "approve"
    },
    {
      "id": "adversarial-01",
      "group": "policy-conflict",
      "difficulty": "adversarial",
      "weight": 1,
      "critical": true,
      "input": "A damaged final-sale item arrived after the normal window.",
      "approved_output": "manual_review"
    }
  ],
  "quality_threshold": 0.95,
  "difficulty_thresholds": {
    "routine": 0.95,
    "complex": 0.90,
    "adversarial": 0.85
  }
}
```

If one example declares a group, every example must declare one, and every group needs
at least five scenarios so it can contribute evidence to all three splits. Weights
affect measured pass rates; they do not decide which examples are hidden. A production
route is promoted only when it clears both the overall weighted target and every named
difficulty floor. Keep rare catastrophic rules in an explicit hard-constraint judge as
well; a statistical slice is not a substitute for a deterministic veto.

Speed is durable route state, not a one-time benchmark option. There is no latency
ceiling by default. Set
`max_latency_seconds=3.0` when each production response needs a ceiling. This does not
limit the total tournament wall time. Evalt reports measured p50 and
p90 for every completed route, persists the limit in SQLite, and refuses to promote a
later maintenance winner whose measured p90 misses it. OpenRouter's provider
price/latency/throughput preferences help search, but cannot replace those frozen-run measurements. The default deadline
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

Prompt search can rewrite instructions, select customer-approved training examples as
few-shot demonstrations, or combine both. It never selects demonstrations from the
validation or final-test partitions, excludes a training case from its own prompt, and
records each selected example with <code>source_split: "train"</code>. A successful prompt
package is re-screened on cheap model lanes that looked weak under the original prompt;
those lanes still must pass the untouched final test before they can win.
Set `"optimize_prompt": false` in a suite—or `optimize_prompt=False` on a durable
`Evalt.run(...)` route—to hold the supplied prompt exactly fixed. That disables rewrites,
few-shot selection, and cross-model prompt propagation while leaving model, reasoning,
provider, validation, and final-test comparisons intact.
The CLI equivalent is `evalt optimize evalt.json --fixed-prompt` (or
`evalt run ... --fixed-prompt` for a durable production route).

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
Every result also carries `quality_gate_status`. `NO_CONFIGURATION_PASSED` means the
reported best-observed configuration is diagnostic only and must not be promoted as a
production route.
If the cap leaves configurations unfinished, the result also records a structured
`continuation_recommendation` with those configuration IDs and a bounded next cap. It
uses a transparent 1.5× heuristic with at least $0.25 additional headroom;
`automatic_spend` is always `false`, so Evalt never silently extends a tournament.

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
against the one hard suite budget. Evalt measures both training evidence and validation
before skipping prompt learning; a perfect score on a small validation slice alone does
not suppress a potentially useful rewrite. The frozen final test remains promotion-only.
Exact-text and exact-JSON suites use contract-sized visible-output reservations, while
explicit reasoning configurations retain large hidden-reasoning headroom.

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
