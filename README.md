# Evalt Python SDK

[![PyPI](https://img.shields.io/pypi/v/evalt?label=PyPI&logo=pypi&logoColor=white)](https://pypi.org/project/evalt/)
[![Python](https://img.shields.io/pypi/pyversions/evalt?logo=python&logoColor=white)](https://pypi.org/project/evalt/)
[![CI](https://github.com/JarJarBeatyourattitude/evalt/actions/workflows/ci.yml/badge.svg)](https://github.com/JarJarBeatyourattitude/evalt/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-2f7a65.svg)](LICENSE)

Evalt turns a recurring AI task into a tested production route. On a route's first
call, it can design and calibrate the test, compare prompt/model/reasoning/few-shot
configurations under one hard budget, remember the lowest-cost passing package, and
then answer the real input through that package. Later calls reuse it immediately.

`target_accuracy` is an observed frozen-suite gate, not a guarantee about the unknown
production distribution. Results separately report distinct final-test scenarios,
repeated executions, a one-sided 95% exact lower bound when equal-weight binomial
evidence is appropriate, and whether that bound actually supports the requested
target. Repeats measure consistency and never inflate the distinct-scenario count. A
10/10 final test therefore installs a clearly provisional route: its lower bound is
about 74.1%, and 59/59 distinct successes would be needed to support a 95% target
under those assumptions.

## Install

```bash
python3 -m pip install --upgrade -r https://evalt.onrender.com/python-sdk/latest.txt
```

Use the same interpreter for installation and every Evalt command. Bare `pip`
can update a different Python, and PyPI may trail the exact hosted release. Run
`python3 -m evalt doctor` to see the imported version, executable, hosted version,
the exact hosted wheel URL, and a copyable corrective command without printing
the workspace capability. The stable requirements URL contains one pinned wheel,
so an already-installed CLI can recommend a release channel that does not go stale.

Optionally connect local routes to the hosted workspace:

```bash
python3 -m evalt connect
```

The command opens the private dashboard, stores its capability token once for the
current user, and immediately publishes any existing sanitized route summaries from
the current `.evalt/evalt.db` without calling a model. Routes remain connected when
scripts run from another project folder.
Pass `--state path/to/evalt.db` only when a project needs a separate workspace. The CLI
and browser show a safe `ws_...` workspace ID; those IDs must
match. Evalt synchronizes only operational route metadata and bounded progress;
prompts, inputs, outputs, cases, provider keys, request bodies, and raw responses stay
local. Each invocation gets an opaque run ID, so repeated tournaments on one durable
route have separate running, completed, or failed progress streams. Every visible run
prints either `DASHBOARD SYNCED` or `DASHBOARD SYNC FAILED`. A newly optimized route
also reports separate test-design, tournament, route-install, production-call,
orchestration, and total timings so a slow first run has an inspectable cause. Use
`python3 -m evalt dashboard` to reopen it, `python3 -m evalt --status` to compare the
ID without opening a browser, and `python3 -m evalt disconnect` to remove the local
connection. If a route is missing, run `python3 -m evalt doctor` to compare the
imported package, Python executable, PATH console shim, safe workspace ID, and
local/hosted route counts, then `python3 -m evalt dashboard --sync-existing` to recover
current route summaries without provider spend. The interpreter-bound form is the
canonical path when a machine has more than one Python installation.
Dashboard
availability never affects a production call.

Run comparisons include test-suite and evaluator-contract lineage. The dashboard says
`Same`, `Changed`, or `Unknown` for each and refuses to label quality movement when a
known contract changed. Suite contents, raw hashes, evaluator prompts, and evaluator
credentials remain local. The SDK sends only workspace-keyed opaque identifiers, so
the same local contract cannot be correlated across different workspaces.

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
python -m pip install dist/evalt-0.10.29-py3-none-any.whl
python -m evalt --version
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

1. chooses a test designer from the current intelligence-and-price catalog, then creates
   25 routine, ambiguous, adversarial, boundary, and realistic-domain cases in five
   parallel batches;
2. calibrates the proposed exact, semantic, or numeric-tolerance judge on separate controls;
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

To deliberately re-test an existing text route after changing its quality target,
budget, latency ceiling, prompt policy, or candidate models, pass
`reoptimize=True`. Evalt runs a new qualifying test and replaces the saved package only
if the new result clears the frozen gates:

```python
answer = evalt.run(
    prompt,
    input,
    task=task,
    route="support-routing",
    reoptimize=True,
    target_accuracy=0.97,
    test_budget_usd="auto",
    max_test_budget_usd=1.25,
    models=["openai/gpt-5.4-nano", "google/gemini-2.5-flash-lite"],
)
```

Human-labeled image routes must re-run their approved `Suite`; Evalt never invents
image evidence or pulls image bytes from the hosted dashboard. Install the qualified
winner explicitly:

```python
result = evalt.qualify_route(approved_suite, route="image-condition")
```

The tested model, prompt, request envelope, evidence summary, and modality contract
become an immutable local route package. Image bytes and image-bearing few-shot
examples are never copied into SQLite or the dashboard.

## Qualified route versions and rollback

Every initial qualification, maintenance promotion, and rollback creates an immutable
local package. Exploratory, failed, skipped, and bootstrap-only runs never become
rollback targets.

```bash
python3 -m evalt versions --route support-routing
python3 -m evalt annotate-version --route support-routing \
  --version rv_0123456789abcdefabcd \
  --alias known-good \
  --note "Approved before the July catalog refresh."
python3 -m evalt rollback --route support-routing --version known-good --yes
```

Rollback atomically restores the selected qualified package and records the decision
without calling a provider. It preserves production call and feedback history and
keeps current per-call spending controls in place. A later call with a different
source prompt deliberately unqualifies the restored package instead of silently
borrowing its evidence. The hosted dashboard can show bounded version metadata and
copy these commands, but it cannot mutate the locally serving route.

Aliases and notes are optional private operator metadata. They remain in the local
route database, are excluded from dashboard synchronization and decision events, and
never replace the canonical `rv_…` ID. Aliases are unique per route, lowercase, and
accepted by rollback only; production `route_version=` pins still require the exact
immutable ID. Use `--clear-alias` or `--clear-note` to remove either value.

Pin deployed code to the exact package it was tested with:

```python
from evalt import Evalt, RouteVersionMismatch

evalt = Evalt()
version = evalt.route_status("support-routing")["current_package_id"]

try:
    answer = evalt.run(
        prompt,
        input,
        route="support-routing",
        route_version=version,
    )
except RouteVersionMismatch as error:
    # No provider call started. Review the new current version or roll back.
    print(error.current_package_id)
    raise
```

The pin loads and serves the immutable package snapshot itself. A stale, missing,
cross-route, bootstrap-only, unqualified, or damaged version fails before model spend.
After qualification or rollback, read the new `current_package_id`, review it, and
update the deployment deliberately. Omitting `route_version` preserves the existing
automatic route behavior for scripts that do not need deployment pinning.

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
result = evalt.qualify_route(suite, route="support-routing")
print(result.winner.model, result.winner.holdout_pass_rate)
```

The catalog-selected designer covers routine, ambiguous, adversarial, boundary, and
realistic-domain cases, and recommends exact, semantic, or numeric-tolerance judging.
Scalar ratings such as 0–10 or 0–100 use an explicit scale and absolute tolerance
instead of exact equality or an LLM's unspoken tolerance. AI-generated semantic suites
must use a judge model different from the suite designer; Evalt does not use its
designer as a correlated fallback judge. Evalt then
splits the frozen contract, varies prompts and approved few-shot examples, tests current
models and supported reasoning levels in parallel, and promotes only on the untouched
final test. A balanced 25-case automatic suite reserves ten unique cases for final
confirmation and runs each twice after the prompt/model package is frozen.

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

### OpenRouter request settings are part of the route

Use `request_options` for the OpenRouter Chat Completions settings your production
task needs. Target-model tests receive the exact same canonical envelope; Evalt's test
designer, prompt optimizer, and judge do not.

```python
answer = evalt.run(
    prompt,
    ticket,
    route="support-routing",
    max_tokens=2048,
    request_options={
        "temperature": 0.2,
        "top_p": 0.95,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "support_route",
                "strict": True,
                "schema": route_schema,
            },
        },
        "tools": tools,
        "tool_choice": "auto",
        "parallel_tool_calls": False,
        "provider": {
            "require_parameters": True,
            "data_collection": "deny",
            "zdr": True,
        },
        "plugins": [{"id": "response-healing"}],
    },
)
```

The forward-compatible mapping also passes sampling and penalty controls, stop
conditions, prediction, provider preferences, transforms, web search, verbosity, and
new JSON-serializable OpenRouter request fields. Evalt owns `model`, fallback `models`,
`messages`, `prompt`, usage accounting, streaming lifecycle, output-token fields, and
the reasoning effort encoded in each tested model configuration. Zero Data Retention,
denied provider data collection, and required-parameter routing remain enforced safety
defaults even if a conflicting provider option is supplied.

The winning route stores the envelope and its SHA-256 fingerprint. Later calls that
omit `request_options` and `max_tokens` reuse the tested values. An explicit change is
allowed, but emits `RequestEnvelopeDriftWarning`, records an audit event, and returns
`answer.request_envelope_validated == False`. Set `strict_request_options=True` to stop
before provider spend instead. Normalized tool responses are available as
`answer.tool_calls`. Complete user/tool/assistant message lists may be supplied as the
production input. Image input is a separately validated contract described below; it
is no longer treated as an unqualified generic passthrough.

### Image input

Evalt supports image understanding through OpenRouter Chat Completions for local PNG,
JPEG, WebP, and GIF files and public HTTPS image URLs whose path ends in one of those supported extensions. It does not claim support for
the separate image-generation, PDF, audio, or video endpoints.

You do not need AI to make the examples. Choose images whose answers you already
know, write each expected label yourself, and review the set before running it.
AI-generated or automatically inferred image labels do not count as approved ground
truth.

```python
from evalt import Evalt, Example, ImageInput, Suite, multimodal_input

examples = tuple(
    Example(
        multimodal_input("Return exactly one label: damaged or intact.", ImageInput.from_path(path)),
        approved_output=label,
        id=f"package-{index}",
    )
    for index, (path, label) in enumerate(approved_images, start=1)
)

suite = Suite(
    name="package-condition",
    prompt="Inspect the package image. Return exactly one label: damaged or intact.",
    examples=examples,
    models=("google/gemini-2.5-flash-lite", "openai/gpt-5.4-nano"),
    evaluator={"type": "exact_text"},
    optimize_prompt=False,
    max_optimization_cost_usd=0.25,
)

result = Evalt().run(suite)
print(result.winner.model, result.winner.holdout_pass_rate)
```

For a complete runnable version that reads a small human-labeled JSON manifest, see
[`examples/image_suite.py`](examples/image_suite.py).

`ImageInput.from_path(...)` validates the file structure, dimensions, checksums or
container boundaries, and the 20 MiB per-image safety
limit, then creates a data URL in memory without changing the original. Use
`ImageInput.from_url(...)` for a public HTTPS URL without embedded credentials and with a `.png`, `.jpg`, `.jpeg`, `.webp`, or `.gif` path suffix.
Embedded images are limited to 40 MiB per message and per suite, so malformed or
resource-exhausting image work fails before any provider call.
`multimodal_input(...)` puts the text instruction first, followed by the images.

Image-bearing suites currently require `optimize_prompt=False`: Evalt compares models
against the frozen prompt and customer-approved images instead of pretending an
automatic text-only prompt writer saw the evidence. Text-only target models are omitted
before provider spend. Automatic first-route design refuses image input; use an explicit
approved suite, or `first_run="bootstrap"` for an unqualified one-off call. Raw images,
filenames, paths, source URLs, and thumbnails never enter hosted dashboard metadata.
Local call history stores only a bounded image descriptor, and image feedback is not
used for maintenance unless it is supplied again in an explicit approved suite.

Automatic first-route requests give each one-time AI suite-design request 45 seconds,
then use a 120-second per-provider deadline for candidate and production responses. Set
`designer_request_timeout_seconds=` or `test_request_timeout_seconds=` on
`Evalt.run(...)` when the workload needs different limits. A candidate effort that
times out cannot earn a higher reasoning rung. Explicit `Suite` workflows keep their
independently configurable 600-second default. During the broad screen, interactive
progress reports each settled model configuration, validation rate, latency, spend,
and total elapsed time. The same stream names every designer model and attempt. A
malformed structured draft is rejected and retried once inside the same workflow
budget before Evalt falls back through a separate live-catalog designer shortlist or
fails closed. The judge is never reused as a designer fallback. For an automatic first
route, `optimization_rounds=1` tests one prompt rewrite by default; raise it up to eight
when a harder workload justifies deeper prompt search.
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
`Evalt(request_timeout_seconds=...)`, or with `python3 -m evalt optimize --request-timeout ...`.
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
The CLI equivalent is `python3 -m evalt optimize evalt.json --fixed-prompt` (or
`python3 -m evalt run ... --fixed-prompt` for a durable production route).

## Explicit optimization and CI

```bash
python3 -m evalt init evalt.json
python3 -m evalt validate evalt.json
export OPENROUTER_API_KEY="..."
python3 -m evalt optimize evalt.json --output evalt-result.json
python3 -m evalt check evalt-result.json --min-pass-rate 0.95
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
python3 -m evalt check evalt-result.json \
  --min-pass-rate 0.95 \
  --max-cost-per-success 0.002 \
  --require-complete-coverage
```

The command exits `0` on pass, `1` when the measured result fails the gate, and `2` for an
invalid file or runtime error.

Protect a deployment from regressions against an earlier result from the identical
frozen suite:

```bash
python3 -m evalt check candidate.json \
  --baseline baseline.json \
  --min-pass-rate 0.95 \
  --max-regressions 0 \
  --max-quality-drop-pp 0 \
  --max-cost-increase-pct 10 \
  --max-p90-increase-ms 250 \
  --require-complete-coverage \
  --json
```

The default baseline contract allows no newly failing frozen cases, no missing cases,
and no aggregate quality drop. Different or missing suite hashes fail closed instead of
producing a misleading score comparison. Cost and p90 limits are opt-in; when requested,
both results must contain the corresponding measurement. The JSON report contains
`absolute_gate` and `baseline_gate` decisions but no prompts, inputs, outputs, approved
answers, or provider credentials.

To inspect the CI contract without a provider call:

```bash
python3 -m evalt check examples/passing-result.json --min-pass-rate 0.95 --require-complete-coverage
```

## Frozen route health

Recheck the exact selected configuration without starting another search or changing
the route:

```bash
python3 -m evalt monitor evalt-baseline.json \
  --route support-routing \
  --max-cost-usd 0.10 \
  --max-regressions 0 \
  --max-quality-drop-pp 0 \
  --max-cost-increase-pct 15 \
  --max-p90-increase-ms 250 \
  --output evalt-monitor-result.json \
  --history .evalt/monitor-history.jsonl
```

The positive spend cap is mandatory. Evalt verifies the frozen suite hash, selected
prompt/model/few-shot package, evaluator, request options, final-test split, and complete
baseline before the first provider call. It returns `0` for `HEALTHY`, `1` for a measured
`REGRESSION`, and `2` for an invalid contract, exhausted budget, provider failure, or
runtime error. The full result remains compatible with `check`, `compare`, HTML, and
JUnit reporting. History contains aggregate deltas only.

Pass `--route` to synchronize only the verdict and aggregate quality, case, cost,
latency, and spend values to an already connected dashboard. Prompts, inputs, approved
answers, outputs, and judge reasons remain local. For image baselines, add
`--suite reviewed-image-suite.json`; exported results intentionally omit raw image data,
and recurring checks reject mutable HTTPS fixtures in favor of embedded local fixtures.

The SDK surface is the same:

```python
from evalt import Evalt

health = Evalt().monitor(
    baseline_result,
    route="support-routing",
    max_cost_usd=0.10,
)
print(health.status)  # HEALTHY or REGRESSION
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
  "schema": "evalt-suite-v2",
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

For suites with at least 25 approved scenarios, the default split reserves 40% for the
final test, yielding ten distinct final-test scenarios from 25. Five remains the minimum
for a non-exploratory result, not a strong reliability claim. Repeats measure consistency;
they never inflate the distinct scenario count. A scenario may contain
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
