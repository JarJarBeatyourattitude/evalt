# Migrate OpenAI Evals results without inventing the missing parts

OpenAI Evals result JSONL records what happened in historical runs. It is not, by
itself, a runnable test definition: it may omit the original prompt, graders, model
settings, and approved answers. Evalt's importer is deliberately conservative about
that boundary.

## Import the recoverable cases

```bash
evalt import-openai-results results.jsonl \
  --prompt-file system-prompt.txt \
  --output evalt.json \
  --model qwen/qwen3.5-9b \
  --model google/gemini-3-flash-preview
```

The command runs offline. It writes:

- `evalt.json`, only when at least three rows contain both an input and an explicit
  approved/reference output; and
- `evalt.json.migration-report.json`, which lists imported, skipped, and malformed
  rows plus the fields used for each import.

Candidate model outputs in fields such as `output`, `response`, or `completion` are
never promoted to approved answers. Review every imported answer before spending:

```bash
evalt validate evalt.json
evalt optimize evalt.json --output evalt-result.json
```

## What still needs a human decision

You must supply the original task prompt. You should also check the recovered labels or
reference answers, add important edge cases that were absent from the historical run,
and confirm that the candidate models and semantic evaluator fit the task. Migration
gets a real suite to the review boundary; it does not certify historical results.

## Report a migration gap

If the importer cannot represent your export, open the
[structured migration report](https://github.com/JarJarBeatyourattitude/evalt/issues/new?template=openai-evals-migration.yml).
Do not include prompts, API keys, customer data, request IDs, or raw production rows.
