# Security

Please do not open a public issue for a vulnerability that could expose credentials, prompts, model responses, or route history. Use GitHub's private vulnerability reporting for this repository.

Evalt reads `OPENROUTER_API_KEY` at runtime and must never persist it in suites, result files, logs, or SQLite route state. Reports and examples should contain synthetic data unless the data owner has explicitly approved disclosure.
