# Contributing

Issues and focused pull requests are welcome. Before opening a pull request:

1. Create a Python 3.10+ virtual environment.
2. Install the package with `python -m pip install -e .`.
3. Run `python -m unittest discover -s tests -v`.
4. Keep provider calls out of tests unless the test is explicitly marked and bounded.
5. Never include API keys, prompts from private production data, or provider responses containing private data.

For a behavior change, include a regression test and update the relevant README contract. Evalt promotes configurations only from frozen, inspectable evidence; changes that weaken budget reservation, split isolation, or failure receipts will not be accepted.
