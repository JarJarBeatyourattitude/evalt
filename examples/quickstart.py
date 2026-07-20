"""Minimal Evalt production-route example.

Evalt reads OPENROUTER_API_KEY from the environment or a local .env file.
"""

from evalt import Evalt


ticket = "Please help—the website won't load."
expected = "technical"

evalt = Evalt()
answer = evalt.run(
    "Classify this request. Return exactly one lowercase label: billing, account, or technical.",
    ticket,
    route="support-routing",
    target_accuracy=0.95,
    test_budget_usd="auto",
)

print(answer.content)
if answer.content.strip().lower() == expected:
    answer.accept()
else:
    answer.correct(expected)

print(evalt.route_status("support-routing"))
