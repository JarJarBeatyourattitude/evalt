"""Minimal Evalt production-route example.

Evalt reads OPENROUTER_API_KEY from the environment or a local .env file.
"""

from evalt import Evalt


ticket = "Please help—the website won't load."
expected = "technical"

# The first call designs and judges 25 cases, runs the bounded prompt/model/reasoning
# tournament, saves the passing route, then answers this real ticket. Later calls reuse it.
# Interactive terminals show compact design, tournament, route, spend, and maintenance
# progress on stderr.
# Pass show_progress=False in a server, or progress_callback=... for structured events.
evalt = Evalt(show_progress=True)
answer = evalt.run(
    "Classify this request. Return exactly one lowercase label: billing, account, or technical.",
    ticket,
    task="Route recurring support tickets to billing, account, or technical.",
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
