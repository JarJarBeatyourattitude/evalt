"""Recommended first optimization: AI drafts; a human approves; Evalt tests."""

from evalt import Evalt


evalt = Evalt()
draft = evalt.optimize_task(
    task="Route recurring support tickets to billing, account, or technical.",
    prompt="Return exactly one lowercase label: billing, account, or technical.",
    route="support-routing",
    case_control="review",
    workflow_budget_usd=1.00,
)

draft.save("support-routing-draft.json")
for example in draft.examples:
    print(f"\n[{example.difficulty}] {example.id}")
    for turn in example.conversation():
        print("INPUT:", turn.input)
        print("EXPECTED:", turn.approved_output)

confirmation = input("\nType APPROVE only after reviewing every expected output: ")
if confirmation.strip() != "APPROVE":
    raise SystemExit("Draft saved without running a tournament.")

# This explicit call is the trust boundary. Pass edited examples to approve(...)
# when any AI-authored input or expected output needs correction.
suite = draft.approve()
result = evalt.run(suite)

print(result.winner.model)
print(result.winner.holdout_pass_rate)
print(result.regression_suite["evidence_provenance"])
