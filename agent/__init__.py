"""Next-best-action agent with a propose-then-approve gate + learning memory.

The agent reads an at-risk account's state (churn risk, SHAP drivers, intent-
tagged tickets), consults a hybrid memory of past operator decisions, and
proposes a single intervention. Nothing is executed: a human approves or rejects,
and that decision is written back to memory so future proposals adapt — the
system "learns from every interaction".
"""
