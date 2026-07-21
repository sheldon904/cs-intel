"""LLM-based intent & sentiment labeling of support tickets.

Each ticket is classified into one of four intents (cancel-risk, expansion-
signal, confusion, bug-report) plus a sentiment. Quality is measured against a
frozen 100-ticket gold set with accuracy and Cohen's kappa. Predicted intent
rates per account are then fed back to the churn model as features.
"""
