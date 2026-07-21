"""Synthetic B2B customer-success data generator.

Produces ~500 accounts observed over 24 months with a *known* churn-generating
process and one deliberately injected confounder (account size), so that both
the churn model's ground truth and the causal layer's true effect are knowable.
"""

from datagen.generate import GenParams, generate  # noqa: F401
