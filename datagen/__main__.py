"""CLI: ``python -m datagen`` regenerates the synthetic dataset."""

from __future__ import annotations

import argparse

from common.config import get_settings
from datagen.generate import GenParams, generate


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Generate the cs-intel synthetic dataset.")
    parser.add_argument("--seed", type=int, default=settings.seed)
    parser.add_argument("--n-accounts", type=int, default=settings.n_accounts)
    parser.add_argument("--n-months", type=int, default=settings.n_months)
    args = parser.parse_args()

    params = GenParams(seed=args.seed, n_accounts=args.n_accounts, n_months=args.n_months)
    out = generate(params)
    accounts, monthly, tickets, gt = (
        out["accounts"],
        out["monthly"],
        out["tickets"],
        out["ground_truth"],
    )
    print(f"accounts:  {len(accounts):>6}  (churn rate {gt['churn_base_rate']:.1%})")
    print(f"monthly:   {len(monthly):>6}  rows")
    print(f"tickets:   {len(tickets):>6}  ({int(tickets['in_gold_eval'].sum())} in gold eval)")
    print(f"true ATE (risk diff):  {gt['true_ate_risk_diff']:+.4f}")
    print(f"naive diff (confounded): {gt['naive_diff_risk_diff']:+.4f}")


if __name__ == "__main__":
    main()
