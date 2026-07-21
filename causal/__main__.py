"""CLI: ``python -m causal`` runs the DoWhy confounding analysis."""

from __future__ import annotations

from causal.analysis import run_causal_analysis


def main() -> None:
    r = run_causal_analysis()
    print("When your churn model lies to you")
    print("=" * 40)
    print(f"  true causal effect (known):     {r['true_ate_risk_diff']:+.4f}")
    print(f"  naive (confounded) estimate:    {r['naive_estimate']:+.4f}")
    print(f"  backdoor-adjusted (DoWhy LR):   {r['backdoor_linear_regression']:+.4f}")
    if r["backdoor_propensity_stratification"] is not None:
        print(f"  backdoor-adjusted (propensity): {r['backdoor_propensity_stratification']:+.4f}")
    print(f"  bias removed by adjustment:     {r['bias_removed_by_adjustment']:+.4f}")
    print()
    print(r["interpretation"])
    print()
    print("Refutations:")
    for name, ref in r["refutations"].items():
        if "error" in ref:
            print(f"  {name}: error - {ref['error']}")
        else:
            print(f"  {name}: effect {ref['estimated_effect']:+.4f} -> {ref['new_effect']:+.4f}")


if __name__ == "__main__":
    main()
