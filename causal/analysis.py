"""DoWhy confounding analysis for feature-adoption -> churn.

The synthetic data has a *known* causal structure::

    account size ---> feature adoption        (small accounts adopt less)
    account size ---> churn                    (small accounts churn more)
    feature adoption ---> churn                (the effect we care about)

Because account size is a common cause of both adoption and churn, the naive
difference in churn between high- and low-adoption accounts is **confounded**:
it mixes adoption's real effect with the size effect. This module quantifies the
bias and recovers the true effect with a backdoor adjustment on account size.
"""

from __future__ import annotations

import json
import warnings
from typing import Any

import numpy as np
import pandas as pd

from common.paths import ACCOUNTS_PARQUET, CAUSAL_RESULTS_JSON, GROUND_TRUTH_JSON, ensure_dirs
from datagen.schema import AccountCols

warnings.filterwarnings("ignore")

TREATMENT = "high_adoption"
OUTCOME = "churned"
CONFOUNDER = "log_seats"


def build_causal_frame() -> pd.DataFrame:
    """Assemble the observable analysis frame (treatment, outcome, confounder)."""
    acc = pd.read_parquet(ACCOUNTS_PARQUET)
    return pd.DataFrame(
        {
            TREATMENT: acc[AccountCols.HIGH_ADOPTION].astype(int),
            OUTCOME: acc[AccountCols.CHURNED].astype(int),
            CONFOUNDER: np.log(acc[AccountCols.SEATS].astype(float)),
        }
    )


def naive_effect(df: pd.DataFrame) -> float:
    """Unadjusted difference in churn between high- and low-adoption accounts."""
    hi = df.loc[df[TREATMENT] == 1, OUTCOME].mean()
    lo = df.loc[df[TREATMENT] == 0, OUTCOME].mean()
    return float(hi - lo)


def gformula_logistic(df: pd.DataFrame, confounders: list[str]) -> float:
    """Backdoor adjustment via a logistic outcome model + g-computation.

    Correct for a binary outcome (unlike a linear probability model): fit
    P(churn | treatment, confounders), then average the predicted risk
    difference of do(T=1) vs do(T=0) over the observed confounder distribution.
    """
    import statsmodels.api as sm

    cols = [TREATMENT, *confounders]
    X = sm.add_constant(df[cols].astype(float))
    model = sm.Logit(df[OUTCOME].astype(float), X).fit(disp=0)
    d1, d0 = df.copy(), df.copy()
    d1[TREATMENT], d0[TREATMENT] = 1, 0
    p1 = model.predict(sm.add_constant(d1[cols].astype(float), has_constant="add"))
    p0 = model.predict(sm.add_constant(d0[cols].astype(float), has_constant="add"))
    return float((p1 - p0).mean())


def run_causal_analysis(write: bool = True) -> dict[str, Any]:
    """Full DoWhy pipeline: identify -> estimate (naive + backdoor) -> refute."""
    from dowhy import CausalModel

    ensure_dirs()
    df = build_causal_frame()
    gt = json.loads(GROUND_TRUTH_JSON.read_text(encoding="utf-8"))
    true_ate = float(gt["true_ate_risk_diff"])

    naive = naive_effect(df)

    model = CausalModel(
        data=df,
        treatment=TREATMENT,
        outcome=OUTCOME,
        common_causes=[CONFOUNDER],
    )
    identified = model.identify_effect(proceed_when_unidentifiable=True)

    est_lr = model.estimate_effect(
        identified, method_name="backdoor.linear_regression", test_significance=True
    )
    adjusted_lr = float(est_lr.value)

    # g-computation with a logistic outcome model (correct for the binary outcome).
    adjusted_gformula = gformula_logistic(df, [CONFOUNDER])

    # Second estimator (propensity-score stratification) as a robustness check.
    try:
        est_ps = model.estimate_effect(
            identified, method_name="backdoor.propensity_score_stratification"
        )
        adjusted_ps: float | None = float(est_ps.value)
    except Exception as exc:  # pragma: no cover - estimator availability varies
        adjusted_ps = None
        est_ps = None
        print(f"[causal] propensity estimator skipped: {exc}")

    # Refutations: a credible causal estimate should survive these.
    refutations: dict[str, Any] = {}
    for label, method in [
        ("random_common_cause", "random_common_cause"),
        ("placebo_treatment", "placebo_treatment_refuter"),
        ("data_subset", "data_subset_refuter"),
    ]:
        try:
            ref = model.refute_estimate(identified, est_lr, method_name=method)
            new_effect = float(np.ravel(np.asarray(ref.new_effect, dtype=float))[0])
            refutations[label] = {
                "estimated_effect": adjusted_lr,
                "new_effect": new_effect,
                "p_value": _extract_pvalue(ref),
            }
        except Exception as exc:  # pragma: no cover
            refutations[label] = {"error": str(exc)}

    bias_removed = naive - adjusted_gformula
    results = {
        "treatment": TREATMENT,
        "outcome": OUTCOME,
        "confounder_adjusted": CONFOUNDER,
        "true_ate_risk_diff": true_ate,
        "naive_estimate": naive,
        "backdoor_gformula_logistic": adjusted_gformula,
        "backdoor_linear_regression": adjusted_lr,
        "backdoor_propensity_stratification": adjusted_ps,
        "naive_bias_vs_true": naive - true_ate,
        "adjusted_bias_vs_true": adjusted_gformula - true_ate,
        "bias_removed_by_adjustment": bias_removed,
        "interpretation": _interpret(naive, adjusted_gformula, true_ate),
        "refutations": refutations,
    }
    if write:
        CAUSAL_RESULTS_JSON.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def _extract_pvalue(ref: Any) -> float | None:
    p = getattr(ref, "refutation_result", None)
    if isinstance(p, dict) and "p_value" in p:
        val = p["p_value"]
        return float(np.ravel(val)[0]) if val is not None else None
    return None


def _interpret(naive: float, adjusted: float, true_ate: float) -> str:
    return (
        f"Naive analysis says high adoption cuts churn by {abs(naive) * 100:.1f} "
        f"percentage points. After adjusting for account size, the effect is "
        f"{abs(adjusted) * 100:.1f}pp — close to the true {abs(true_ate) * 100:.1f}pp. "
        f"The naive number overstates adoption's impact by "
        f"{abs(naive - true_ate) * 100:.1f}pp because small accounts both adopt "
        f"less and churn more; that size effect was masquerading as an adoption effect."
    )
