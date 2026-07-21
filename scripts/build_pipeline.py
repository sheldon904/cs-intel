"""Run the full offline pipeline end to end.

    python scripts/build_pipeline.py            # build only if artifacts are missing
    python scripts/build_pipeline.py --force    # always rebuild from scratch

Order: datagen -> intents (label + eval + features) -> churn (train + SHAP) ->
causal (DoWhy) -> LLMOps rollup. Safe to run repeatedly; deterministic given the
seed. This is what the Docker entrypoint calls before starting the API.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Allow running as a bare script (`python scripts/build_pipeline.py`) by putting
# the repo root on the path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.config import get_settings  # noqa: E402
from common.paths import CHURN_MODEL_PKL  # noqa: E402


def _log(msg: str) -> None:
    print(f"[build] {msg}", flush=True)


def run(force: bool = False) -> None:
    if CHURN_MODEL_PKL.exists() and not force:
        _log(f"artifacts present ({CHURN_MODEL_PKL.name}); skipping build (use --force to rebuild)")
        return

    settings = get_settings()
    t0 = time.perf_counter()

    # Reset the per-run LLM trace log so cost-per-account isn't double counted on rebuilds.
    from common.paths import LLM_USAGE_JSONL

    LLM_USAGE_JSONL.unlink(missing_ok=True)

    _log(f"1/5 datagen (seed={settings.seed}, n_accounts={settings.n_accounts})")
    from datagen.generate import GenParams, generate

    generate(GenParams(seed=settings.seed, n_accounts=settings.n_accounts, n_months=settings.n_months))

    _log(f"2/5 intents: labeling tickets (provider={settings.effective_provider})")
    from intents.eval import evaluate_labeler
    from intents.labeler import build_account_intent_features, label_all

    label_all()
    build_account_intent_features()
    ev = evaluate_labeler()
    _log(f"    intent kappa={ev['intent']['cohen_kappa']:.3f}  sentiment kappa={ev['sentiment']['cohen_kappa']:.3f}")

    _log("3/5 churn: train XGBoost + logistic, SHAP, calibration")
    from churn.train import train_and_save

    m = train_and_save()
    _log(f"    XGBoost AUC={m['xgboost']['auc']:.3f}  intent lift={m['intent_lift']['auc_lift']:+.3f}")

    _log("4/5 causal: DoWhy backdoor analysis")
    from causal.analysis import run_causal_analysis

    c = run_causal_analysis()
    _log(f"    naive={c['naive_estimate']:+.3f} -> adjusted={c['backdoor_gformula_logistic']:+.3f} (true={c['true_ate_risk_diff']:+.3f})")

    _log("5/5 LLMOps rollup")
    from common.llmops import append_eval_history, build_llmops_summary

    append_eval_history()
    s = build_llmops_summary()
    _log(f"    {s['total_calls']} LLM calls, projected ${s['projected_cost_per_account']}/account")

    _log(f"done in {time.perf_counter() - t0:.1f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the cs-intel pipeline artifacts.")
    parser.add_argument("--force", action="store_true", help="rebuild even if artifacts exist")
    args = parser.parse_args()
    try:
        run(force=args.force)
    except Exception as exc:  # surface a clean error for the container logs
        print(f"[build] FAILED: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
