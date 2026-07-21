"""Shared fixtures for the cs-intel test suite.

The dataset + model artifacts are already built (read-only) under ``data/`` and
``artifacts/``. Fixtures that read those are session-scoped so the (comparatively
expensive) reads/feature-builds happen once for the whole run. Tests that need
their own data generate a tiny in-memory dataset instead.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

# Ensure the repo root is importable regardless of how pytest is invoked
# (``pytest`` vs ``python -m pytest``, cwd, etc.). The source packages
# (datagen, churn, intents, ...) live at the repo root.
ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def tiny_gen():
    """A small deterministic in-memory dataset (no files written)."""
    from datagen.generate import GenParams, generate

    return generate(GenParams(seed=123, n_accounts=60), write=False)


@pytest.fixture(scope="session")
def feature_table():
    """The real per-account feature table built from the pre-built artifacts."""
    from churn.features import build_feature_table

    return build_feature_table(include_intents=True)


@pytest.fixture(scope="session")
def repo():
    """Repository over the pre-built pipeline artifacts."""
    from api.repository import Repository

    return Repository()


@pytest.fixture(scope="session")
def intent_eval():
    """Labeler evaluation on the frozen gold set (reads existing labels)."""
    from intents.eval import evaluate_labeler

    return evaluate_labeler()


@pytest.fixture(scope="session")
def causal_frame():
    """Observable causal analysis frame from the pre-built accounts."""
    from causal.analysis import build_causal_frame

    return build_causal_frame()


@pytest.fixture(scope="session")
def causal_run():
    """Full causal pipeline result (write=False so no artifacts are touched)."""
    from causal.analysis import run_causal_analysis

    return run_causal_analysis(write=False)
