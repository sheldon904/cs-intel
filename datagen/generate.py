"""The synthetic data generator.

Generative model (the "known" ground truth)
--------------------------------------------
Latent per-account variables::

    z_size          ~ standardized log(seats)          # ACCOUNT SIZE = confounder
    adoption_latent = a0 + a_size * z_size + noise      # big accounts adopt MORE
    mean_adoption   = sigmoid(adoption_latent)
    high_adoption   = 1[mean_adoption > median]         # binary treatment
    friction        ~ N(0, 1)                           # independent churn driver

Structural churn probability (this is what makes ground truth knowable)::

    logit P(churn) = c0
                   + c_size    * z_size                 # big accounts churn LESS (direct)
                   + c_treat   * high_adoption          # adoption REDUCES churn  (causal)
                   + c_friction * friction               # friction RAISES churn   (independent)
                   + c_intent  * intent_latent          # dissatisfaction (seen only via intent)
                   + c_inter   * relu(friction)*relu(intent_latent)   # super-additive interaction

Because ``account size`` drives *both* adoption (via ``a_size``) and churn (via
``c_size``), the naive association between adoption and churn is **confounded**.
The true average causal effect of ``high_adoption`` on churn is computed exactly
by intervening on the structural equation and stored in ``ground_truth.json`` —
that is the number the causal layer must recover via backdoor adjustment. The
interaction deliberately does NOT involve account size, so size's effect stays
linear and the backdoor adjustment cleanly identifies the adoption effect.

``intent_latent`` is observable ONLY through the cancel-risk *intent* of tickets
(not their volume or sentiment), so it gives the LLM intent features genuine,
orthogonal predictive lift. Support tickets otherwise proxy friction but do NOT
enter the structural churn equation, keeping the causal identification exact.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from common.paths import (
    ACCOUNTS_PARQUET,
    GROUND_TRUTH_JSON,
    MONTHLY_PARQUET,
    TICKETS_PARQUET,
    ensure_dirs,
)
from datagen.schema import (
    COMPANY_PREFIXES,
    COMPANY_SUFFIXES,
    INDUSTRIES,
    PRODUCT_FEATURES,
    AccountCols,
    Intent,
    MonthlyCols,
    Sentiment,
    TicketCols,
)
from datagen.tickets import generate_ticket_text

ANCHOR_DATE = pd.Timestamp("2023-01-01")  # fixed so dates are reproducible
SENTIMENT_SCORE = {Sentiment.POSITIVE: 1.0, Sentiment.NEUTRAL: 0.0, Sentiment.NEGATIVE: -1.0}


@dataclass(slots=True)
class GenParams:
    """All knobs of the generative process in one place."""

    seed: int = 42
    n_accounts: int = 500
    n_months: int = 24
    feature_window: int = 6  # trailing months used to build churn features

    # adoption ~ account size  (a_size > 0 => bigger accounts adopt more).
    # Strong coupling so account size is the DOMINANT confounder of adoption->churn.
    a0: float = 0.0
    a_size: float = 1.05
    a_noise: float = 0.6

    # structural churn logit coefficients. Kept deliberately moderate so churn
    # is genuinely stochastic (Bernoulli noise) rather than near-deterministic —
    # this caps the achievable AUC at a realistic ~0.80-0.88 instead of 1.0.
    c0: float = -1.05
    c_size: float = -0.85  # bigger accounts churn less (the confounding back-door)
    c_treat: float = -0.90  # high adoption reduces churn (TRUE causal effect)
    c_friction: float = 0.60  # friction raises churn (independent driver)
    # "explicit dissatisfaction": a churn driver observable ONLY through the
    # cancel-risk INTENT of tickets (not their volume or sentiment). This is what
    # gives LLM intent labels genuine, orthogonal predictive lift.
    c_intent: float = 1.00
    # friction x dissatisfaction interaction: accounts that are BOTH frustrated
    # and voicing cancel-risk churn super-additively. A nonlinearity gradient-
    # boosted trees exploit but a main-effects logistic cannot. Deliberately does
    # NOT involve account size, so size's effect on churn stays linear and the
    # backdoor adjustment cleanly recovers the true adoption effect.
    c_inter: float = 1.35

    # ticket intensity
    base_ticket_rate: float = 0.42
    g_friction: float = 0.55
    g_lowadopt: float = 1.10

    # fraction of tickets whose text is lexically ambiguous vs its true intent
    p_ambiguous: float = 0.18

    gold_eval_size: int = 100


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _standardize(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / (x.std() + 1e-9)


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def _segment_for_seats(seats: int) -> str:
    if seats < 20:
        return "SMB"
    if seats < 100:
        return "Mid-Market"
    return "Enterprise"


def _build_accounts(rng: np.random.Generator, p: GenParams) -> pd.DataFrame:
    n = p.n_accounts
    # Account size: log-normal seat counts -> the confounder.
    seats = np.clip(np.round(rng.lognormal(mean=3.1, sigma=0.9, size=n)), 3, 1000).astype(int)
    log_seats = np.log(seats)
    z_size = _standardize(log_seats)

    # Adoption driven by size (this coupling is what creates confounding).
    adoption_latent = p.a0 + p.a_size * z_size + p.a_noise * rng.standard_normal(n)
    mean_adoption = np.clip(_sigmoid(adoption_latent), 0.02, 0.98)
    threshold = float(np.median(mean_adoption))
    high_adoption = (mean_adoption > threshold).astype(int)

    friction = rng.standard_normal(n)
    # Seen ONLY through cancel-risk intent labels — never through ticket volume
    # or sentiment — so it is orthogonal to the base churn features.
    intent_latent = rng.standard_normal(n)

    # Structural churn probability + realized churn.
    interaction = np.maximum(0.0, friction) * np.maximum(0.0, intent_latent)
    logit_p = (
        p.c0
        + p.c_size * z_size
        + p.c_treat * high_adoption
        + p.c_friction * friction
        + p.c_intent * intent_latent
        + p.c_inter * interaction
    )
    p_churn = _sigmoid(logit_p)
    churned = (rng.random(n) < p_churn).astype(int)

    seg = [_segment_for_seats(int(s)) for s in seats]
    price_per_seat = np.where(z_size > 1, 45, np.where(z_size > 0, 30, 20))
    mrr = np.round(seats * price_per_seat * rng.uniform(0.85, 1.15, n), 2)

    renewal_month = rng.integers(p.feature_window + 2, p.n_months + 1, size=n)
    renewal_date = [ANCHOR_DATE + pd.DateOffset(months=int(m)) for m in renewal_month]

    names = [
        f"{rng.choice(COMPANY_PREFIXES)} {rng.choice(COMPANY_SUFFIXES)}" for _ in range(n)
    ]
    industry = rng.choice(INDUSTRIES, size=n)

    df = pd.DataFrame(
        {
            AccountCols.ID: [f"ACC-{i:04d}" for i in range(n)],
            AccountCols.NAME: names,
            AccountCols.INDUSTRY: industry,
            AccountCols.SEGMENT: seg,
            AccountCols.SEATS: seats,
            AccountCols.MRR: mrr,
            AccountCols.RENEWAL_MONTH: renewal_month,
            "renewal_date": renewal_date,
            AccountCols.TENURE: renewal_month,
            AccountCols.CHURNED: churned,
            # latents kept for the causal ground truth (NOT model features)
            AccountCols.Z_SIZE: z_size,
            AccountCols.FRICTION_LATENT: friction,
            AccountCols.INTENT_LATENT: intent_latent,
            AccountCols.MEAN_ADOPTION: mean_adoption,
            AccountCols.HIGH_ADOPTION: high_adoption,
            "p_churn_true": p_churn,
        }
    )
    return df


def _monthly_and_tickets(
    rng: np.random.Generator, accounts: pd.DataFrame, p: GenParams
) -> tuple[pd.DataFrame, pd.DataFrame]:
    monthly_rows: list[dict] = []
    ticket_rows: list[dict] = []
    ticket_id = 0

    for acct in accounts.itertuples(index=False):
        aid = getattr(acct, AccountCols.ID)
        seats = getattr(acct, AccountCols.SEATS)
        base_adopt = getattr(acct, AccountCols.MEAN_ADOPTION)
        friction = getattr(acct, AccountCols.FRICTION_LATENT)
        high_adopt = getattr(acct, AccountCols.HIGH_ADOPTION)
        churned = getattr(acct, AccountCols.CHURNED)
        # The hidden "explicit dissatisfaction" driver: accounts high on it file
        # more cancel-risk tickets (same volume, same sentiment), so the signal is
        # visible ONLY by classifying ticket intent — giving LLM labels genuine
        # orthogonal lift over raw ticket counts.
        intent_latent = float(getattr(acct, AccountCols.INTENT_LATENT))
        renewal_month = getattr(acct, AccountCols.RENEWAL_MONTH)

        last_month = renewal_month if churned else p.n_months - 1
        last_month = min(last_month, p.n_months - 1)

        for m in range(last_month + 1):
            # Adoption drifts: a gentle secular upward trend, plus a *friction-driven*
            # decline (high-friction accounts slowly disengage). The decline is keyed
            # to the latent friction, NOT to the churn label, so it correlates with
            # churn realistically without leaking the outcome deterministically.
            trend = 0.010 * (m - p.n_months / 2)
            friction_drift = -0.020 * max(0.0, friction) * (m / p.n_months)
            adoption = float(
                np.clip(
                    base_adopt + trend + friction_drift + 0.05 * rng.standard_normal(),
                    0.02,
                    0.98,
                )
            )

            active_seats = int(np.clip(round(seats * (0.4 + 0.6 * adoption)), 1, seats))
            logins = int(rng.poisson(active_seats * (2.0 + 6.0 * adoption)))
            feature_events = int(rng.poisson(active_seats * adoption * 3.0))

            lam = p.base_ticket_rate * np.exp(
                p.g_friction * friction + p.g_lowadopt * (0.5 - adoption)
            )
            n_tickets = int(rng.poisson(np.clip(lam, 0.0, 6.0)))

            month_scores: list[float] = []
            for _ in range(n_tickets):
                intent = _sample_intent(rng, adoption, friction, high_adopt, intent_latent)
                sentiment = _sample_sentiment(rng, intent, friction)
                feature = str(rng.choice(PRODUCT_FEATURES))
                ambiguous = bool(rng.random() < p.p_ambiguous)
                text = generate_ticket_text(rng, intent, sentiment, feature, ambiguous=ambiguous)
                ticket_rows.append(
                    {
                        TicketCols.ID: f"TCK-{ticket_id:06d}",
                        TicketCols.ACCOUNT: aid,
                        TicketCols.MONTH: m,
                        TicketCols.TEXT: text,
                        TicketCols.TRUE_INTENT: intent.value,
                        TicketCols.TRUE_SENTIMENT: sentiment.value,
                        TicketCols.IN_GOLD_EVAL: False,
                    }
                )
                ticket_id += 1
                month_scores.append(SENTIMENT_SCORE[sentiment])

            avg_sent = float(np.mean(month_scores)) if month_scores else np.nan
            # NPS survey fielded quarterly.
            nps = np.nan
            if m % 3 == 0:
                nps = float(
                    np.clip(
                        round(7 + 4 * (adoption - 0.5) - 1.6 * friction + rng.normal(0, 0.6)),
                        0,
                        10,
                    )
                )

            monthly_rows.append(
                {
                    MonthlyCols.ID: aid,
                    MonthlyCols.MONTH: m,
                    MonthlyCols.ACTIVE_SEATS: active_seats,
                    MonthlyCols.ADOPTION: adoption,
                    MonthlyCols.LOGINS: logins,
                    MonthlyCols.FEATURE_EVENTS: feature_events,
                    MonthlyCols.TICKETS: n_tickets,
                    MonthlyCols.AVG_SENTIMENT: avg_sent,
                    MonthlyCols.NPS: nps,
                    MonthlyCols.IS_ACTIVE: 1,
                }
            )

    monthly = pd.DataFrame(monthly_rows)
    tickets = pd.DataFrame(ticket_rows)
    return monthly, tickets


def _sample_intent(
    rng: np.random.Generator,
    adoption: float,
    friction: float,
    high_adopt: int,
    intent_latent: float,
) -> Intent:
    scores = np.array(
        [
            0.2 + 0.5 * friction + 0.4 * (0.5 - adoption) + 2.3 * intent_latent,  # cancel-risk
            0.2 + 1.0 * (adoption - 0.5) - 0.8 * friction + 0.6 * high_adopt,  # expansion
            0.4 + 0.5 * (0.5 - adoption),  # confusion
            0.3 + 0.4 * friction,  # bug-report
        ]
    )
    probs = _softmax(scores)
    idx = rng.choice(4, p=probs)
    return [Intent.CANCEL_RISK, Intent.EXPANSION_SIGNAL, Intent.CONFUSION, Intent.BUG_REPORT][idx]


# Base sentiment mix per intent, before the friction shift.
_SENT_BASE = {
    Intent.CANCEL_RISK: np.array([0.05, 0.25, 0.70]),
    Intent.EXPANSION_SIGNAL: np.array([0.70, 0.25, 0.05]),
    Intent.CONFUSION: np.array([0.10, 0.55, 0.35]),
    Intent.BUG_REPORT: np.array([0.05, 0.35, 0.60]),
}


def _sample_sentiment(rng: np.random.Generator, intent: Intent, friction: float) -> Sentiment:
    base = _SENT_BASE[intent].copy()
    shift = np.clip(0.15 * friction, -0.2, 0.2)  # more friction -> more negative
    base = base + np.array([-shift, 0.0, shift])
    base = np.clip(base, 0.01, None)
    base = base / base.sum()
    idx = rng.choice(3, p=base)
    return [Sentiment.POSITIVE, Sentiment.NEUTRAL, Sentiment.NEGATIVE][idx]


def _select_gold_eval(rng: np.random.Generator, tickets: pd.DataFrame, size: int) -> pd.DataFrame:
    """Freeze a stratified sample of tickets as the hand-label eval set.

    Because tickets are synthetic, the generative ``true_intent`` *is* the gold
    label — equivalent to a perfect human annotator. We still stratify by intent
    so the Cohen's-kappa estimate is not dominated by the majority class.
    """
    per_intent = max(1, size // 4)
    picks: list[int] = []
    for intent in [i.value for i in Intent]:
        pool = tickets.index[tickets[TicketCols.TRUE_INTENT] == intent].to_numpy()
        take = min(per_intent, len(pool))
        if take:
            picks.extend(rng.choice(pool, size=take, replace=False).tolist())
    # Top up to exactly ``size`` from whatever remains.
    remaining = np.setdiff1d(tickets.index.to_numpy(), np.array(picks))
    shortfall = size - len(picks)
    if shortfall > 0 and len(remaining):
        picks.extend(rng.choice(remaining, size=min(shortfall, len(remaining)), replace=False).tolist())
    tickets.loc[picks, TicketCols.IN_GOLD_EVAL] = True
    return tickets


def _ground_truth(accounts: pd.DataFrame, p: GenParams) -> dict:
    z = accounts[AccountCols.Z_SIZE].to_numpy()
    fr = accounts[AccountCols.FRICTION_LATENT].to_numpy()
    il = accounts[AccountCols.INTENT_LATENT].to_numpy()
    interaction = np.maximum(0.0, fr) * np.maximum(0.0, il)
    common = p.c0 + p.c_size * z + p.c_friction * fr + p.c_intent * il + p.c_inter * interaction
    # Exact structural intervention: do(high_adoption = 1) vs do(high_adoption = 0).
    p1 = _sigmoid(common + p.c_treat * 1)
    p0 = _sigmoid(common + p.c_treat * 0)
    true_ate = float(np.mean(p1 - p0))

    high = accounts[AccountCols.HIGH_ADOPTION].to_numpy()
    churn = accounts[AccountCols.CHURNED].to_numpy()
    naive_diff = float(churn[high == 1].mean() - churn[high == 0].mean())

    return {
        "params": asdict(p),
        "confounder": "account size (z_size = standardized log seats)",
        "treatment": "high_adoption (feature adoption above the median)",
        "outcome": "churned",
        "true_ate_risk_diff": true_ate,
        "naive_diff_risk_diff": naive_diff,
        "churn_base_rate": float(churn.mean()),
        "n_accounts": int(len(accounts)),
        "note": (
            "true_ate_risk_diff is the exact interventional effect from the "
            "structural churn equation. naive_diff is confounded by account size "
            "and overstates adoption's protective effect; backdoor adjustment on "
            "z_size should recover the true effect."
        ),
    }


def generate(params: GenParams | None = None, write: bool = True) -> dict:
    """Generate the full dataset. Returns the frames + ground truth dict."""
    import json

    p = params or GenParams()
    rng = np.random.default_rng(p.seed)

    accounts = _build_accounts(rng, p)
    monthly, tickets = _monthly_and_tickets(rng, accounts, p)
    tickets = _select_gold_eval(rng, tickets, min(p.gold_eval_size, len(tickets)))
    gt = _ground_truth(accounts, p)

    if write:
        ensure_dirs()
        accounts.to_parquet(ACCOUNTS_PARQUET, index=False)
        monthly.to_parquet(MONTHLY_PARQUET, index=False)
        tickets.to_parquet(TICKETS_PARQUET, index=False)
        GROUND_TRUTH_JSON.write_text(json.dumps(gt, indent=2), encoding="utf-8")

    return {"accounts": accounts, "monthly": monthly, "tickets": tickets, "ground_truth": gt}
