"""CLI: ``python -m intents`` labels tickets, evaluates, and builds features."""

from __future__ import annotations

import argparse

from common.config import get_settings
from intents.eval import evaluate_labeler
from intents.labeler import build_account_intent_features, label_all


def main() -> None:
    parser = argparse.ArgumentParser(description="Label tickets + evaluate + build features.")
    parser.add_argument("--limit", type=int, default=None, help="label only the first N tickets")
    args = parser.parse_args()

    provider = get_settings().effective_provider
    print(f"Labeling tickets (provider={provider}) ...")
    labels = label_all(limit=args.limit)
    print(f"  labeled {len(labels)} tickets")

    feats = build_account_intent_features()
    print(f"  built intent features for {len(feats)} accounts")

    ev = evaluate_labeler()
    print(f"Gold eval set: {ev['gold_set_size']} tickets")
    print(
        f"  intent:    accuracy={ev['intent']['accuracy']:.3f}  "
        f"kappa={ev['intent']['cohen_kappa']:.3f}  macro-F1={ev['intent']['macro_f1']:.3f}"
    )
    print(
        f"  sentiment: accuracy={ev['sentiment']['accuracy']:.3f}  "
        f"kappa={ev['sentiment']['cohen_kappa']:.3f}  macro-F1={ev['sentiment']['macro_f1']:.3f}"
    )


if __name__ == "__main__":
    main()
