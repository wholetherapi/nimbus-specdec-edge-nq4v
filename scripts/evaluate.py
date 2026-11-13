"""Evaluate a decoder checkpoint on the held-out split."""
import argparse
import json

import yaml

from src.specdec.scheduler import LatencyAwareScheduler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    scheduler = LatencyAwareScheduler(cfg["latency_lookup"], budget_ms=24.0)
    depth = scheduler.choose()
    report = {"split": args.split, "draft_depth": depth, "bleu": None}
    print(json.dumps(report))


if __name__ == "__main__":
    main()
