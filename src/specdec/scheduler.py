"""Latency-aware draft-depth scheduler."""
from __future__ import annotations

import yaml


class LatencyAwareScheduler:
    """Select draft depth from a measured per-depth latency table.

    The scheduler keeps the expected wall-clock cost of a speculation round
    under ``budget_ms`` while preferring deeper drafts whenever they fit.
    """

    def __init__(self, lookup_path: str, budget_ms: float = 24.0):
        with open(lookup_path, "r", encoding="utf-8") as handle:
            table = yaml.safe_load(handle)
        self.costs = {int(k): float(v) for k, v in table["latency_ms"].items()}
        self.acceptance = {int(k): float(v) for k, v in table["acceptance"].items()}
        self.budget_ms = budget_ms

    def expected_gain(self, depth: int) -> float:
        cost = self.costs[depth]
        accepted = self.acceptance[depth]
        return accepted / cost

    def choose(self, remaining_ms: float | None = None) -> int:
        budget = self.budget_ms if remaining_ms is None else remaining_ms
        feasible = [d for d, c in self.costs.items() if c <= budget]
        if not feasible:
            return min(self.costs)
        return max(feasible, key=self.expected_gain)
