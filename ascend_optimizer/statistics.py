"""Paired statistics; reject invalid evidence instead of treating it as a speedup."""

import math
import random
import statistics


def geometric_mean(values):
    values = list(values)
    if not values or any(not math.isfinite(x) or x <= 0 for x in values):
        raise ValueError("Timings/speedups must be finite and positive")
    return math.exp(statistics.fmean(math.log(x) for x in values))


def summarize(cases, policy):
    if not cases:
        raise ValueError("No benchmark evidence")
    groups, detail = [], []
    for case in cases:
        pairs = case["pairs"]
        if len(pairs) < 5:
            raise ValueError("At least five paired rounds required")
        ratios = [
            p["baseline_us"] / p["candidate_us"]
            for p in pairs
            if math.isfinite(p["baseline_us"])
            and math.isfinite(p["candidate_us"])
            and p["baseline_us"] > 0
            and p["candidate_us"] > 0
        ]
        if len(ratios) != len(pairs):
            raise ValueError("Invalid timing evidence")
        med = statistics.median(ratios)
        groups.append(ratios)
        detail.append(
            {
                "case": case["case"],
                "speedup": med,
                "baseline_us": statistics.median(p["baseline_us"] for p in pairs),
                "candidate_us": statistics.median(p["candidate_us"] for p in pairs),
            }
        )
    observed = geometric_mean(d["speedup"] for d in detail)
    rng = random.Random(20261015)
    boot = sorted(
        geometric_mean(statistics.median(rng.choices(g, k=len(g))) for g in groups)
        for _ in range(policy["bootstrap_samples"])
    )
    low = boot[int(0.025 * (len(boot) - 1))]
    high = boot[int(0.975 * (len(boot) - 1))]
    improves = low >= policy["min_speedup"]
    no_regression = min(d["speedup"] for d in detail) >= policy["min_case_speedup"]
    return {
        "speedup_geomean": observed,
        "bootstrap_95_interval": [low, high],
        "per_case": detail,
        "accepted": improves and no_regression,
        "reason": "accepted"
        if improves and no_regression
        else "case_regression"
        if not no_regression
        else "improvement_not_established",
        "method": "ABBA/BAAB paired NPU events; equal-weight cases; median ratios; bootstrap exploratory interval",
    }
