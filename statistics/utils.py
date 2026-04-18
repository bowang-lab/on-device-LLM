"""
Shared utilities for statistical analyses.
"""

import re
import unicodedata
import numpy as np
from scipy.stats import chi2


# ─── Text normalization ──────────────────────────────────────────────────────

def norm_text(s: str) -> str:
    t = s or ""
    try:
        t = t.encode("cp1252").decode("utf-8")
    except (UnicodeDecodeError, UnicodeEncodeError):
        pass
    t = unicodedata.normalize("NFKC", t)
    t = t.replace("\u2013", "-").replace("\u2014", "-")
    t = re.sub(r"[^\w\s-]", "", t)
    t = " ".join(t.strip().split())
    return t.lower()


# ─── Wilson Score CI ─────────────────────────────────────────────────────────

def wilson_ci(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0, 0.0
    p_hat = k / n
    denom = 1 + z**2 / n
    centre = (p_hat + z**2 / (2 * n)) / denom
    margin = z * np.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n)) / n) / denom
    return p_hat, max(0.0, centre - margin), min(1.0, centre + margin)


def fmt_pct(k, n):
    p, lo, hi = wilson_ci(k, n)
    return f"{p*100:.1f}% ({lo*100:.1f}--{hi*100:.1f})"


# ─── McNemar's test ──────────────────────────────────────────────────────────

def mcnemar_test(correct_a, correct_b):
    assert len(correct_a) == len(correct_b)
    b = sum(a and not bb for a, bb in zip(correct_a, correct_b))
    c = sum(not a and bb for a, bb in zip(correct_a, correct_b))
    if b + c == 0:
        return 1.0, 0.0
    stat = (abs(b - c) - 1) ** 2 / (b + c)
    p_val = 1 - chi2.cdf(stat, df=1)
    # Odds ratio (effect size)
    odds_ratio = b / c if c > 0 else float("inf")
    return p_val, odds_ratio


# ─── Holm-Bonferroni correction ──────────────────────────────────────────────

def holm_bonferroni(p_values):
    """
    Apply Holm-Bonferroni step-down correction.
    Input: list of (label, p_value) tuples.
    Returns: list of (label, raw_p, adjusted_p, significant) tuples.
    """
    n = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1][1])
    adjusted = [None] * n
    max_p = 0.0
    for rank, (orig_idx, (label, raw_p)) in enumerate(indexed):
        adj_p = min(1.0, raw_p * (n - rank))
        adj_p = max(adj_p, max_p)  # enforce monotonicity
        max_p = adj_p
        adjusted[orig_idx] = (label, raw_p, adj_p, adj_p < 0.05)
    return adjusted


# ─── Fleiss' Kappa ───────────────────────────────────────────────────────────

def fleiss_kappa(table):
    N, k = table.shape
    n = table.sum(axis=1)[0]
    if n <= 1:
        return float("nan")
    p_j = table.sum(axis=0) / (N * n)
    P_i = (np.sum(table**2, axis=1) - n) / (n * (n - 1))
    P_bar = np.mean(P_i)
    P_e = np.sum(p_j**2)
    if P_e == 1.0:
        return 1.0
    return (P_bar - P_e) / (1 - P_e)


# ─── Bootstrap CI ────────────────────────────────────────────────────────────

def bootstrap_ci(data, stat_fn, n_boot=2000, ci=95, seed=42):
    """
    Bootstrap CI for an arbitrary statistic.
    data: array-like or tuple of arrays passed to stat_fn.
    stat_fn: callable returning a scalar.
    Returns: (point_estimate, lo, hi)
    """
    rng = np.random.RandomState(seed)
    point = stat_fn(data)

    if isinstance(data, tuple):
        n = len(data[0])
        boot_stats = []
        for _ in range(n_boot):
            idx = rng.randint(0, n, size=n)
            sample = tuple(np.array(d)[idx] for d in data)
            boot_stats.append(stat_fn(sample))
    else:
        n = len(data)
        boot_stats = []
        for _ in range(n_boot):
            idx = rng.randint(0, n, size=n)
            boot_stats.append(stat_fn(np.array(data)[idx]))

    alpha = (100 - ci) / 2
    lo = np.percentile(boot_stats, alpha)
    hi = np.percentile(boot_stats, 100 - alpha)
    return point, lo, hi


# ─── Formatting helpers ──────────────────────────────────────────────────────

def fmt_p(p):
    if p < 0.001:
        return "<0.001"
    elif p < 0.01:
        return f"{p:.3f}"
    else:
        return f"{p:.3f}"


def fmt_kappa(k, lo=None, hi=None):
    if lo is not None and hi is not None:
        return f"{k:.3f} ({lo:.3f}--{hi:.3f})"
    return f"{k:.3f}"


def print_section(title):
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")
