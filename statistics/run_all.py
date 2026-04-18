#!/usr/bin/env python3
"""
Run all statistical analyses and save outputs.

Usage:
    python statistics/run_all.py
    python statistics/run_all.py --outdir statistics/
"""

import argparse
import os
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="statistics/")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print("=" * 80)
    print("  EURORAD STATISTICS (Nominal Diagnostic Accuracy)")
    print("=" * 80)
    from eurorad_stats import main as eurorad_main
    sys.argv = ["eurorad_stats.py", "--output",
                os.path.join(args.outdir, "eurorad_results.csv")]
    eurorad_main()

    print("\n\n")
    print("=" * 80)
    print("  NMED STATISTICS (Ordinal Clinical Judgment)")
    print("=" * 80)
    from nmed_stats import main as nmed_main
    sys.argv = ["nmed_stats.py", "--output",
                os.path.join(args.outdir, "nmed_results.csv")]
    nmed_main()

    print("\n\n" + "=" * 80)
    print("  All analyses complete.")
    print(f"  Results saved to: {args.outdir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
