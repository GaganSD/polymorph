"""SOTA-claim tool: is the trained LaMR pruner significantly better than
keep-severity on the held-out SEMANTIC needles (the discriminating class)?

Runs both methods on a saved held-out triple set, reports per-class survival
across a drop sweep, and runs the paired McNemar test on the semantic subset
(structural needles saturate under the floor and don't discriminate). A SOTA
claim requires: semantic survival > keep-severity AND McNemar p < 0.05.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .build_heldout import load_triples
from .methods import KeepSeverityHeuristic, LaMRMethod
from .survival import answer_survives, compression_ratio, mcnemar, survival_vector


def _mean_ratio(method, triples, r) -> float:
    rs = [compression_ratio(t.text, method.compress(t.text, r)) for t in triples]
    return sum(rs) / len(rs) if rs else 0.0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="lamr-compare", description=__doc__)
    p.add_argument("--triples", required=True, type=Path)
    p.add_argument("--lamr-ckpt", required=True, type=Path)
    p.add_argument("--floor", action="store_true", help="apply the structural decode floor")
    p.add_argument("--drop-rates", default="0.4,0.6,0.8")
    args = p.parse_args(argv)

    triples = load_triples(args.triples)
    semantic = [t for t in triples if t.fact_type.startswith("semantic:")]
    rates = [float(x) for x in args.drop_rates.split(",") if x.strip()]

    ks = KeepSeverityHeuristic()
    lamr = LaMRMethod(ckpt=args.lamr_ckpt, floor=args.floor)
    ok, reason = lamr.available()
    if not ok:
        print(f"LaMR unavailable: {reason}")
        return 1

    print(f"held-out triples: {len(triples)} total, {len(semantic)} semantic")
    print(f"model: {args.lamr_ckpt}  floor={args.floor}\n")
    print(f"{'rate':>5} | {'keepsev%':>8} {'ratio':>6} | {'lamr%':>7} {'ratio':>6} | "
          f"{'better':>6} {'worse':>5} {'p':>8}  verdict")
    print("-" * 78)

    any_win = False
    for r in rates:
        ks_vec = survival_vector(ks, semantic, r)
        lamr_vec = survival_vector(lamr, semantic, r)
        ks_s = 100.0 * sum(ks_vec) / len(ks_vec) if ks_vec else 0.0
        lamr_s = 100.0 * sum(lamr_vec) / len(lamr_vec) if lamr_vec else 0.0
        m = mcnemar(lamr_vec, ks_vec)  # A=lamr, B=keepsev
        ks_ratio = _mean_ratio(ks, semantic, r)
        lamr_ratio = _mean_ratio(lamr, semantic, r)
        sig = m["p_value"] < 0.05
        win = lamr_s > ks_s and sig
        any_win = any_win or win
        verdict = "SOTA✓" if win else ("(sig but worse)" if sig and lamr_s < ks_s else "ns")
        print(f"{r:>5.2f} | {ks_s:>7.1f}% {ks_ratio:>5.2f}x | {lamr_s:>6.1f}% {lamr_ratio:>5.2f}x | "
              f"{int(m['b10_a_better']):>6} {int(m['b01_a_worse']):>5} {m['p_value']:>8.4f}  {verdict}")

    print()
    print("SOTA on semantic needles: " + (
        "YES — beats keep-severity, McNemar-significant at >=1 rate." if any_win
        else "NOT YET — no rate where LaMR significantly beats keep-severity."))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
