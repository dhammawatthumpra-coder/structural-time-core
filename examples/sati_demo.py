#!/usr/bin/env python3
"""
Sati × StructuralTime-Core Integration Demo.

Loads real Sati K-vector data from SQLite, maps through SatiAdapter,
and runs STC's QuarticPotential + GradientFlow + HybridRegimeClustering.
Compares STC regimes with Sati's own regime labels.

Usage:
    pip install -e /f/_Ai/structural-time-core  # if not already
    python examples/sati_demo.py
"""

import sys, os, sqlite3
from collections import Counter

# ── Path setup ──────────────────────────────────────────────
# STC root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
# Sati core (for THMMonitor constants reference)
SATI_ROOT = r'F:\_Ai\sati-ecosystem'
sys.path.insert(0, SATI_ROOT)

# ── Imports ─────────────────────────────────────────────────
import numpy as np
from examples.sati_adapter import SatiAdapter
from analytics.clustering import HybridRegimeClustering
from dynamics.potential import QuarticPotentialSolver
from dynamics.integrator import GradientFlowIntegrator
from analytics.visualization import VisualizationAPI


def load_sati_data() -> list:
    """Load all interactive samples from both profiles."""
    hermes_home = os.path.expanduser(r'~/AppData/Local/hermes')
    rows = []

    for name in ['default', 'lmstudio']:
        db = os.path.join(hermes_home, 'sati_history.db') if name == 'default' \
            else os.path.join(hermes_home, 'profiles', name, 'sati_history.db')
        if not os.path.exists(db):
            continue
        conn = sqlite3.connect(db)
        cur = conn.execute("""
            SELECT dim1,dim2,dim3,dim4,dim5,dim6,dim7,dim8,dim9,
                   C_K,gamma,T_ops,rate_op,regime,created_at
            FROM k_vector_samples WHERE session_type='interactive'
            ORDER BY id
        """)
        for r in cur.fetchall():
            rows.append({
                'profile': name,
                'k_vector': [r[0] or 0.0 for _ in range(9)],
                'k_vector_raw': [r[d] or 0.0 for d in range(9)],
                'C_K': r[9],
                'gamma': r[10],
                'T_ops': r[11],
                'rate_op': r[12] or 0.0,
                'regime_sati': r[13],
                'timestamp': r[14][:16] if r[14] else '',
            })
        conn.close()

    return rows


def main():
    print("=" * 65)
    print("  SATI × STRUCTURAL TIME CORE — CROSS-VALIDATION")
    print("=" * 65)

    # ── 1. Load data ────────────────────────────────────────
    all_rows = load_sati_data()
    interactive = [r for r in all_rows if isinstance(r['regime_sati'], str)]
    total = len(interactive)
    print(f"\n📦 Loaded {total} interactive samples")

    # Filter to rows with real metrics (C_K or rate_op available)
    valid = [r for r in interactive if r['C_K'] is not None or r['rate_op'] > 0]
    print(f"📊 Valid for STC analysis: {len(valid)}/{total}")

    # ── 2. Map through SatiAdapter ──────────────────────────
    adapter = SatiAdapter()
    stc_samples = []
    for r in valid:
        mapped = adapter.map_from_sqlite_row(
            r['k_vector_raw'], r['C_K'], r['gamma'],
            r['rate_op'], r['T_ops'], r['regime_sati']
        )
        mapped['profile'] = r['profile']
        mapped['timestamp'] = r['timestamp']
        stc_samples.append(mapped)

    # ── 3. Run STC HybridRegimeClustering ────────────────────
    print("\n——— STC ANALYTICS ———")

    # Build trajectory matrix [E_K, dK_dt, gamma]
    X = np.array([[s['E_K'], s['dK_dt'], s['gamma']] for s in stc_samples])

    # 3a. Theory-based classification (no fitting)
    stc_theory = HybridRegimeClustering()
    theory_labels = [stc_theory.classify_by_theory(s['E_K'], s['dK_dt'], s['gamma'])
                     for s in stc_samples]

    # 3b. Hybrid KMeans classification
    stc_hybrid = HybridRegimeClustering()
    stc_hybrid.fit(X)
    hybrid_labels = stc_hybrid.fit_predict_regimes(X)

    # ── 4. Sati regime distribution (for comparison) ─────────
    sati_regimes = Counter(s['sati_regime'] for s in stc_samples)
    print(f"\n{'Regime':>12} | {'Sati':>8} | {'STC Theory':>12} | {'STC Hybrid':>12}")
    print("-" * 50)
    all_regimes = sorted(set(list(sati_regimes.keys()) +
                             list(set(theory_labels)) +
                             list(set(hybrid_labels))))
    for reg in all_regimes:
        sc = sati_regimes.get(reg, 0)
        tc = sum(1 for l in theory_labels if l == reg)
        hc = sum(1 for l in hybrid_labels if l == reg)
        print(f"{reg:>12} | {sc:>8} | {tc:>12} | {hc:>12}")

    # ── 5. Agreement analysis ────────────────────────────────
    print(f"\n——— CROSS-VALIDATION ———")
    # Can only compare rows where both have meaningful labels
    comparison_points = [(s, tl, hl) for s, tl, hl in
                         zip(stc_samples, theory_labels, hybrid_labels)]

    # Agreement: Sati vs STC Theory
    agree_theory = sum(1 for s, tl, _ in comparison_points
                       if s['sati_regime'] == tl)
    # Agreement: Sati vs STC Hybrid
    agree_hybrid = sum(1 for s, _, hl in comparison_points
                       if s['sati_regime'] == hl)

    n = len(comparison_points)
    print(f"\nSati vs STC Theory:  {agree_theory}/{n} "
          f"({100*agree_theory/n:.1f}%) agreement")
    print(f"Sati vs STC Hybrid: {agree_hybrid}/{n} "
          f"({100*agree_hybrid/n:.1f}%) agreement")

    # Disagreement breakdown
    print(f"\nDisagreement detail (Sati vs STC Theory, top 10):")
    disagrees = [(s, tl) for s, tl, _ in comparison_points
                 if s['sati_regime'] != tl]
    for s, tl in disagrees[:10]:
        sr = s['sati_regime']
        print(f"  {s['profile']:>6} | Sati={sr:>10} | STC={tl:>10} | "
              f"E_K={s['E_K']:.2f} dK/dt={s['dK_dt']:.2f} γ={s['gamma']:.3f}")

    # ── 6. Case studies from the bug analysis ────────────────
    print(f"\n——— CASE STUDIES ———")

    # Find specific samples matching our bug-analysis cases
    for desc, ck_min, ck_max, rop_min, rop_max, n_act in [
        ("#01 idle (K=[0])",        0, 0.1,   0, 0.01, 0),
        ("#03 C_K=11.5 (extreme)", 10, 12,    4, 6,    1),
        ("#10 T_ops=0.59 (spike)",  0, 0.01,  0, 0.02, 0),
    ]:
        matches = [s for s in stc_samples
                   if (s['raw']['C_K'] is not None
                       and ck_min <= s['raw']['C_K'] <= ck_max
                       and rop_min <= s['raw']['rate_op'] <= rop_max)]
        if matches:
            s = matches[0]
            sr = s['sati_regime']
            tl = stc_theory.classify_by_theory(s['E_K'], s['dK_dt'], s['gamma'])
            ck = s['raw']['C_K']
            print(f"\n  {desc}:")
            print(f"    Sati says:  {sr}")
            print(f"    STC Theory: {tl}")
            print(f"    E_K={s['E_K']:.3f}  dK/dt={s['dK_dt']:.3f}  γ={s['gamma']:.3f}")
            print(f"    (raw C_K={ck:.2f}  rate_op={s['raw']['rate_op']:.4f})")

    # ── 7. Quartic Potential Analysis ────────────────────────
    print(f"\n——— POTENTIAL LANDSCAPE ———")
    # Fit quartic potential to the actual data distribution
    # E_K is our coordinate; we want F(E_K) to have wells at stable regimes
    # Use default parameters: F(K) = K^4 - K^2
    potential = QuarticPotentialSolver(a=1.0, b=0.0, c=-1.0, d=0.0)
    equilibria = potential.find_equilibria()

    print(f"Quartic potential: F(K) = K⁴ - K²")
    print(f"Equilibria:")
    for eq in equilibria:
        marker = "🟢" if eq['type'] == 'stable' else "🔴"
        print(f"  {marker} K={eq['K']:.3f} ({eq['type']}) F={eq['F']:.3f}")

    # Current average E_K
    avg_E_K = np.mean([s['E_K'] for s in stc_samples])
    print(f"\n  Current mean E_K = {avg_E_K:.3f}")
    dF = potential.compute_dF_dK(avg_E_K)
    d2F = potential.compute_d2F_dK2(avg_E_K)
    print(f"  dF/dK({avg_E_K:.3f}) = {dF:.3f} "
          f"→ {'stable' if dF > 0 else 'unstable' if dF < 0 else 'critical'}")

    # ── 8. Summary ────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  SUMMARY")
    print(f"{'='*65}")
    print(f"  Sati regime classification:     {dict(sati_regimes)}")
    print(f"  STC Theory classification:      {dict(Counter(theory_labels))}")
    print(f"  STC Hybrid classification:      {dict(Counter(hybrid_labels))}")
    print(f"  Sati vs STC Theory agreement:   {100*agree_theory/n:.1f}%")
    print(f"  Sati vs STC Hybrid agreement:   {100*agree_hybrid/n:.1f}%")
    eq_str = ', '.join(f"K={e['K']:.2f}({e['type']})" for e in equilibria)
    print(f"  Quartic equilibria:             [{eq_str}]")
    print()


if __name__ == '__main__':
    main()
