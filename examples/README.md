# Sati Ecosystem Integration Examples

This directory contains examples that bridge **StructuralTime-Core** with the
**Sati Ecosystem** — a metacognitive layer for Hermes Agent.

## Files

| File | Description |
|------|-------------|
| `sati_adapter.py` | `SatiAdapter` class — maps Sati's 9-dim K-vector + THM metrics to STC's 3-dim K-state (E_K, dK_dt, γ) |
| `sati_demo.py` | Full cross-validation pipeline — loads real Sati SQLite data, runs STC analytics, compares regimes |

## How It Works

```
Sati SQLite (9-dim)  →  SatiAdapter  →  STC 3-dim K-state
                                         ├── QuarticPotential → equilibria
                                         ├── HybridRegimeClustering → regime
                                         └── GradientFlowIntegrator → trajectory
```

## Run

```bash
pip install -e /f/_Ai/structural-time-core  # if not installed
python examples/sati_demo.py
```

## Results (2026-05-29)

- **Agreement:** Sati vs STC Theory = 20.3%, Sati vs STC Hybrid = 24.1%
- **Potential landscape:** Mean E_K≈0.39 near unstable equilibrium (bifurcation point)
- **Key insight:** Sati conservative (Active/Frozen), STC aggressive (Turbulent/Critical) — complement each other

Full documentation: `F:\_Ai\sati-ecosystem\architecture\sati-stc-integration.md`
