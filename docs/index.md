# StructuralTime-Core

Welcome to the documentation for **`structural-time-core`**, a modular and domain-agnostic Python library designed to implement the Core Structural Time Ontology and Dynamics framework.

This library acts as an "interdisciplinary mathematical bridge" for researchers, facilitating the mapping of raw system telemetry data into information-dynamic variables to verify logical stability, evaluateexperienced temporal densities, and detect behavioral regimes (Active, Critical, Turbulent, Decayed, Frozen).

---

## Architecture Overview

The codebase is strictly separated into Level A (timeless logical structures) and Level B (computational perception/dynamics) according to the core framework tenets:

```
                  +-------------------------------+
                  |      Domain Adapter Layer     |
                  |  (Transformer, Sociology, etc)|
                  +---------------+---------------+
                                  |
                                  v
                  +---------------+---------------+
                  |  Ontology Engine (Level A)    |
                  |  - State Space / Constraints  |
                  |  - Asymmetry Conjecture Check |
                  +---------------+---------------+
                                  |
                                  v
                  +---------------+---------------+
                  |   Dynamics Engine (Level B)   |
                  |  - Quartic Potential Wells    |
                  |  - RK4 Integrator & T(K) calc |
                  +---------------+---------------+
                                  |
                                  v
                  +---------------+---------------+
                  |  Regime Detection & Analytics |
                  |  - 5-Regime Clustering        |
                  |  - Phase Space Visualizations |
                  +-------------------------------+
```

---

## Key Theoretical Tenets Implemented

1.  **Asymmetry Conjecture (Level A):** Complex systems reject perfectly symmetric operators ($K$). Trajectories that lack structural asymmetry (stagnant/derivative zero states) are flagged as logically incompatible.
2.  **Double-Well Potential (Level B):** Free energy dynamics simulated over a quartic potential: $\mathcal{F}(K) = aK^4 + bK^3 + cK^2 + dK$ (where $a > 0$ guarantees bounded energy).
3.  **Runge-Kutta 4 (RK4) Solver:** Used in continuous gradient flow integrations ($dK/dt$) to prevent numeric divergence (exploding gradients) near phase transition (bifurcation) boundaries.
4.  **$\gamma$-Guided Regime Separation:** Distinguishes **Frozen** (low energy, low velocity, low decay $\gamma$) from **Decayed** (low energy, low velocity, high decay $\gamma$) states on the 3D cognitive manifold.

---

## Installation

Install via pip in editable mode:
```bash
git clone https://github.com/dhammawatthumpra-coder/structural-time-core.git
cd structural-time-core
pip install -e .
```

Or install directly from GitHub:
```bash
pip install git+https://github.com/dhammawatthumpra-coder/structural-time-core.git
```
