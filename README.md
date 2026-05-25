# StructuralTime-Core v0.1 (MVP)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20371010.svg)](https://doi.org/10.5281/zenodo.20371010)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhammawatthumpra-coder/structural-time-core/blob/main/QuickStart.ipynb)

A modular, domain-agnostic Python library implementing the **Core Structural Time** framework. It bridges abstract logical constraints (Level A Ontology) with continuous information-dynamic equations (Level B Dynamics) and behavioral regime classification.

This framework allows researchers across different fields (e.g., Artificial Intelligence, Sociology, Biology, Astrophysics) to map domain-specific raw telemetry into information parameters, helping detect phase transitions and system stability.

> For the Thai version of this document, see [README_TH.md](README_TH.md).

---

## Epistemic Disclaimer

**Important Notice for Researchers:**

This library packages two distinct layers of the **Structural Time Framework (STF)**:
* **Level A (Ontology Engine):** Models abstract, timeless logical state spaces and checks structural compatibility.
* **Level B (Dynamics Engine):** Models continuous numerical simulations (such as quartic potential wells) and performs empirical clustering.

These layers operate on **different mathematical objects** ($K$ as a stochastic kernel in Ontology vs. $K$ as a vector in Dynamics). They are complementary conceptual tools, but are **epistemically independent**:
* Running simulations in the Dynamics Engine **does not prove or validate** the logical conjectures of the Ontology Engine.
* The Ontology Engine's constraints **do not mathematically validate** the specific numerical design choices (such as the quartic potential formulation or the experienced time metric $T(K)$) in the Dynamics Engine.
* The `LogicalCompatibilityChecker` does not mathematically prove the *Asymmetry Conjecture*; it only checks if an empirical trajectory is compatible with the conjecture's constraints.
* The `SociologyAdapter` (now relocated to `examples/sociology_adapter_demo.py`) is purely illustrative. The mapping of sociological variables to $K$-state coordinates is conceptual and metaphorical, not a quantitative prediction tool.

---

## 1. Core Modules

The library is organized into four main functional modules:

*   **`adapters` (Domain Adapter Layer):** Maps domain-specific raw data (e.g., Transformer cross-layer similarity, Neural Network training telemetry) to standardized $K$-state vectors.
*   **`ontology` (Ontology Engine - Level A):** Manages the Platonist State Space ($S$), Valid Configuration Set ($V$), and checks compatibility under the **Asymmetry Conjecture** along behavioral trajectories.
*   **`dynamics` (Dynamics Engine - Level B):** Computes Quartic Potential ($\mathcal{F}(K)$) multi-stability, sweeps parameter bifurcations, simulates gradient flow trajectories using Runge-Kutta 4th Order (RK4), and calculates Bounded Temporal Density ($T(K)$).
*   **`analytics` (Regime Clustering & Visualization):** Classifies trajectories into five theoretical regimes (Active, Critical, Turbulent, Decayed, Frozen) using KMeans and the decay rate parameter ($\gamma$) to distinguish Frozen and Decayed states.

---

## 2. Installation & Quick Start

You can install this library locally in editable mode (recommended for development) or directly via pip:

```bash
# Clone the repository and install in editable mode
git clone https://github.com/dhammawatthumpra-coder/structural-time-core.git
cd structural-time-core
pip install -e .
```

Or install it directly from GitHub:
```bash
pip install git+https://github.com/dhammawatthumpra-coder/structural-time-core.git
```

Once installed, import the modules in Python:

```python
import numpy as np
from structural_time_core import (
    TransformerAdapter,
    NeuralNetworkTelemetryAdapter,
    LogicalCompatibilityChecker,
    QuarticPotentialSolver,
    GradientFlowIntegrator,
    HybridRegimeClustering
)
```

---

## 3. Usage Examples

### 3.1 Logical Compatibility & Asymmetry Conjecture (Level A)
Filters out symmetric behavior under high system complexity:

```python
checker = LogicalCompatibilityChecker(complexity_threshold=0.6)

# Symmetric, flat/stagnant trajectory
flat_K = np.array([0.5, 0.5, 0.5, 0.5])

# In a complex environment (complexity = 0.8 > 0.6), symmetric states are rejected
is_valid = checker.is_compatible(flat_K, system_complexity=0.8)
print("Is compatible:", is_valid)  # Output: False (Logically Excluded)
```

### 3.2 Simulating Structural Time Dynamics (Level B)
Simulates state updates within a double-well potential and calculates experiences of time density $T(K)$:

```python
# F(K) = K^4 - 2*K^2 (Stable wells at K = -1 and 1, unstable peak at 0)
potential = QuarticPotentialSolver(a=1.0, b=0.0, c=-2.0, d=0.0)
integrator = GradientFlowIntegrator(gamma=0.1, dt=0.01)

# Run simulation starting from K = 0.5
K = 0.5
for step in range(100):
    # Use RK4 solver for numerical stability (prevents exploding gradients)
    K = integrator.step_rk4(K, potential.compute_dF_dK, u=0.0)
    print(f"Step {step}: K = {K:.4f}")
```

### 3.3 Hybrid Regime Clustering (Analytics)
Groups trajectory snapshots and distinguishes Frozen from Decayed using the structural decay coefficient ($\gamma$):

```python
clustering = HybridRegimeClustering()

# Snapshots format: [E_K, dK_dt, gamma]
# 1. Low energy/velocity, high decay rate -> Decayed
# 2. Low energy/velocity, low decay rate -> Frozen
trajectory_data = np.array([
    [0.2, 0.05, 0.75],  # Sample 1
    [0.1, 0.02, 0.12],  # Sample 2
])

regimes = clustering.fit_predict_regimes(trajectory_data)
print("Regimes mapped:", regimes) 
# Output: ['Decayed', 'Frozen']
```

### 3.4 Deep Learning Telemetry Adaptation
Maps deep learning training telemetry to standard $K$-state representation:

```python
from structural_time_core import NeuralNetworkTelemetryAdapter

# Max expected weight norm and gradient norm for scaling
adapter = NeuralNetworkTelemetryAdapter(max_weight_norm=100.0, max_grad_norm=10.0)

raw_telemetry = {
    'train_loss': 0.05,
    'val_loss': 1.62,
    'val_accuracy': 0.35,
    'weight_norm': 74.2,
    'gradient_norm': 0.12
}

K = adapter.map_to_K(raw_telemetry)
print("Complexity, Stability, Error Rate:", K)
# Output: [0.742, 0.35, 0.618]
```

---

## 4. Running Demos and Visualization

The package contains built-in simulation demos that plot dynamics, experienced time, and temporal regimes:

```bash
# 1. Basic potential well and trajectory integration demo
python examples/simulation_demo.py

# 2. Deep learning telemetry mapping demo (Grokking & GAN Mode Collapse)
python examples/nn_telemetry_demo.py
```

This generates visualization plots in the `examples/` directory (`potential_well.png`, `trajectory.png`, `regime_clustering.png`, `nn_grokking.png`, `nn_mode_collapse.png`, and `nn_clustering_3d.png`).

---

## 5. Running Unit Tests

Verify package functionality and mathematical constraints:

```powershell
python -m unittest discover -s tests -p "test_*.py" -v
```

---

## 6. Citation & Archiving

If you use this library in your research, please cite the software using the metadata in the `CITATION.cff` file:

*   **DOI Allocation:** When linked to a **Zenodo** repository, GitHub releases automatically archive and assign a permanent **Digital Object Identifier (DOI)** for academic reference.
