# API Reference Guide

This document details the main classes and components exposed by the `structural-time-core` package.

---

## 1. Domain Adapters (`adapters`)

Adapters are responsible for mapping domain-specific telemetry to a standardized 3-dimensional $K$-state vector: `[complexity, stability, decay_or_error]`.

### `TransformerAdapter`
Maps attention/similarity profiles and loss telemetry in Transformer models:
```python
from structural_time_core.adapters import TransformerAdapter

adapter = TransformerAdapter(n_layers=12)
raw_telemetry = {
    'cross_layer_similarity': [0.91, 0.88, 0.90],
    'prediction_error_rates': [0.12, 0.15],
    'sequence_age': 20,
    'max_sequence_len': 1024
}
K_state = adapter.map_to_K(raw_telemetry)
```

### `SociologyAdapter`
Maps social media or demographic generational interactions:
```python
from structural_time_core.adapters import SociologyAdapter

adapter = SociologyAdapter()
raw_data = {
    'info_frequency': 0.7,
    'social_pressure': 0.8,
    'generation_sizes': [1000, 1500, 800]
}
K_state = adapter.map_to_K(raw_data)
noise = adapter.estimate_noise(raw_data)  # Estimated xi
```

---

## 2. Ontology Engine (`ontology`)

Handles timeless logical validations (Level A) and verifies compatibility under the **Asymmetry Conjecture**.

### `StateSpaceManager`
Defines boundaries and constraint predicates for the valid configuration set $V$:
```python
from structural_time_core.ontology import StateSpaceManager

# Create a 3D state space with specific bounds
bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]
manager = StateSpaceManager(bounds)

# Add a constraint predicate: state[0] + state[1] must be <= 1.2
manager.add_constraint_predicate(lambda s: s[0] + s[1] <= 1.2)

is_valid = manager.is_in_V(np.array([0.5, 0.5, 0.5]))  # Returns True
```

### `LogicalCompatibilityChecker`
Verifies that the operator K retains structural asymmetry:
```python
from structural_time_core.ontology import LogicalCompatibilityChecker

checker = LogicalCompatibilityChecker(complexity_threshold=0.6)

# Perfectly symmetric/flat trajectory K-states (zero rate of change)
symmetric_trajectory = np.array([0.5, 0.5, 0.5, 0.5])

# In high complexity (0.8 > 0.6), flat/symmetric states are logically incompatible
is_ok = checker.is_compatible(symmetric_trajectory, system_complexity=0.8)
# returns: False
```

---

## 3. Dynamics Engine (`dynamics`)

Simulates continuous flows, bifurcation equilibria, and experienced time density (Level B).

### `QuarticPotentialSolver`
Calculates Free Energy $\mathcal{F}(K) = aK^4 + bK^3 + cK^2 + dK$ (enforces $a > 0$):
```python
from structural_time_core.dynamics import QuarticPotentialSolver

potential = QuarticPotentialSolver(a=1.0, b=0.0, c=-2.0, d=0.0)
equilibria = potential.find_equilibria()
# returns: List of stable/unstable critical points (wells and peaks)
```

### `GradientFlowIntegrator`
Simulates gradient flow path using Euler-Maruyama or Runge-Kutta 4th Order (RK4):
```python
from structural_time_core.dynamics import GradientFlowIntegrator

integrator = GradientFlowIntegrator(gamma=0.05, dt=0.01)

# Step using RK4 (highly stable deterministic step)
K_next = integrator.step_rk4(K_current, potential.compute_dF_dK, u=0.0)
```

### `BoundedTemporalDensityCalculator`
Computes theExperienced Temporal Density $T(K)$ preventing pole singularities:
```python
from structural_time_core.dynamics import BoundedTemporalDensityCalculator

calc = BoundedTemporalDensityCalculator(alpha=1.5)
T_ops = calc.compute_T_ops(velocity=0.02, equilibrium_distance=0.05)
```

---

## 4. Regime Clustering & Analytics (`analytics`)

Groups trajectories into one of the 5 theoretical regimes.

### `TheoryGuidedClustering`
Classifies states based on Energy ($E_K$), Velocity ($dK/dt$), and Decay ($\gamma$):
```python
from structural_time_core.analytics import TheoryGuidedClustering

clustering = TheoryGuidedClustering()

# Fits and maps 5 clusters based on Centers (separates Frozen and Decayed)
regimes = clustering.fit_predict_regimes(dataset)
```
