# Visualization Examples

The library comes with a built-in simulation demo that generates plots of potential wells, trajectory integration, and regime clustering.

To run the demo:
```bash
python examples/simulation_demo.py
```

This will create several plots in the `examples/` directory:

---

## 1. Quartic Potential Well (`potential_well.png`)

This diagram plots the Free Energy curve:
$$\mathcal{F}(K) = a K^4 + b K^3 + c K^2 + d K$$

It automatically finds and tags critical points where $d\mathcal{F}/dK = 0$, classifying them into:
*   **Stable Equilibria (Green dots):** Valleys where the system settles (minima).
*   **Unstable Equilibria (Red dots):** Peaks representing bifurcation boundaries (maxima).

![Potential Well Curve](../examples/potential_well.png)

---

## 2. Gradient Flow & experienced Time (`trajectory.png`)

This plot displays the structural state $K$ trajectory simulated using **Runge-Kutta 4th Order (RK4)** alongside the experienced temporal density $T(K)$ calculated as:
$$T_{\text{ops}} = \exp( -(\alpha \cdot \|\dot{K}\| \cdot \text{dist})^2 )$$

*   **K-State (Blue line):** Flows starting near the unstable peak (K=0.1) and smoothly settling into the stable well equilibrium (K=1.0).
*   **T(K) Experienced Time (Red line):** Approaching 1.0 at stable equilibrium (time perceived as stationary/active flow) and dropping towards 0 during rapid acceleration/displacement.

![Trajectory Plot](../examples/trajectory.png)

---

## 3. Theory-Guided Regime Clustering (`regime_clustering.png`)

A 3D Phase Space plot representing:
*   **X-axis:** Systemic Energy ($E_K$)
*   **Y-axis:** Velocity ($dK/dt$)
*   **Z-axis:** Decay Rate ($\gamma$)

By incorporating the structural decay coefficient ($\gamma$) as the third feature, the **TheoryGuidedClustering** (KMeans) model cleanly separates the **Frozen** and **Decayed** regimes which otherwise overlap on 2D velocity-energy projections.

![Clustering Plot](../examples/regime_clustering.png)
