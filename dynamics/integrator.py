import numpy as np
from typing import Callable, Optional

class GradientFlowIntegrator:
    """
    Gradient Flow Integrator (Level B Dynamics).
    Simulates the structural K-state trajectory over time using:
        dK/dt = -dF/dK(K) - gamma*K + u + xi
        
    Provides both Euler-Maruyama (for stochastic simulations) and 
    Runge-Kutta 4th Order (RK4) integration step methods.
    """
    def __init__(self, gamma: float = 0.05, dt: float = 0.01, random_state: Optional[int] = None):
        self.gamma = float(gamma)  # Systemic decay rate
        self.dt = float(dt)
        if not np.isfinite(self.gamma):
            raise ValueError("gamma must be finite")
        if not np.isfinite(self.dt) or self.dt <= 0.0:
            raise ValueError("dt must be a positive finite value")
        self.rng = np.random.default_rng(random_state)

    def step_euler(self, 
                   K: float, 
                   dF_dK: float, 
                   u: float = 0.0, 
                   noise_scale: float = 0.0) -> float:
        """
        Calculates K_next using Euler-Maruyama:
        dK = (-dF_dK - gamma*K + u)*dt + noise_scale*sqrt(dt)*W
        """
        noise = noise_scale * np.sqrt(self.dt) * self.rng.normal(0, 1)
        dK_dt = -dF_dK - (self.gamma * K) + u
        next_K = K + dK_dt * self.dt + noise
        return float(next_K)

    def step_rk4(self,
                 K: float,
                 dF_dK_func: Callable[[float], float],
                 u: float = 0.0,
                 noise_scale: float = 0.0) -> float:
        """
        Calculates K_next using Runge-Kutta 4th Order (RK4) for deterministic flow
        with additive stochastic noise.
        """
        # Deterministic derivative helper
        def f(k):
            return -dF_dK_func(k) - (self.gamma * k) + u

        dt = self.dt
        k1 = f(K)
        k2 = f(K + 0.5 * dt * k1)
        k3 = f(K + 0.5 * dt * k2)
        k4 = f(K + dt * k3)

        deterministic_step = (k1 + 2.0 * k2 + 2.0 * k3 + k4) * dt / 6.0
        stochastic_step = noise_scale * np.sqrt(dt) * self.rng.normal(0, 1)

        return float(K + deterministic_step + stochastic_step)


class BoundedTemporalDensityCalculator:
    """
    Bounded Temporal Density Calculator.
    
    This class provides TWO different formulas for temporal density,
    which serve different purposes:
    
    1. compute_T_ops(): STABILITY INDICATOR
       Formula: T_ops = exp(-(alpha * dK_dt * dist)^2)
       - T_ops = 1.0 at equilibrium (dK_dt -> 0 or dist -> 0)
       - T_ops -> 0 when far from equilibrium with high velocity
       - Interpretation: measures structural STABILITY, not temporal density
       - Used internally for regime monitoring
       
       NOTE: This formula is NOT the T(K) described in the paper (§2.5).
       Despite the class name, this is a stability metric, not temporal density.
       The docstring previously described this as "Bell-curve" which was
       incorrect — it is a monotonically decreasing Gaussian.
    
    2. compute_T_K_paper(): PAPER TEMPORAL DENSITY (§2.5)
       Formula: T(K) = alpha * dK_dt * dist * exp(-beta * dist^2)
       - T(K) = 0 at equilibrium (no temporal experience when static)
       - T(K) peaks at intermediate distance from equilibrium
       - T(K) -> 0 when very far from equilibrium
       - This IS a true bell-curve and matches the paper formula exactly
       - alpha=1.0, beta=0.5 per §3.1 parameters
    
    See Dynamics §5.6.12 and §2.5 for context on these two formulas.
    See Dynamics §5.6.11 for the comparison between Critical and Active T(K).
    """
    def __init__(self, alpha: float = 1.5):
        self.alpha = alpha

    def compute_T_ops(self, dK_dt: float, equilibrium_distance: float) -> float:
        """
        Stability indicator (NOT the paper's T(K) formula).
        
        Formula: T_ops = exp( - (alpha * ||dK/dt|| * dist)^2 )
        
        This is a monotonically decreasing function:
            T_ops = 1.0: At equilibrium (dK_dt ≈ 0 or dist ≈ 0) — MAXIMUM STABILITY
            T_ops → 0:  Far from equilibrium with high velocity — LOW STABILITY
        
        Use for: real-time regime stability monitoring, not for computing
        experienced temporal density as defined in the Dynamics paper §2.5.
        For the paper formula, use compute_T_K_paper() instead.
        """
        variance_term = np.square(self.alpha * dK_dt * equilibrium_distance)
        T_ops = np.exp(-variance_term)
        return float(T_ops)

    def compute_T_K_paper(
        self,
        dK_dt: float,
        equilibrium_distance: float,
        alpha: float = 1.0,
        beta: float = 0.5,
    ) -> float:
        """
        Paper temporal density formula (Dynamics §2.5).
        
        Formula: T(K) = alpha * ||dK/dt|| * dist * exp(-beta * dist^2)
        
        Default parameters match §3.1: alpha=1.0, beta=0.5
        
        This IS a true bell-curve:
            T(K) = 0: At equilibrium (minimal change = no temporal experience)
            T(K) peaks at dist = 1/sqrt(2*beta) ≈ 1.0 (for beta=0.5)
            T(K) → 0: Very far from equilibrium (exponential decay dominates)
        
        Results from Item 10 (§5.7, 2026-06-17):
            Active:   T(K) = 0.0031 ± 0.0001
            Critical: T(K) = 0.2281 ± 0.0248
            Ratio:    74.3× (Critical > Active confirmed)
        
        Use for: computing temporal density as described in the paper.
        """
        T_K = alpha * dK_dt * equilibrium_distance * np.exp(-beta * equilibrium_distance ** 2)
        return float(T_K)
