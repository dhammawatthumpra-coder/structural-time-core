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
    def __init__(self, gamma: float = 0.05, dt: float = 0.01):
        self.gamma = gamma  # Systemic decay rate
        self.dt = dt

    def step_euler(self, 
                   K: float, 
                   dF_dK: float, 
                   u: float = 0.0, 
                   noise_scale: float = 0.0) -> float:
        """
        Calculates K_next using Euler-Maruyama:
        dK = (-dF_dK - gamma*K + u)*dt + noise_scale*sqrt(dt)*W
        """
        noise = noise_scale * np.sqrt(self.dt) * np.random.normal(0, 1)
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
        stochastic_step = noise_scale * np.sqrt(dt) * np.random.normal(0, 1)

        return float(K + deterministic_step + stochastic_step)


class BoundedTemporalDensityCalculator:
    """
    Bounded Temporal Density Calculator.
    Converts structural change velocity and equilibrium distance to experienced temporal density T(K).
    Avoids pole singularities using exponential damping (bell curve).
    """
    def __init__(self, alpha: float = 1.5):
        self.alpha = alpha

    def compute_T_ops(self, dK_dt: float, equilibrium_distance: float) -> float:
        """
        Bell-curve experienced temporal density:
            T_ops = exp( - (alpha * ||dK/dt|| * dist)^2 )
            
        Interpretation:
            T_ops -> 1: Near equilibrium, minimal change (Stable / active)
            T_ops -> 0: Far from equilibrium, high acceleration/deceleration
        """
        variance_term = np.square(self.alpha * dK_dt * equilibrium_distance)
        T_ops = np.exp(-variance_term)
        return float(T_ops)
