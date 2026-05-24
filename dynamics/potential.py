import numpy as np
from typing import List, Dict, Any

class QuarticPotentialSolver:
    """
    Quartic Potential Solver (Level B Perception).
    Calculates the Free Energy Potential:
        F(K) = a*K^4 + b*K^3 + c*K^2 + d*K
        
    Enforces a > 0 for stability (bounded potential, avoiding gradient explosion).
    """
    def __init__(self, a: float = 1.0, b: float = 0.0, c: float = -1.0, d: float = 0.0):
        # Enforce positive coefficient on quartic term for bounded potential
        self.a = max(1e-5, float(a))
        self.b = float(b)
        self.c = float(c)
        self.d = float(d)

    def compute_F(self, K: float) -> float:
        """Evaluate free energy F(K)"""
        return self.a * (K**4) + self.b * (K**3) + self.c * (K**2) + self.d * K

    def compute_dF_dK(self, K: float) -> float:
        """Evaluate first derivative dF/dK"""
        return 4.0 * self.a * (K**3) + 3.0 * self.b * (K**2) + 2.0 * self.c * K + self.d

    def compute_d2F_dK2(self, K: float) -> float:
        """Evaluate second derivative d^2F/dK^2"""
        return 12.0 * self.a * (K**2) + 6.0 * self.b * K + 2.0 * self.c

    def find_equilibria(self) -> List[Dict[str, Any]]:
        """
        Find critical points where dF/dK = 0 using polynomial roots.
        Classifies them as Stable (d^2F/dK^2 > 0) or Unstable (d^2F/dK^2 < 0).
        """
        coeffs = [4.0 * self.a, 3.0 * self.b, 2.0 * self.c, self.d]
        roots = np.roots(coeffs)
        
        equilibria = []
        for r in roots:
            # We are interested in real equilibria
            if np.isreal(r):
                val = float(np.real(r))
                d2 = self.compute_d2F_dK2(val)
                eq_type = "stable" if d2 > 0 else ("unstable" if d2 < 0 else "neutral")
                equilibria.append({
                    "K": val,
                    "type": eq_type,
                    "second_derivative": d2,
                    "F": self.compute_F(val)
                })
        
        # Sort by K value
        equilibria.sort(key=lambda x: x["K"])
        return equilibria


class PartitionOperator:
    """
    Partition Operator (Pi).
    Divides the continuous state space into equivalence classes/bins.
    Represents the observer's grain of perception.
    """
    def __init__(self, resolution: float = 0.1):
        self.delta = max(1e-5, float(resolution))  # Bin size / grain

    def apply(self, K_vector: np.ndarray) -> np.ndarray:
        """
        Groups/bins states: Pi(K) = round(K / delta) * delta
        """
        K_vector = np.asarray(K_vector, dtype=float)
        return np.round(K_vector / self.delta) * self.delta
