"""
SatiAdapter — Bridge between Sati Ecosystem and StructuralTime-Core.

Maps Sati's 9-dim K-vector and THM metrics to STC's 3-dim K-state:
    Sati C_K      →  STC E_K  (energy/complexity)
    Sati rate_op  →  STC dK/dt (velocity)
    Sati γ        →  STC γ   (structural decay rate, shared concept)

Usage:
    from examples.sati_adapter import SatiAdapter
    adapter = SatiAdapter()
    sample = adapter.map_from_sqlite_row(k_vector, C_K, gamma, rate_op)
    # Returns: {"E_K": float, "dK_dt": float, "gamma": float}
"""

from typing import Optional, Dict, Any, List
import numpy as np


class SatiAdapter:
    """Maps Sati metrics → STC K-state for regime analysis."""

    def map_to_E_K(self, C_K: Optional[float],
                   dim1: float, dim3: float, dim4: float) -> float:
        """
        Derive STC E_K from Sati C_K and K-vector activity.

        Strategy:
          - If C_K is available (≥7 samples), use C_K directly
            (C_K = LID of 9-dim K-vector → complexity measure)
          - If C_K is None (cold start), use dim1+depth+ctx as proxy
          - Normalize to [0, 1] range using known percentiles
            (P99 ≈ 10.88, P90 ≈ 2.68, saturating at 1.0)
        """
        if C_K is not None:
            # Map C_K [0, ~12] → E_K [0, 1] with soft saturation at P90
            raw = C_K / 3.0  # P90 ≈ 2.68 → ~0.89
            return min(1.0, raw)
        else:
            # Cold start proxy: task activity
            return min(1.0, (dim1 + dim3 + dim4) / 3.0)

    def map_to_dK_dt(self, rate_op: Optional[float],
                     T_ops: Optional[float]) -> float:
        """
        Derive STC dK/dt from Sati rate_op.

        rate_op = Euclidean distance between consecutive K-vectors.
        This is exactly STC's dK/dt (velocity in state space).

        During cold start (< 20 samples), rate_op is 0.0.
        Fall back to T_ops magnitude as activity proxy.
        """
        if rate_op is not None and rate_op > 0:
            # Normalize: observed range [0, ~5.0], soft at 1.0
            return min(1.0, rate_op / 0.6)
        if T_ops is not None and abs(T_ops) > 0:
            return min(1.0, abs(T_ops))
        return 0.0

    def map_from_sqlite_row(self, k_vector: List[float],
                            C_K: Optional[float],
                            gamma: Optional[float],
                            rate_op: Optional[float],
                            T_ops: Optional[float],
                            regime: str) -> Dict[str, Any]:
        """
        Full mapping from Sati SQLite row to STC analysis input.

        Args:
            k_vector: 9-dim K-vector [dim1..dim9]
            C_K: Sati complexity metric
            gamma: Sati decay rate
            rate_op: Sati velocity metric
            T_ops: Sati acceleration metric
            regime: Sati's regime label (for comparison)

        Returns:
            dict with E_K, dK_dt, gamma_stc, sati_regime, raw data
        """
        dim1, _, dim3, dim4 = k_vector[0], k_vector[1], k_vector[2], k_vector[3]

        stc_E_K = self.map_to_E_K(C_K, dim1, dim3, dim4)
        stc_dK_dt = self.map_to_dK_dt(rate_op, T_ops)
        stc_gamma = gamma if gamma is not None else 0.0

        return {
            "E_K": stc_E_K,
            "dK_dt": stc_dK_dt,
            "gamma": stc_gamma,
            "sati_regime": regime,
            "raw": {
                "C_K": C_K, "rate_op": rate_op,
                "T_ops": T_ops, "gamma_sati": gamma,
                "dim1": dim1, "dim3": dim3, "dim4": dim4,
                "k_vector": k_vector,
            }
        }
