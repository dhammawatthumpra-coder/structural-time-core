"""
SociologyAdapter Conceptual Demonstration
=========================================
This script demonstrates the SociologyAdapter, which serves as a conceptual,
metaphorical mapping layer to show how multi-generational and social telemetry
might be projected onto a K-state space.

WARNING: This is an educational tool. The mapping is highly metaphorical
and has not been validated for quantitative sociological prediction.
"""

import warnings
import numpy as np
from structural_time_core.adapters import DomainAdapter

class SociologyAdapter(DomainAdapter):
    """
    Adapter for sociological models (such as demographic dynamics and multi-generational interaction).
    Maps information flow frequency and social pressure indexes to a K-vector and environmental noise.
    """
    def __init__(self):
        warnings.warn(
            "SociologyAdapter is a conceptual metaphor and is not validated for quantitative sociological prediction.",
            UserWarning,
            stacklevel=2
        )

    def map_to_K(self, raw_data: dict) -> np.ndarray:
        """
        Maps sociological data to a normalized 3-dimensional K-vector:
        [interaction_density, constraint_load, generational_gap]
        """
        info_freq = float(raw_data.get('info_frequency', 0.5))
        pressure = float(raw_data.get('social_pressure', 0.5))
        sizes = np.asarray(raw_data.get('generation_sizes', [100, 100]))

        # 1. Interaction Density
        interaction_density = max(0.0, min(1.0, info_freq))

        # 2. Constraint Load
        constraint_load = max(0.0, min(1.0, pressure))

        # 3. Generational Gap
        if sizes.size > 1:
            norm_sizes = sizes / np.sum(sizes)
            generational_gap = float(np.std(norm_sizes) * np.sqrt(sizes.size))
        else:
            generational_gap = 0.0
        generational_gap = max(0.0, min(1.0, generational_gap))

        return np.array([interaction_density, constraint_load, generational_gap], dtype=float)

    def estimate_noise(self, raw_data: dict) -> float:
        """
        Estimate external environmental noise (xi) from the complexity of social dynamics.
        """
        pressure = float(raw_data.get('social_pressure', 0.5))
        info_freq = float(raw_data.get('info_frequency', 0.5))
        return float(0.1 * (pressure * info_freq))

if __name__ == "__main__":
    print("=== SociologyAdapter Demonstration ===")
    
    # Instantiate the adapter (triggers the warning)
    adapter = SociologyAdapter()
    
    # Raw sociological survey metrics
    raw_data = {
        'info_frequency': 0.8,
        'social_pressure': 0.6,
        'generation_sizes': [100, 200, 100]
    }
    
    K = adapter.map_to_K(raw_data)
    noise = adapter.estimate_noise(raw_data)
    
    print("\nInputs:")
    print(f"  - Information Frequency: {raw_data['info_frequency']}")
    print(f"  - Social Pressure Index: {raw_data['social_pressure']}")
    print(f"  - Generational Sizes:    {raw_data['generation_sizes']}")
    
    print("\nMapped K-State Vector:")
    print(f"  - Interaction Density:  {K[0]:.4f}")
    print(f"  - Constraint Load:      {K[1]:.4f}")
    print(f"  - Generational Gap:     {K[2]:.4f}")
    print(f"  - Estimated Noise (xi): {noise:.4f}")
    
    print("\n=== Demo completed successfully ===")
