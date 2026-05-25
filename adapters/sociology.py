import numpy as np
from .base import DomainAdapter

class SociologyAdapter(DomainAdapter):
    """
    Adapter for sociological models (such as demographic dynamics and multi-generational interaction).
    Maps information flow frequency and social pressure indexes to a K-vector and environmental noise.
    
    WARNING: This adapter is purely illustrative and serves as a conceptual scaffolding. 
    The mapping between sociological variables and K-state coordinates is highly metaphorical 
    and should not be used for quantitative sociological predictions without a rigorous, 
    domain-validated translation matrix.
    """
    def map_to_K(self, raw_data: dict) -> np.ndarray:
        """
        Maps sociological data to a normalized 3-dimensional K-vector:
        [interaction_density, constraint_load, generational_gap]
        
        Args:
            raw_data: dict containing:
                - 'info_frequency': float, frequency/volume of information streams (normalized [0, 1])
                - 'social_pressure': float, index of social pressure/constraints (normalized [0, 1])
                - 'generation_sizes': list or np.ndarray of population sizes across generations
                
        Returns:
            np.ndarray: [interaction_density, constraint_load, generational_gap] normalized to [0.0, 1.0]
        """
        info_freq = float(raw_data.get('info_frequency', 0.5))
        pressure = float(raw_data.get('social_pressure', 0.5))
        sizes = np.asarray(raw_data.get('generation_sizes', [100, 100]))

        # 1. Interaction Density is mapped directly from information frequency
        interaction_density = max(0.0, min(1.0, info_freq))

        # 2. Constraint Load is mapped directly from social pressure
        constraint_load = max(0.0, min(1.0, pressure))

        # 3. Generational Gap: standard deviation of normalized generational sizes (0 means equal sizes, higher means gap/imbalance)
        if sizes.size > 1:
            norm_sizes = sizes / np.sum(sizes)
            generational_gap = float(np.std(norm_sizes) * np.sqrt(sizes.size))  # scaled standard deviation
        else:
            generational_gap = 0.0
        generational_gap = max(0.0, min(1.0, generational_gap))

        return np.array([interaction_density, constraint_load, generational_gap], dtype=float)

    def estimate_noise(self, raw_data: dict) -> float:
        """
        Estimate external environmental noise (xi) from the complexity of social dynamics.
        
        Args:
            raw_data: dict containing 'social_pressure' and 'info_frequency'
            
        Returns:
            float: Noise scale (xi)
        """
        pressure = float(raw_data.get('social_pressure', 0.5))
        info_freq = float(raw_data.get('info_frequency', 0.5))
        # High pressure + high frequency creates higher systemic friction/noise
        return float(0.1 * (pressure * info_freq))
