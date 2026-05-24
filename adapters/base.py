from abc import ABC, abstractmethod
import numpy as np

class DomainAdapter(ABC):
    """
    Base class for all domain-specific adapters.
    Responsible for mapping physical/domain-specific metrics to a normalized K-state vector.
    """
    @abstractmethod
    def map_to_K(self, raw_data) -> np.ndarray:
        """
        Map domain-specific raw data to a standardized K-state vector (in R^n).
        
        Args:
            raw_data: Raw input data specific to the domain (dict, array, etc.)
            
        Returns:
            np.ndarray: Normalized K-state vector.
        """
        pass
