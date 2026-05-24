import numpy as np

class LogicalCompatibilityChecker:
    """
    Logical Compatibility Checker (Level A Core).
    Verifies that a cognitive operator (K) is logically compatible with the state space (S, V).
    Specifically implements the Asymmetry Conjecture.
    """
    def __init__(self, complexity_threshold: float = 0.7):
        self.threshold = complexity_threshold

    def calculate_asymmetry(self, K_operator: np.ndarray) -> float:
        """
        Computes the structural asymmetry of the K-operator.
        
        For a Matrix Operator:
            Asymmetry = ||K - K^T|| / (||K|| + epsilon)
            
        For a Trajectory Vector (representing temporal sequence states):
            Asymmetry = Mean absolute first-difference along the trajectory.
            
        Args:
            K_operator: numpy array representing the K-operator matrix or trajectory vector.
            
        Returns:
            float: Asymmetry coefficient.
        """
        K_operator = np.asarray(K_operator, dtype=float)
        if K_operator.size == 0:
            return 0.0

        if K_operator.ndim < 2:
            # Trajectory sequence vector: measure mean rate of change over token steps
            if len(K_operator) < 2:
                return 0.0
            return float(np.abs(np.diff(K_operator)).mean())
        
        # Matrix operator: measure matrix asymmetry
        K_trans = K_operator.T
        numerator = np.linalg.norm(K_operator - K_trans)
        denominator = np.linalg.norm(K_operator) + 1e-9
        return float(numerator / denominator)

    def is_compatible(self, K_operator: np.ndarray, system_complexity: float, is_trajectory: bool = True) -> bool:
        """
        Verifies the Asymmetry Conjecture:
        In complex systems (complexity > threshold), perfectly symmetric operators 
        (e.g., zero temporal trajectory derivative or symmetric matrices) are logically
        incompatible and excluded.
        
        WARNING: Do NOT apply this check to static batch-level classification data 
        (like SST-2 single forward pass). It must run on sequence generation trajectories
        where sequential progression provides the directional arrow of time.
        
        Args:
            K_operator: np.ndarray representing K-state trajectory or matrix.
            system_complexity: float representing current system configuration complexity.
            is_trajectory: bool, asserts that the input is sequential trajectory data.
            
        Returns:
            bool: True if compatible, False if incompatible (logically rejected).
        """
        if not is_trajectory and K_operator.ndim < 2:
            # Warn if static evaluation is attempted on vector without explicit parameter override
            import warnings
            warnings.warn("LogicalCompatibilityChecker should be used with trajectory sequences, not static batches.")

        asymmetry = self.calculate_asymmetry(K_operator)
        
        # If the environment is complex and the operator lacks structural asymmetry, reject it
        if system_complexity > self.threshold and asymmetry < 1e-4:
            return False
        return True
