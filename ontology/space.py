import numpy as np
from typing import List, Callable, Set, Tuple

class StateSpaceManager:
    """
    Manages the timeless State Space (S) and the Valid Configurations Set (V) 
    at Level A of the Core Structural Time Ontology.
    """
    def __init__(self, dimension_bounds: List[Tuple[float, float]] = None):
        # S: Bounds of possible values for each dimension in Platonist state space
        # Defaults to [0.0, 1.0] for a 3-dimensional space if none provided
        self.bounds = dimension_bounds or [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]
        self.predicates: List[Callable[[np.ndarray], bool]] = []

    def add_constraint_predicate(self, predicate: Callable[[np.ndarray], bool]):
        """
        Add a constraint predicate P to verify state validity.
        """
        self.predicates.append(predicate)

    def is_in_S(self, state: np.ndarray) -> bool:
        """
        Check if a state lies within the mathematical boundary of State Space S.
        """
        state = np.asarray(state)
        if len(state) != len(self.bounds):
            return False
        for val, (low, high) in zip(state, self.bounds):
            if val < low or val > high:
                return False
        return True

    def is_in_V(self, state: np.ndarray) -> bool:
        """
        Check if a state belongs to the Valid Configurations Set V.
        V = { s in S | P(s) is True for all constraint predicates P }
        """
        if not self.is_in_S(state):
            return False
        # Verify all predicates P
        for p in self.predicates:
            if not p(state):
                return False
        return True
