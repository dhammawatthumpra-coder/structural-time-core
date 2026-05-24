import unittest
import numpy as np
from structural_time_core.ontology import StateSpaceManager, LogicalCompatibilityChecker

class TestOntology(unittest.TestCase):
    def test_state_space_manager_bounds(self):
        # 3D state space bounds
        bounds = [(0.0, 1.0), (0.0, 2.0), (-1.0, 1.0)]
        manager = StateSpaceManager(bounds)
        
        # Test basic membership in S
        self.assertTrue(manager.is_in_S(np.array([0.5, 1.0, 0.0])))
        self.assertFalse(manager.is_in_S(np.array([1.5, 1.0, 0.0]))) # out of bounds dim 0
        self.assertFalse(manager.is_in_S(np.array([0.5, 1.0, -2.0]))) # out of bounds dim 2

    def test_state_space_manager_constraints(self):
        manager = StateSpaceManager()
        
        # Add constraint: dim0 + dim1 <= 1.0
        manager.add_constraint_predicate(lambda state: state[0] + state[1] <= 1.0)
        
        self.assertTrue(manager.is_in_V(np.array([0.3, 0.4, 0.5])))
        self.assertFalse(manager.is_in_V(np.array([0.6, 0.5, 0.5]))) # violates predicate

    def test_asymmetry_conjecture_checker(self):
        checker = LogicalCompatibilityChecker(complexity_threshold=0.6)
        
        # High complexity (e.g. 0.8 > 0.6)
        complexity = 0.8
        
        # 1. Test Symmetric Matrix Operator
        symmetric_matrix = np.array([
            [1.0, 0.2, 0.3],
            [0.2, 1.0, 0.4],
            [0.3, 0.4, 1.0]
        ])
        # A symmetric matrix should have asymmetry = 0 and be rejected (incompatible)
        self.assertFalse(checker.is_compatible(symmetric_matrix, complexity, is_trajectory=False))
        
        # 2. Test Asymmetric Matrix Operator
        asymmetric_matrix = np.array([
            [1.0, 0.9, 0.3],
            [0.2, 1.0, 0.4],
            [0.3, 0.8, 1.0]
        ])
        self.assertTrue(checker.is_compatible(asymmetric_matrix, complexity, is_trajectory=False))
        
        # 3. Test Symmetric (Flat/Zero derivative) Trajectory Sequence
        flat_trajectory = np.array([0.5, 0.5, 0.5, 0.5])
        # Zero change -> asymmetry = 0 -> rejected under high complexity
        self.assertFalse(checker.is_compatible(flat_trajectory, complexity))
        
        # 4. Test Asymmetric Trajectory Sequence
        changing_trajectory = np.array([0.1, 0.3, 0.6, 0.9])
        self.assertTrue(checker.is_compatible(changing_trajectory, complexity))

if __name__ == '__main__':
    unittest.main()
