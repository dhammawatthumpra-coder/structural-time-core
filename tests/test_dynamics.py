import unittest
import numpy as np
from structural_time_core.dynamics import (
    QuarticPotentialSolver,
    PartitionOperator,
    GradientFlowIntegrator,
    BoundedTemporalDensityCalculator
)

class TestDynamics(unittest.TestCase):
    def test_quartic_potential_solver(self):
        # F(K) = K^4 - 2*K^2 (double-well potential with minima at -1 and 1, unstable max at 0)
        solver = QuarticPotentialSolver(a=1.0, b=0.0, c=-2.0, d=0.0)
        
        # Test values
        self.assertEqual(solver.compute_F(0.0), 0.0)
        self.assertEqual(solver.compute_F(1.0), -1.0)
        self.assertEqual(solver.compute_F(-1.0), -1.0)
        
        # First derivatives
        self.assertEqual(solver.compute_dF_dK(0.0), 0.0)
        self.assertEqual(solver.compute_dF_dK(1.0), 0.0)
        self.assertEqual(solver.compute_dF_dK(-1.0), 0.0)
        
        # Equilibria points check
        eqs = solver.find_equilibria()
        self.assertEqual(len(eqs), 3)
        
        # Check stable vs unstable classification
        # K = -1.0 should be stable (min)
        self.assertEqual(eqs[0]['K'], -1.0)
        self.assertEqual(eqs[0]['type'], 'stable')
        
        # K = 0.0 should be unstable (max)
        self.assertEqual(eqs[1]['K'], 0.0)
        self.assertEqual(eqs[1]['type'], 'unstable')
        
        # K = 1.0 should be stable (min)
        self.assertEqual(eqs[2]['K'], 1.0)
        self.assertEqual(eqs[2]['type'], 'stable')

    def test_partition_operator(self):
        pi = PartitionOperator(resolution=0.2)
        v = np.array([0.09, 0.28, 0.39, 0.52])
        v_pi = pi.apply(v)
        
        expected = np.array([0.0, 0.2, 0.4, 0.6])
        np.testing.assert_array_almost_equal(v_pi, expected)

    def test_gradient_flow_integrator(self):
        integrator = GradientFlowIntegrator(gamma=0.1, dt=0.01)
        solver = QuarticPotentialSolver(a=1.0, b=0.0, c=-1.0, d=0.0)
        
        # Deterministic check from K=0.5
        K_val = 0.5
        dF = solver.compute_dF_dK(K_val) # 4*(0.5^3) - 2*(0.5) = 0.5 - 1.0 = -0.5
        
        # Euler step
        K_next_euler = integrator.step_euler(K_val, dF, u=0.0, noise_scale=0.0)
        # Expected: K_next = K + (-dF - gamma*K)*dt = 0.5 + (0.5 - 0.1*0.5)*0.01 = 0.5 + 0.45*0.01 = 0.5045
        self.assertAlmostEqual(K_next_euler, 0.5045, places=5)
        
        # RK4 step
        K_next_rk4 = integrator.step_rk4(K_val, solver.compute_dF_dK, u=0.0, noise_scale=0.0)
        self.assertGreater(K_next_rk4, 0.5)  # Flow should move towards stable equilibrium at K=1.0

    def test_bounded_temporal_density(self):
        calc = BoundedTemporalDensityCalculator(alpha=1.0)
        
        # T_ops = exp( - (alpha * velocity * distance)^2 )
        # case 1: zero velocity -> T_ops = 1.0 (minimal experienced time flow / steady state)
        self.assertAlmostEqual(calc.compute_T_ops(0.0, 2.0), 1.0)
        
        # case 2: high velocity and distance -> T_ops -> 0
        self.assertLess(calc.compute_T_ops(2.0, 3.0), 0.1)

if __name__ == '__main__':
    unittest.main()
