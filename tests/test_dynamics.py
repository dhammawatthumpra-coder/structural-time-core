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

    def test_gradient_flow_integrator_reproducible_noise(self):
        solver = QuarticPotentialSolver(a=1.0, b=0.0, c=-1.0, d=0.0)
        first = GradientFlowIntegrator(gamma=0.1, dt=0.01, random_state=123)
        second = GradientFlowIntegrator(gamma=0.1, dt=0.01, random_state=123)

        first_value = first.step_rk4(0.5, solver.compute_dF_dK, noise_scale=0.2)
        second_value = second.step_rk4(0.5, solver.compute_dF_dK, noise_scale=0.2)

        self.assertAlmostEqual(first_value, second_value)

    def test_gradient_flow_integrator_rejects_invalid_dt(self):
        with self.assertRaises(ValueError):
            GradientFlowIntegrator(dt=0.0)

    def test_bounded_temporal_density(self):
        calc = BoundedTemporalDensityCalculator(alpha=1.0)
        
        # --- compute_T_ops: STABILITY INDICATOR ---
        # case 1: zero velocity -> T_ops = 1.0 (maximum stability, minimal structural change)
        # NOTE: T_ops=1.0 means HIGH STABILITY, not "high temporal density"
        # This is the OPPOSITE of the paper's T(K) which equals 0 at equilibrium.
        self.assertAlmostEqual(calc.compute_T_ops(0.0, 2.0), 1.0)
        
        # case 2: high velocity and distance -> T_ops near 0 (low stability)
        self.assertLess(calc.compute_T_ops(2.0, 3.0), 0.1)
        
        # case 3: zero distance -> T_ops = 1.0 regardless of velocity
        self.assertAlmostEqual(calc.compute_T_ops(5.0, 0.0), 1.0)
        
        # --- compute_T_K_paper: PAPER FORMULA (§2.5) ---
        # case 4: zero velocity -> T_K = 0 (no movement = no temporal experience)
        # This is the OPPOSITE of compute_T_ops
        self.assertAlmostEqual(calc.compute_T_K_paper(0.0, 2.0), 0.0)
        
        # case 5: zero distance -> T_K = 0 (at equilibrium = no temporal experience)
        self.assertAlmostEqual(calc.compute_T_K_paper(1.0, 0.0), 0.0)
        
        # case 6: peak near dist = 1/sqrt(2*beta) ≈ 1.0 with beta=0.5
        # T_K at dist=1.0 should be higher than at dist=0.5 and dist=3.0
        t_at_peak = calc.compute_T_K_paper(1.0, 1.0)
        t_at_near = calc.compute_T_K_paper(1.0, 0.5)
        t_at_far  = calc.compute_T_K_paper(1.0, 3.0)
        self.assertGreater(t_at_peak, t_at_near)
        self.assertGreater(t_at_peak, t_at_far)
        
        # case 7: verify exact value at peak (alpha=1.0, beta=0.5, dK_dt=1.0, dist=1.0)
        # T_K = 1.0 * 1.0 * 1.0 * exp(-0.5 * 1.0) = exp(-0.5) ≈ 0.6065
        self.assertAlmostEqual(calc.compute_T_K_paper(1.0, 1.0), np.exp(-0.5), places=5)

if __name__ == '__main__':
    unittest.main()
