import unittest
import numpy as np
from structural_time_core.analytics import TheoryGuidedClustering, VisualizationAPI

class TestAnalytics(unittest.TestCase):
    def test_theory_guided_clustering_fallback(self):
        clustering = TheoryGuidedClustering()
        
        # Test fallback rules (when fit has not run or has too few samples)
        self.assertEqual(clustering.predict_sample(E_K=0.8, dK_dt=0.8, gamma=0.1), "Turbulent")
        self.assertEqual(clustering.predict_sample(E_K=0.2, dK_dt=0.4, gamma=0.1), "Active")
        # Low velocity, high decay -> Decayed
        self.assertEqual(clustering.predict_sample(E_K=0.2, dK_dt=0.05, gamma=0.6), "Decayed")
        # Low velocity, low decay -> Frozen
        self.assertEqual(clustering.predict_sample(E_K=0.2, dK_dt=0.05, gamma=0.1), "Frozen")
        # Intermediate -> Critical
        self.assertEqual(clustering.predict_sample(E_K=0.5, dK_dt=0.2, gamma=0.1), "Critical")

    def test_theory_guided_clustering_fit(self):
        # Create a synthetic dataset containing 5 distinct regime behaviors
        # columns: [E_K, dK_dt_norm, gamma]
        samples = []
        for _ in range(10): # Turbulent
            samples.append([0.8, 0.8, 0.2])
        for _ in range(10): # Active
            samples.append([0.3, 0.5, 0.1])
        for _ in range(10): # Decayed
            samples.append([0.2, 0.05, 0.7])
        for _ in range(10): # Frozen
            samples.append([0.1, 0.02, 0.1])
        for _ in range(10): # Critical
            samples.append([0.5, 0.22, 0.2])
            
        data = np.array(samples)
        clustering = TheoryGuidedClustering()
        
        regimes = clustering.fit_predict_regimes(data)
        self.assertTrue(clustering.is_fitted)
        self.assertEqual(len(regimes), 50)
        
        # Let's check that we mapped cluster IDs successfully
        unique_mapped_regimes = set(clustering.regime_map.values())
        # The fitted centers should have successfully resolved our regimes
        self.assertTrue("Active" in unique_mapped_regimes or len(unique_mapped_regimes) >= 3)

    def test_visualization_api_formatting(self):
        history = [
            {'E_K': 0.2, 'dK_dt': 0.5, 'regime': 'Active'},
            {'E_K': 0.3, 'dK_dt': 0.1, 'regime': 'Critical'},
            {'E_K': 0.8, 'dK_dt': 0.9, 'regime': 'Turbulent'}
        ]
        
        formatted = VisualizationAPI.format_phase_space(history)
        
        self.assertEqual(formatted['E_K'], [0.2, 0.3, 0.8])
        self.assertEqual(formatted['dK_dt'], [0.5, 0.1, 0.9])
        self.assertEqual(formatted['regime'], ['Active', 'Critical', 'Turbulent'])

if __name__ == '__main__':
    unittest.main()
