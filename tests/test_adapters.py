import unittest
import numpy as np
from structural_time_core.adapters import TransformerAdapter, NeuralNetworkTelemetryAdapter

class TestAdapters(unittest.TestCase):
    def test_transformer_adapter_mapping(self):
        adapter = TransformerAdapter(n_layers=12)
        raw_data = {
            'cross_layer_similarity': [0.9, 0.9, 0.9, 0.9],
            'prediction_error_rates': [0.1, 0.2, 0.15],
            'sequence_age': 50,
            'max_sequence_len': 100
        }
        
        K = adapter.map_to_K(raw_data)
        
        # Output should be a 3D numpy array
        self.assertIsInstance(K, np.ndarray)
        self.assertEqual(K.shape, (3,))
        
        # High similarity -> complexity should be low (1.0 - 0.9 = 0.1)
        self.assertAlmostEqual(K[0], 0.1, places=3)
        # Identical similarity values -> std = 0 -> stability = 1.0
        self.assertAlmostEqual(K[1], 1.0, places=3)
        # Mean error rate is (0.1+0.2+0.15)/3 = 0.15
        self.assertAlmostEqual(K[2], 0.15, places=3)

    def test_sociology_adapter_mapping_and_noise(self):
        import sys
        import os
        # Ensure examples folder is importable
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
        from examples.sociology_adapter_demo import SociologyAdapter

        with self.assertWarns(UserWarning):
            adapter = SociologyAdapter()
            
        raw_data = {
            'info_frequency': 0.8,
            'social_pressure': 0.6,
            'generation_sizes': [100, 200, 100]
        }
        
        K = adapter.map_to_K(raw_data)
        
        self.assertIsInstance(K, np.ndarray)
        self.assertEqual(K.shape, (3,))
        
        # Dimension mappings
        self.assertEqual(K[0], 0.8) # info_freq
        self.assertEqual(K[1], 0.6) # pressure
        self.assertGreater(K[2], 0.0) # generational gap should be non-zero due to size differences
        
        # Noise estimation
        noise = adapter.estimate_noise(raw_data)
        self.assertAlmostEqual(noise, 0.1 * 0.8 * 0.6, places=4)

    def test_neural_net_adapter_mapping(self):
        adapter = NeuralNetworkTelemetryAdapter(max_weight_norm=100.0, max_grad_norm=10.0)
        raw_data = {
            'train_loss': 0.1,
            'val_loss': 0.5,
            'val_accuracy': 0.85,
            'weight_norm': 50.0,
            'gradient_norm': 2.0
        }
        
        K = adapter.map_to_K(raw_data)
        
        self.assertIsInstance(K, np.ndarray)
        self.assertEqual(K.shape, (3,))
        
        # Complexity: weight_norm (50.0) / max_weight_norm (100.0) = 0.5
        self.assertAlmostEqual(K[0], 0.5, places=3)
        # Stability: val_accuracy (0.85) - gradient_norm (2.0) / max_grad_norm (10.0) = 0.85 - 0.2 = 0.65
        self.assertAlmostEqual(K[1], 0.65, places=3)
        # Error rate: val_loss (0.5) / (val_loss + 1.0) = 0.5 / 1.5 = 0.3333
        self.assertAlmostEqual(K[2], 0.3333, places=3)

if __name__ == '__main__':
    unittest.main()
