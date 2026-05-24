import numpy as np
from .base import DomainAdapter

class TransformerAdapter(DomainAdapter):
    """
    Adapter for Transformer models (such as LLMs during training or sequence generation).
    Maps token-level cross-layer similarity, prediction errors, and sequence age to a K-vector.
    """
    def __init__(self, n_layers: int = 12):
        self.n_layers = n_layers

    def map_to_K(self, raw_data: dict) -> np.ndarray:
        """
        Maps transformer raw telemetry to a normalized 3-dimensional K-vector:
        [complexity, stability, error_rate]
        
        Args:
            raw_data: dict containing:
                - 'cross_layer_similarity': list or np.ndarray of shape (seq_len, n_layers) or (n_layers,)
                - 'prediction_error_rates': list or np.ndarray of prediction errors/loss values
                - 'sequence_age': current generation step or token position
                - 'max_sequence_len': maximum sequence limit (optional)
                
        Returns:
            np.ndarray: [complexity, stability, error_rate] normalized to [0.0, 1.0]
        """
        cls_data = np.asarray(raw_data.get('cross_layer_similarity', [0.8]))
        errors = np.asarray(raw_data.get('prediction_error_rates', [0.1]))
        age = float(raw_data.get('sequence_age', 0))
        max_age = float(raw_data.get('max_sequence_len', 1024))

        # 1. Complexity: inverse of mean cross-layer similarity (high similarity -> low complexity)
        mean_cls = np.mean(cls_data)
        complexity = 1.0 - float(mean_cls)
        complexity = max(0.0, min(1.0, complexity))

        # 2. Stability: inverse of standard deviation (high variance -> low stability)
        std_cls = np.std(cls_data) if cls_data.size > 1 else 0.0
        stability = 1.0 - float(std_cls * 2.0)  # scale to make variance visible
        stability = max(0.0, min(1.0, stability))

        # 3. Error Rate: normalized mean error
        mean_err = np.mean(errors)
        error_rate = max(0.0, min(1.0, float(mean_err)))

        return np.array([complexity, stability, error_rate], dtype=float)
