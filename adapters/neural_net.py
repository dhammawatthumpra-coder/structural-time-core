import numpy as np
from .base import DomainAdapter

class NeuralNetworkTelemetryAdapter(DomainAdapter):
    """
    Adapter for Neural Network training telemetry (such as during Grokking or Mode Collapse).
    Maps train/validation loss, validation accuracy, weight norms, and gradient norms to a K-vector.
    """
    def __init__(self, max_weight_norm: float = 100.0, max_grad_norm: float = 10.0):
        self.max_weight_norm = max_weight_norm
        self.max_grad_norm = max_grad_norm

    def map_to_K(self, raw_data: dict) -> np.ndarray:
        """
        Maps neural network training telemetry to a normalized 3-dimensional K-vector:
        [complexity, stability, error_rate]

        Args:
            raw_data: dict containing:
                - 'train_loss': float
                - 'val_loss': float
                - 'val_accuracy': float (between 0 and 1)
                - 'weight_norm': float
                - 'gradient_norm': float
                
        Returns:
            np.ndarray: [complexity, stability, error_rate] normalized to [0.0, 1.0]
        """
        train_loss = float(raw_data.get('train_loss', 1.0))
        val_loss = float(raw_data.get('val_loss', 1.0))
        val_accuracy = float(raw_data.get('val_accuracy', 0.0))
        weight_norm = float(raw_data.get('weight_norm', 10.0))
        gradient_norm = float(raw_data.get('gradient_norm', 1.0))

        # 1. Complexity: derived from weight norm normalized to self.max_weight_norm
        complexity = min(1.0, max(0.0, weight_norm / self.max_weight_norm))

        # 2. Stability: validation accuracy is a direct measure of generalization stability,
        # penalized if the gradient norm is exploding (indicating numeric instability).
        grad_penalty = min(0.5, gradient_norm / self.max_grad_norm)
        stability = max(0.0, val_accuracy - grad_penalty)

        # 3. Error Rate: standard sigmoid-like scaling of validation loss
        error_rate = val_loss / (val_loss + 1.0)
        error_rate = max(0.0, min(1.0, error_rate))

        return np.array([complexity, stability, error_rate], dtype=float)
