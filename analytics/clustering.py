import numpy as np
from sklearn.cluster import KMeans
from typing import List, Dict, Any, Union

class TheoryGuidedClustering:
    """
    Theory-Guided Clustering Module (Level B Analytics).
    Groups K-state trajectory samples into one of the 5 Temporal Regimes:
      - Active (Normal execution)
      - Critical (Near bifurcation point)
      - Turbulent (High dissonance, loop/instability)
      - Decayed (Structural dissolution)
      - Frozen (Meditative stillness, stagnation)
      
    Features used:
      1. E_K: Complexity/Energy measure (e.g. C_K or K-norm)
      2. dK_dt: Velocity of state changes (norm of derivative)
      3. gamma: Structural decay rate (deviation rate from equilibrium)
      
    Inclusion of gamma resolves the classification overlap between Frozen and Decayed.
    """
    def __init__(self, random_state: int = 42):
        self.model = KMeans(n_clusters=5, random_state=random_state, n_init=10)
        self.regime_map: Dict[int, str] = {}
        self.is_fitted = False

    def fit(self, trajectory_data: np.ndarray):
        """
        Train the KMeans clustering model and map clusters to theoretical regimes.
        
        Args:
            trajectory_data: np.ndarray of shape (n_samples, 3) where columns are:
                             [E_K, dK_dt_norm, gamma]
        """
        X = np.asarray(trajectory_data, dtype=float)
        n_samples = X.shape[0]

        if n_samples < 5:
            # Insufficient data to fit 5 clusters, use fallback heuristic
            self.is_fitted = False
            return self

        self.model.fit(X)
        centers = self.model.cluster_centers_
        self.regime_map = {}

        # Classify each cluster center according to structural time theory
        for idx, center in enumerate(centers):
            E_K, dK_dt, gamma = center
            
            if dK_dt > 0.6 and E_K > 0.6:
                self.regime_map[idx] = "Turbulent"
            elif dK_dt > 0.3:
                self.regime_map[idx] = "Active"
            elif dK_dt <= 0.15 and gamma > 0.4:
                self.regime_map[idx] = "Decayed"
            elif dK_dt <= 0.15 and gamma <= 0.4:
                self.regime_map[idx] = "Frozen"
            else:
                self.regime_map[idx] = "Critical"
                
        self.is_fitted = True
        return self

    def predict_sample(self, E_K: float, dK_dt: float, gamma: float) -> str:
        """
        Predict the regime for a single trajectory snapshot.
        If the model is not fitted, falls back to a deterministic rule-based classifier.
        """
        sample = np.array([[E_K, dK_dt, gamma]], dtype=float)
        
        if self.is_fitted:
            cluster_id = int(self.model.predict(sample)[0])
            return self.regime_map.get(cluster_id, "Active")
        
        # Rule-based fallback (Level B Heuristic classifier)
        if dK_dt > 0.6 and E_K > 0.6:
            return "Turbulent"
        elif dK_dt > 0.3:
            return "Active"
        elif dK_dt <= 0.15:
            return "Decayed" if gamma > 0.4 else "Frozen"
        return "Critical"

    def fit_predict_regimes(self, trajectory_data: np.ndarray) -> List[str]:
        """
        Fits the model and returns mapped regime strings for each sample.
        """
        X = np.asarray(trajectory_data, dtype=float)
        self.fit(X)
        
        if self.is_fitted:
            labels = self.model.labels_
            return [self.regime_map[L] for L in labels]
        else:
            # Fallback for small datasets
            return [self.predict_sample(row[0], row[1], row[2]) for row in X]
