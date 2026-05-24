import json
import numpy as np
from typing import List, Dict, Any

class VisualizationAPI:
    """
    Visualization API (Level B Analytics).
    Formats and prepares simulation/trajectory data for plotting and dashboard integrations.
    Supports exporting Phase Space coordinates and Bifurcation data.
    """
    @staticmethod
    def format_phase_space(trajectory_history: List[Dict[str, Any]]) -> Dict[str, List[float]]:
        """
        Formats trajectory history into coordinate lists for plotting E_K vs dK_dt.
        
        Args:
            trajectory_history: List of dicts, each containing 'E_K' and 'dK_dt'
            
        Returns:
            Dict: {'E_K': [...], 'dK_dt': [...], 'regime': [...]}
        """
        e_k_coords = []
        dk_dt_coords = []
        regime_labels = []
        
        for entry in trajectory_history:
            e_k_coords.append(float(entry.get('E_K', 0.0)))
            dk_dt_coords.append(float(entry.get('dK_dt', 0.0)))
            regime_labels.append(str(entry.get('regime', 'Active')))
            
        return {
            'E_K': e_k_coords,
            'dK_dt': dk_dt_coords,
            'regime': regime_labels
        }

    @staticmethod
    def generate_bifurcation_diagram(dF_dK_func_builder: Any, 
                                     parameter_range: List[float], 
                                     k_search_range: List[float]) -> Dict[str, Any]:
        """
        Generates equilibrium bifurcation diagram data across a parameter range.
        
        Args:
            dF_dK_func_builder: A function that takes a parameter value (e.g. c coefficient) 
                                 and returns a QuarticPotentialSolver.
            parameter_range: List of float parameter values to sweep.
            k_search_range: Search bounds for K values (e.g. [-2.0, 2.0])
            
        Returns:
            Dict: {'parameter': [...], 'equilibria_K': [...], 'equilibria_type': [...]}
        """
        param_coords = []
        k_coords = []
        type_coords = []
        
        for p in parameter_range:
            solver = dF_dK_func_builder(p)
            equilibria = solver.find_equilibria()
            
            for eq in equilibria:
                k_val = eq['K']
                if k_search_range[0] <= k_val <= k_search_range[1]:
                    param_coords.append(float(p))
                    k_coords.append(float(k_val))
                    type_coords.append(eq['type'])
                    
        return {
            'parameter': param_coords,
            'equilibria_K': k_coords,
            'equilibria_type': type_coords
        }

    @staticmethod
    def export_to_json(data: Dict[str, Any], filepath: str) -> bool:
        """
        Exports visualization data to a JSON file.
        """
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            return True
        except Exception:
            return False
