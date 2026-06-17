"""
StructuralTime-Core v0.1 (MVP)
==============================
A modular domain-agnostic library implementing the Core Structural Time Ontology and Dynamics framework.
"""

from structural_time_core.adapters import TransformerAdapter, NeuralNetworkTelemetryAdapter
from structural_time_core.ontology import StateSpaceManager, LogicalCompatibilityChecker
from structural_time_core.dynamics import (
    QuarticPotentialSolver,
    PartitionOperator,
    GradientFlowIntegrator,
    BoundedTemporalDensityCalculator
)
from structural_time_core.analytics import HybridRegimeClustering, VisualizationAPI

__version__ = "0.1.4"

__all__ = [
    'TransformerAdapter',
    'NeuralNetworkTelemetryAdapter',
    'StateSpaceManager',
    'LogicalCompatibilityChecker',
    'QuarticPotentialSolver',
    'PartitionOperator',
    'GradientFlowIntegrator',
    'BoundedTemporalDensityCalculator',
    'HybridRegimeClustering',
    'VisualizationAPI',
    '__version__'
]
