"""
StructuralTime-Core v0.1 (MVP)
==============================
A modular domain-agnostic library implementing the Core Structural Time Ontology and Dynamics framework.
"""

from .adapters import TransformerAdapter, NeuralNetworkTelemetryAdapter
from .ontology import StateSpaceManager, LogicalCompatibilityChecker
from .dynamics import (
    QuarticPotentialSolver,
    PartitionOperator,
    GradientFlowIntegrator,
    BoundedTemporalDensityCalculator
)
from .analytics import HybridRegimeClustering, VisualizationAPI

__version__ = "0.1.1"

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
