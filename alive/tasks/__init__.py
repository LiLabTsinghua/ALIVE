"""Canonical ALIVE benchmark task classes."""

from .in_distribution import InDistribution
from .out_of_distribution import OutOfDistribution
from .active_learning import ActiveLearning

__all__ = [
    "InDistribution",
    "OutOfDistribution",
    "ActiveLearning",
]
