"""VectorChain public package."""

from importlib.metadata import version

from vectorchain.core import Segment, VectorChain
from vectorchain.features import DEFAULT_FEATURES, SUPPORTED_FEATURES, FeatureName
from vectorchain.metrics import compression_factor, mae, retention_fraction, rmse

__version__ = version("vectorchain")

__all__ = [
    "DEFAULT_FEATURES",
    "SUPPORTED_FEATURES",
    "FeatureName",
    "Segment",
    "VectorChain",
    "__version__",
    "compression_factor",
    "mae",
    "retention_fraction",
    "rmse",
]
