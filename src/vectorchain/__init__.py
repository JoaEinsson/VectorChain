"""VectorChain public package."""

from importlib.metadata import version

from vectorchain.core import Segment, VectorChain
from vectorchain.features import DEFAULT_FEATURES, SUPPORTED_FEATURES, FeatureName

__version__ = version("vectorchain")

__all__ = [
    "DEFAULT_FEATURES",
    "SUPPORTED_FEATURES",
    "FeatureName",
    "Segment",
    "VectorChain",
    "__version__",
]
