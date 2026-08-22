"""VectorChain public package."""

from importlib.metadata import version

from vectorchain.core import Segment, VectorChain
from vectorchain.features import DEFAULT_FEATURES, SUPPORTED_FEATURES, FeatureName
from vectorchain.metrics import compression_factor, mae, retention_fraction, rmse
from vectorchain.synthetic import (
    RngLike,
    generate_chirp,
    generate_first_order_response,
    generate_piecewise_linear,
    generate_ramp,
    generate_regime_change,
    generate_second_order_response,
    generate_sine,
)

__version__ = version("vectorchain")

__all__ = [
    "DEFAULT_FEATURES",
    "SUPPORTED_FEATURES",
    "FeatureName",
    "RngLike",
    "Segment",
    "VectorChain",
    "__version__",
    "compression_factor",
    "generate_chirp",
    "generate_first_order_response",
    "generate_piecewise_linear",
    "generate_ramp",
    "generate_regime_change",
    "generate_second_order_response",
    "generate_sine",
    "mae",
    "retention_fraction",
    "rmse",
]
