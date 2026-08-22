"""VectorChain public package."""

from importlib.metadata import version

from vectorchain.baselines import (
    FixedSegmentation,
    first_difference,
    first_second_difference,
    fixed_linear_segmentation,
    normalized_raw_values,
    raw_values,
)
from vectorchain.core import Segment, VectorChain
from vectorchain.features import DEFAULT_FEATURES, SUPPORTED_FEATURES, FeatureName
from vectorchain.metrics import compression_factor, mae, retention_fraction, rmse
from vectorchain.similarity import FeatureStandardizer, Neighbor, dtw_distance, nearest_neighbors
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
    "FeatureStandardizer",
    "FixedSegmentation",
    "Neighbor",
    "RngLike",
    "Segment",
    "VectorChain",
    "__version__",
    "compression_factor",
    "dtw_distance",
    "first_difference",
    "first_second_difference",
    "fixed_linear_segmentation",
    "generate_chirp",
    "generate_first_order_response",
    "generate_piecewise_linear",
    "generate_ramp",
    "generate_regime_change",
    "generate_second_order_response",
    "generate_sine",
    "mae",
    "nearest_neighbors",
    "normalized_raw_values",
    "raw_values",
    "retention_fraction",
    "rmse",
]
