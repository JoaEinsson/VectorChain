"""Optional Matplotlib visualizations for VectorChain results."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike, NDArray

from vectorchain.core import VectorChain

if TYPE_CHECKING:
    from matplotlib.axes import Axes


def plot_vector_chain(
    x: ArrayLike,
    chain: VectorChain,
    *,
    ax: Axes | None = None,
    title: str | None = None,
) -> Axes:
    """Plot observations, reconstruction, vector segments, and articulation points.

    Matplotlib is an optional dependency. Install ``vectorchain[plot]`` when this
    function is used outside the development environment.
    """

    try:
        import matplotlib.pyplot as plt
    except ImportError as error:  # pragma: no cover - depends on optional environment
        msg = "plot_vector_chain requires the optional 'plot' dependency"
        raise ImportError(msg) from error

    observed = _validate_observed(x, chain)
    reconstructed = chain.inverse_transform(chain.vectors_)
    if ax is None:
        _, ax = plt.subplots(figsize=(10.0, 4.5), constrained_layout=True)

    sample_index = np.arange(chain.n_samples_)
    ax.plot(
        sample_index,
        observed,
        color="0.2",
        linewidth=1.0,
        alpha=0.8,
        label="Original",
        zorder=4,
    )
    ax.plot(
        sample_index,
        reconstructed,
        color="tab:blue",
        linewidth=1.4,
        linestyle="--",
        label="Reconstruction",
        zorder=3,
    )

    for index, segment in enumerate(chain.segments_):
        ax.plot(
            [segment.start, segment.end],
            [reconstructed[segment.start], reconstructed[segment.end]],
            color="tab:orange",
            linewidth=3.0,
            alpha=0.45,
            label="Vector segments" if index == 0 else None,
            zorder=2,
        )

    hinge_indices = np.unique(chain.segment_boundaries_.reshape(-1))
    ax.scatter(
        hinge_indices,
        reconstructed[hinge_indices],
        color="tab:red",
        edgecolor="white",
        linewidth=0.6,
        s=30.0,
        label="Articulation points",
        zorder=5,
    )
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Value")
    ax.set_title(
        title
        if title is not None
        else f"VectorChain articulated reconstruction (n={chain.n_samples_}, tol={chain.tolerance:g})"
    )
    ax.margins(x=0.01)
    ax.grid(True, alpha=0.25)
    ax.legend()
    return ax


def _validate_observed(x: ArrayLike, chain: VectorChain) -> NDArray[np.float64]:
    try:
        observed = np.asarray(x)
    except (TypeError, ValueError) as error:
        msg = "x must be a one-dimensional real numeric series"
        raise ValueError(msg) from error
    if observed.ndim != 1:
        msg = "x must be a one-dimensional real numeric series"
        raise ValueError(msg)
    if observed.size != chain.n_samples_:
        msg = f"x must contain exactly {chain.n_samples_} observations"
        raise ValueError(msg)
    if not np.issubdtype(observed.dtype, np.number) or np.issubdtype(
        observed.dtype, np.complexfloating
    ):
        msg = "x must contain real numeric observations"
        raise TypeError(msg)
    validated = observed.astype(np.float64, copy=False)
    if not np.all(np.isfinite(validated)):
        msg = "x must contain only finite observations"
        raise ValueError(msg)
    return validated
