"""Tests for the optional scientific VectorChain plot."""

from io import BytesIO

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from vectorchain import VectorChain
from vectorchain.plotting import plot_vector_chain


def test_plot_shows_original_reconstruction_segments_and_hinges() -> None:
    signal = np.array([0.0, 1.0, 2.0, 2.0, 2.0])
    chain = VectorChain(tolerance=0.0)
    chain.fit_transform(signal)
    figure, axis = plt.subplots()

    returned = plot_vector_chain(signal, chain, ax=axis, title="Seed 1729")

    assert returned is axis
    assert axis.get_title() == "Seed 1729"
    assert axis.get_xlabel() == "Sample index"
    assert axis.get_ylabel() == "Value"
    assert len(axis.lines) == 2 + len(chain.segments_)
    assert len(axis.collections) == 1
    assert {line.get_label() for line in axis.lines} >= {
        "Original",
        "Reconstruction",
        "Vector segments",
    }
    output = BytesIO()
    figure.savefig(output, format="png")
    assert output.tell() > 0
    plt.close(figure)


def test_plot_can_create_its_own_axes_with_informative_default_title() -> None:
    signal = np.arange(5.0)
    chain = VectorChain(tolerance=0.03)
    chain.fit_transform(signal)

    axis = plot_vector_chain(signal, chain)

    assert "n=5" in axis.get_title()
    assert "tol=0.03" in axis.get_title()
    plt.close(axis.figure)


@pytest.mark.parametrize(
    ("signal", "error"),
    [
        ([[0.0, 1.0]], ValueError),
        ([0.0], ValueError),
        (["zero", "one"], TypeError),
        ([0.0 + 0.0j, 1.0 + 0.0j], TypeError),
        ([0.0, np.nan], ValueError),
    ],
)
def test_plot_rejects_invalid_observed_series(signal: object, error: type[Exception]) -> None:
    chain = VectorChain()
    chain.fit_transform([0.0, 1.0])

    with pytest.raises(error):
        plot_vector_chain(signal, chain)


def test_plot_requires_a_finalized_chain() -> None:
    chain = VectorChain()
    chain.update(0.0)
    chain.update(1.0)

    with pytest.raises(RuntimeError, match="finalized fitted chain"):
        plot_vector_chain([0.0, 1.0], chain)
