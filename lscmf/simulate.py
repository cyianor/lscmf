from numpy.typing import NDArray
from numpy import float64, diag, sum, sqrt
from numpy.linalg import qr
from numpy.random import Generator, default_rng
from collections.abc import Hashable

from .lscmf import ViewDesc


def simulate(
    *,
    viewdims: dict[Hashable, int],
    factor_scales: dict[
        ViewDesc,
        NDArray[float64],
    ],
    scales: dict[ViewDesc, float] | None = None,
    snr: float = 1.0,
    rng: Generator | None = None,
):
    if rng is None:
        rng = default_rng()

    shapes = [s.shape for s in factor_scales.values()]
    assert all(
        [len(s) == 1 and s == shapes[0] for s in shapes]
    ), "Each value in 'factor_scales' needs to be a of shape (max_rank,)"
    max_rank = shapes[0][0]
    assert all(len(k) >= 2 for k in factor_scales.keys()), (
        "Each key in 'factor_scales' needs to be a tuple of two"
        " or more integers"
    )

    views = set([k[i] for k in factor_scales.keys() for i in range(2)])
    assert views == viewdims.keys(), (
        "The keys of 'viewdims' need to appear in the first two entries"
        " of the keys of 'factor_scales'"
    )

    if scales is None:
        scales = {k: 1.0 for k in factor_scales.keys()}

    assert (
        scales.keys() == factor_scales.keys()
    ), "'scales' needs to be compatible with 'factor_scales'"
    assert all(
        s > 0.0 for s in scales.values()
    ), "Each value in 'scales' needs to be positive"

    vs = {
        k: qr(rng.standard_normal((p, max_rank))).Q
        for k, p in viewdims.items()
    }

    xs_truth = {
        k: scales[k] * vs[k[0]] @ diag(d) @ vs[k[1]].T
        for k, d in factor_scales.items()
    }

    xs = {
        k: x
        + sqrt(sum(x**2) / (snr * x.size)) * rng.standard_normal(size=x.shape)
        for k, x in xs_truth.items()
    }

    return {
        "xs_truth": xs_truth,
        "xs": xs,
        "vs": vs,
    }
