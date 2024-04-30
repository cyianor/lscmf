from numpy import float64, median, sqrt, ndim, asarray
from numpy.linalg import svd
from numpy.typing import NDArray

from .dist import MarchenkoPastur


def estimate_noise_level(x: NDArray[float64]) -> float:
    r"""Estimates the noise level of a matrix.

    Assuming the model :math:`X = Y + \sigma Z` for each matrix in the list,
    the function estimates :math:`\sigma` with the method from
    Gavish and Donoho (2014).
    Note that this method is based on an asymptotic model where both the
    number of rows m and the number of columns n go to infinity such
    that :math:`m / n \rightarrow \beta \in (0, 1]`.

    If :math:`\beta > 1` for an input matrix, the noise level will be
    estimated on the transpose.

    Parameters
    ----------
    x : ndarray[float64]
        A matrix for which the noise level will be estimated.

    Returns
    -------
    noise : float
        Noise level estimate

    Raises
    ------
    TypeError
        Gets thrown if x is not a matrix.
    """
    if ndim(x) != 2:
        raise TypeError("x has to be a matrix")

    n, p = x.shape

    beta = n / p
    if beta > 1.0:
        beta = 1.0 / beta
        p = n

    s = float(median(svd(x, compute_uv=False)))
    return s / float(sqrt(p * MarchenkoPastur(beta).ppf(0.5).item()))


def scale_noise_level(
    x: NDArray[float64],
) -> tuple[NDArray[float64], NDArray[float64], float]:
    r"""Scales the noise levels of a list of matrices.

    Assuming the model :math:`X = Y + \sigma Z` the function estimates
    :math:`\sigma` and scales the matrix to have approximate noise standard
    deviation :math:`1 / \sqrt(\max(n, p))` where :math:`n, p` are the rows
    and columns of :math:`X`.

    Parameters
    ----------
    x : ndarray[float64]
        A matrix to be scaled

    Returns
    -------
    y : ndarray[float64]
        Scaled matrix
    spectrum : ndarray[float64]
        Spectrum of the scaled matrix
    scale : float
        Scaling factor applied to the matrix
    """
    if ndim(x) != 2:
        raise TypeError("x needs to be a matrix")

    n, p = x.shape

    beta = n / p
    if beta > 1.0:
        beta = 1.0 / beta

    spectrum = svd(x, compute_uv=False)
    scale = sqrt(MarchenkoPastur(beta).ppf(0.5)) / median(spectrum)
    return (
        asarray(scale * x, dtype=float64),
        asarray(scale * spectrum, dtype=float64),
        scale,
    )
