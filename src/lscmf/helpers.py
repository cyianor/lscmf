from numpy import sqrt, float64, asarray
from numpy.typing import ArrayLike, NDArray


def asymptotic_cosine_left(x: ArrayLike, beta: float) -> NDArray[float64]:
    r"""Asymptotic cosine of angles for left singular vectors.

    Parameters
    ----------
    x : array-like
        The signal singular values
    beta : float
        The asymptotic ratio of the sequence of matrices

    Returns
    -------
    out : ndarray[float64]
        The cosine of the asymptotic angle(s) between the observed left
        singular vector(s) and the left signal singular vector(s).
    """
    x = asarray(x, dtype=float64)
    return asarray(sqrt((x**4 - beta) / (x**4 + beta * x**2)), dtype=float64)


def asymptotic_cosine_right(x: ArrayLike, beta: float) -> NDArray[float64]:
    r"""Asymptotic cosine of angles for right singular vectors.

    Parameters
    ----------
    x : array-like
        The signal singular values
    beta : float
        The asymptotic ratio of the sequence of matrices

    Returns
    -------
    out : ndarray[float64]
        The cosine of the asymptotic angle(s) between the observed right
        singular vector(s) and the right signal singular vector(s).
    """
    x = asarray(x, dtype=float64)
    return asarray(sqrt((x**4 - beta) / (x**4 + x**2)), dtype=float64)


def asymptotic_signal_singular_value(
    y: ArrayLike, beta: float
) -> NDArray[float64]:
    r"""Return an asymptotic estimate of the signal singular values.

    Given data singular values, this function computes an estimate of
    the signal singular values based on asymptotic limit

    .. math::

        y \rightarrow \sqrt{(x + \frac{1}{x}) (x + \frac{\beta}{x})}

    Parameters
    ----------
    y : array-like
        The data singular values
    beta : float
        The asymptotic ratio of the sequence of matrices

    Returns
    -------
    out : ndarray[float64]
        An estimate of the signal singular values
    """
    y = asarray(y, dtype=float64)
    return asarray(
        sqrt(
            0.5
            * (
                (y**2 - beta - 1.0)
                + sqrt((y**2 - beta - 1.0) ** 2 - 4.0 * beta)
            )
        ),
        dtype=float64,
    )
