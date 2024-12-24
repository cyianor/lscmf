from scipy.integrate import quad
from scipy.optimize import root_scalar
from numpy import (
    any,
    newaxis,
    float64,
    ones_like,
    zeros_like,
    logical_and,
    asarray,
    ndim,
    vectorize,
    squeeze,
    inf,
)
from numpy.typing import ArrayLike, NDArray
from typing import Tuple, Union, Optional, Callable, Iterator
from numpy.random import Generator, default_rng


class MixedContinuousDiscreteDistribution:
    r"""Easy construction of mixed continuous-discrete distributions.

    Overwrite functions ``_continuous_fn`` and ``_discrete_fn`` to specialize
    this class. In addition, set ``lower`` and ``upper`` appropriately.

    Attributes
    ----------
    lower : float
        Lower bound of the distributions continous support
    upper : float
        Upper bound of the distributions continous support
    continuous : bool
        Flag indicating whether there is a continous part
    discrete : bool
        Flag indicating whether there is a discrete part
    """

    __slots__ = ("lower", "upper", "continuous", "discrete")

    def __init__(self) -> None:
        self.lower: float = -inf
        self.upper: float = inf

        self.continuous: bool = True
        self.discrete: bool = True

    def _continuous_fn(self, x: NDArray[float64]) -> NDArray[float64]:
        """Continuous density function from which the distribution is derived.

        The user can expect to receive an ndarray of values within the
        distributions support and should return an ndarray of density values.

        Parameters
        ----------
        x : ndarray[float64]
            Quantiles within distributions upper and lower bounds

        Returns
        -------
        y : ndarray[float64]
            PDF values
        """
        return zeros_like(x)

    def _discrete_fn(self) -> Iterator[tuple[float, float]]:
        """Spikes of the distributions discrete part.

        Can be used to introduce spikes in mixed continuous-discrete
        distributions. Locations have to be yielded in sorted
        order. That way, infinite distributions (e.g. Poisson) are possible
        to implement as well.

        Returns
        -------
        it : iterator[(loc, prob)]
            An iterator return spike locations ``loc`` and the associated
            probability ``prob``.
        """
        yield from ()

    def pdf(self, x: ArrayLike) -> NDArray[float64]:
        r"""Probability density function of the continous part.

        Parameters
        ----------
        x : array-like
            Quantiles at which the PDF is evaluated

        Returns
        -------
        y : ndarray[float64]
            PDF values
        """
        x = asarray(x, dtype=float64)
        retscalar = False
        if ndim(x) == 0:
            x = x[newaxis]
            retscalar = True

        out = zeros_like(x)

        if self.continuous:
            idx = logical_and(x <= self.upper, x >= self.lower)
            out[idx] = self._continuous_fn(x[idx])

        if retscalar:
            return squeeze(out)

        return out

    def pmf(self, x: ArrayLike) -> NDArray[float64]:
        r"""Probability mass function of the discrete part.

        Parameters
        ----------
        x : array-like
            Values at which the PMF is to be evaluated

        Returns
        -------
        y : ndarray[float64]
            Probabilities
        """
        x = asarray(x, dtype=float64)
        retscalar = False
        if ndim(x) == 0:
            x = x[newaxis]
            retscalar = True

        out = zeros_like(x)

        if self.discrete:
            x_max = x.max()
            for loc, prob in self._discrete_fn():
                if loc <= x_max:
                    out[x == loc] = prob
                else:
                    break

        if retscalar:
            return squeeze(out)

        return out

    def cdf(self, x: ArrayLike) -> NDArray[float64]:
        r"""Cumulative distribution function.

        Parameters
        ----------
        x : array-like
            Quantiles at which the CDF is evaluated

        Returns
        -------
        p : ndarray[float64]
            Probabilities
        """
        x = asarray(x, dtype=float64)
        retscalar = False
        if ndim(x) == 0:
            x = x[newaxis]
            retscalar = True

        def integrate(z: float) -> float:
            if z > self.lower:
                return float(quad(self.pdf, self.lower, z)[0])
            else:
                return 0.0

        if self.continuous:
            out = asarray(vectorize(integrate)(x), dtype=float64)
        else:
            out = zeros_like(x)

        if self.discrete:
            x_max = x.max()
            for loc, prob in self._discrete_fn():
                if loc <= x_max:
                    out[x >= loc] += prob
                else:
                    break

        if retscalar:
            return squeeze(out)

        return out

    def ppf(self, x: ArrayLike) -> NDArray[float64]:
        r"""Percentile function of the distribution.

        Parameters
        ----------
        x : array-like
            Probabilites at which the percentile function is evaluated

        Returns
        -------
        q : ndarray[float64]
            Quantiles

        Raises
        ------
        ValueError
            Raised if any elements in ``x`` are not in ``[0, 1]``.
        """
        x = asarray(x, dtype=float64)
        retscalar = False
        if ndim(x) == 0:
            x = x[newaxis]
            retscalar = True

        if any(x <= 0.0) or any(x > 1.0):
            raise ValueError("The elements of x need to be in (0, 1]")

        def f(z: float) -> Callable[[float], NDArray[float64]]:
            def inner(x: float) -> NDArray[float64]:
                return self.cdf(x) - z

            return inner

        def root(z: float) -> float:
            # Exact 0.0 and 1.0 are known analytically and lead
            # to numerical errors, so treat separately
            if z == 0.0:
                return self.lower
            if z == 1.0:
                return self.upper

            return float(root_scalar(f(z), bracket=(self.lower, self.upper)).root)

        out = zeros_like(x, dtype=float64)
        rem = ones_like(x, dtype=bool)

        if self.discrete:
            spike_start = 0.0
            for loc, prob in self._discrete_fn():
                if loc < self.lower:
                    ix = logical_and(x > spike_start, x <= spike_start + prob)
                    out[ix] = loc
                    rem[ix] = False
                    spike_start += prob
                elif loc >= self.lower and loc <= self.upper:
                    spike_start = self.cdf(loc).item()
                    ix = logical_and(x > spike_start - prob, x <= spike_start)
                    out[ix] = loc
                    rem[ix] = False
                else:  # loc > self.upper
                    if spike_start < self.cdf(self.upper):
                        spike_start = self.cdf(self.upper).item()
                    ix = logical_and(x > spike_start, x <= spike_start + prob)
                    out[ix] = loc
                    rem[ix] = False
                    spike_start += prob

                if not any(rem):
                    break

        if self.continuous and any(rem):
            out[rem] = vectorize(root)(x[rem])

        if retscalar:
            return squeeze(out)

        return out

    def rvs(
        self, size: Union[int, Tuple[int, ...]], rg: Optional[Generator] = None
    ) -> NDArray[float64]:
        r"""Generate a random variate from the distribution.

        Parameters
        ----------
        size : int or tuple of ints
            Indicates the shape of the variates to generate
        rg : numpy.random.Generator
            Random number generator. Initialised with
            ``numpy.random.default_rng()`` by default if ``rg is None``.

        Returns
        -------
        x : ndarray[float64]
            Random variates in the given shape
        """
        if rg is None:
            rg = default_rng()
        return self.ppf(rg.uniform(size=size))
