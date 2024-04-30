from abc import ABCMeta, abstractmethod

from numpy.typing import ArrayLike, NDArray

from numpy import asarray, ndim, zeros_like, sqrt, float64
from numpy.linalg import svd


class Shrinker(metaclass=ABCMeta):
    """Abstract base class for optimal shrinkers.

    Subclasses need to implement the method :meth:`shrink`, which makes
    the shrinkers into a callable.

    Attributes
    ----------
    beta : float
        Asymptotic ratio
    """

    __slots__ = ("beta",)

    def __init__(self, beta: float) -> None:
        self.beta = beta

    def __call__(self, x: ArrayLike) -> NDArray[float64]:
        x = asarray(x, dtype=float64)
        return self.shrink(x)

    @abstractmethod
    def shrink(self, x: NDArray[float64]) -> NDArray[float64]:
        pass


class FrobeniusHardThreshold(Shrinker):
    r"""Optimal hard threshold for Frobenius norm loss

    Optimal hard threshold for Frobenius norm loss from [1]_. All inputs
    less than

    .. math::

        \sqrt{2(\beta + 1) +
        \frac{8\beta}{(\beta + 1) + \sqrt{\beta^2 + 14\beta + 1}}}

    are set to zero.

    .. [1] Gavish M and Donoho DL (2014) The optimal hard threshold for
           singular values is :math:`4 / \sqrt{3}`.
           IEEE Transactions on Information Theory, 60(8):5040-5053
           DOI 10.1109/tit.2014.2323359

    """

    __slots__ = ("threshold",)

    def __init__(self, beta: float) -> None:
        super().__init__(beta)

        self.threshold = sqrt(
            2.0 * (self.beta + 1.0)
            + 8.0
            * self.beta
            / (self.beta + 1.0 + sqrt(self.beta**2 + 14.0 * self.beta + 1.0))
        )

    def shrink(self, x: NDArray[float64]) -> NDArray[float64]:
        out = zeros_like(x)
        idx = x >= self.threshold
        out[idx] = x[idx]

        return out


class FrobeniusShrinker(Shrinker):
    r"""Optimal shrinker for Frobenius norm loss

    Optimal shrinker for Frobenius norm loss from [1]_. All inputs greater
    or equal to :math:`1 + \sqrt{\beta}` are transformed as

    .. math::

        \frac{1}{x} \sqrt{(x^2 - \beta - 1)^2 - 4\beta}

    Inputs below :math:`1 + \sqrt{\beta}` are set to zero.

    .. [1] Gavish M and Donoho DL (2017) Optimal Shrinkage of Singular Values.
           IEEE Transactions on Information Theory, 63(4):2137-2152
           DOI 10.1109/tit.2017.2653801
    """

    __slots__ = ()

    def shrink(self, x: NDArray[float64]) -> NDArray[float64]:
        out = zeros_like(x)
        idx = x >= 1.0 + sqrt(self.beta)
        out[idx] = (
            sqrt((x[idx] ** 2 - self.beta - 1.0) ** 2 - 4.0 * self.beta)
            / x[idx]
        )

        return out


class OperatorShrinker(Shrinker):
    r"""Optimal shrinker for operator norm loss

    Optimal shrinker for operator norm loss from [1]_. All inputs greater
    or equal to :math:`1 + \sqrt{\beta}` are transformed as

    .. math::

        \frac{1}{\sqrt{2}}
        \sqrt{(x^2 - \beta - 1) + \sqrt{(x^2 - \beta - 1)^2 - 4\beta}}

    Inputs below :math:`1 + \sqrt{\beta}` are set to zero.

    .. [1] Gavish M and Donoho D (2017) Optimal Shrinkage of Singular Values.
           IEEE Transactions on Information Theory, 63(4):2137-2152
           DOI 10.1109/TIT.2017.2653801
    """

    __slots__ = ()

    def shrink(self, x: NDArray[float64]) -> NDArray[float64]:
        out = zeros_like(x)
        idx = x >= 1.0 + sqrt(self.beta)
        out[idx] = sqrt(
            x[idx] ** 2
            - self.beta
            - 1.0
            + sqrt((x[idx] ** 2 - self.beta - 1.0) ** 2 - 4.0 * self.beta)
        ) / sqrt(2.0)

        return out


class NuclearShrinker(Shrinker):
    r"""Optimal shrinker for nuclear norm loss

    Optimal shrinker for nuclear norm loss from [1]_. This is a two-step
    shrinker. In the first step, all inputs :math:`x` greater or equal to
    :math:`1 + \sqrt{\beta}` are transformed as

    .. math::

        y = \frac{1}{\sqrt{2}}
        \sqrt{(x^2 - \beta - 1) + \sqrt{(x^2 - \beta - 1)^2 - 4\beta}}

    Inputs below :math:`1 + \sqrt{\beta}` are set to zero. Then, in a
    second step, all :math:`y^4 \geq \beta + \sqrt{\beta} * y * x` are
    transformed as

    .. math::

        \frac{y^4 - \beta - \sqrt{\beta} y x}{y^2 * x}

    and all other entries are set to zero.

    .. [1] Gavish M and Donoho D (2017) Optimal Shrinkage of Singular Values.
           IEEE Transactions on Information Theory, 63(4):2137-2152
           DOI 10.1109/TIT.2017.2653801
    """

    __slots__ = ()

    def shrink(self, x: NDArray[float64]) -> NDArray[float64]:
        y = zeros_like(x)
        idx = x >= 1.0 + sqrt(self.beta)
        y[idx] = sqrt(
            x[idx] ** 2
            - self.beta
            - 1.0
            + sqrt((x[idx] ** 2 - self.beta - 1.0) ** 2 - 4.0 * self.beta)
        ) / sqrt(2.0)

        out = zeros_like(x)
        idx = y**4 >= self.beta + sqrt(self.beta) * y * x
        out[idx] = (
            y[idx] ** 4 - self.beta - sqrt(self.beta) * y[idx] * x[idx]
        ) / (y[idx] ** 2 * x[idx])

        return out


class SingularValueShrinker:
    """Singular value shrinker based on a base shrinker.

    The argument ``base_shrinker`` is the class object to the desired
    shrinker to be applied to the singular values, i.e. a subclass of
    type ``lscmf.Shrinker``. It will be instantiated with the appropriate
    asymptotic ratio before use.

    If ``only_nonzero`` is set, then only singular values larger than
    zero will be returned.
    """

    __slots__ = ("base_shrinker", "only_nonzero")

    def __init__(
        self, base_shrinker: type[Shrinker], only_nonzero: bool = False
    ) -> None:
        self.base_shrinker = base_shrinker
        self.only_nonzero = only_nonzero

    def __call__(self, x: ArrayLike, beta: float) -> NDArray[float64]:
        if ndim(x) not in [1, 2]:
            raise ValueError(
                "x needs to be a data matrix or a vector (spectrum)"
            )

        x = asarray(x, dtype=float64)

        if ndim(x) == 2:
            spectrum = svd(x, compute_uv=False)
        else:
            spectrum = x
        s = self.base_shrinker(beta)(spectrum)
        if self.only_nonzero:
            s = s[s > 0.0]

        return s
