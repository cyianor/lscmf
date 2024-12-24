from numpy import (
    zeros_like,
    sqrt,
    pi,
    float64,
)
from numpy.typing import NDArray
from typing import Iterator
from .base import MixedContinuousDiscreteDistribution


class MarchenkoPastur(MixedContinuousDiscreteDistribution):
    __slots__ = "beta"

    def __init__(self, beta: float) -> None:
        super().__init__()

        if beta < 0.0:
            raise ValueError("beta is required to be a non-negative scalar")

        self.beta = beta
        self.lower: float = (1 - sqrt(beta)) ** 2
        self.upper: float = (1 + sqrt(beta)) ** 2

    def _continuous_fn(self, x: NDArray[float64]) -> NDArray[float64]:
        denominator = sqrt((self.upper - x) * (x - self.lower))
        idx = denominator > 1e-8
        out = zeros_like(x)
        out[idx] = denominator[idx] / (2.0 * pi * self.beta * x[idx])
        return out

    def _discrete_fn(self) -> Iterator[tuple[float, float]]:
        if self.beta > 1.0:
            yield from [(0.0, 1.0 - 1.0 / self.beta)]
        else:
            yield from ()
