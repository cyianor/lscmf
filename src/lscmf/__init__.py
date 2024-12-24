r"""
Large-scale Collective Matrix Factorization
===========================================

The goal of this project is to conceive of a method that makes it
possible to extend the data integration problem formulated below
to large scale inputs.

Given a sequence of matrices :math:`X_1, \dots, X_m` and a sequence
of tuples :math:`(r_1, c_1), \dots, (r_m, c_m)` of row and column
entities for each matrix, the goal is to find shared representations

.. math::
    X_i = U_{r_i} D_i U_{c_i}^\top

for all :math:`i`. The overarching goal is to determine
:math:`U_{r_1}, \dots, U_{r_m}` and :math:`D_1, \dots, D_m`.
"""

__all__ = [
    "__title__",
    "__description__",
    "__version__",
    "__author__",
    "__author_email__",
    "__license__",
    "__copyright__",
    "MixedContinuousDiscreteDistribution",
    "MarchenkoPastur",
    "estimate_noise_level",
    "scale_noise_level",
    "Shrinker",
    "FrobeniusHardThreshold",
    "FrobeniusShrinker",
    "OperatorShrinker",
    "NuclearShrinker",
    "SingularValueShrinker",
    "Properties",
    "PropertyDict",
    "Edge",
    "UndirectedGraph",
    "MultiEdge",
    "UndirectedMultiGraph",
    "HyperGraph",
    "MatchNode",
    "MatchEdge",
    "MatchGraph",
    "ViewGraph",
    "estimate_signal",
    "estimate_angles",
    "joint_matrix",
    "precompute",
    "match_factors",
    "asymptotic_cosine_left",
    "asymptotic_cosine_right",
    "asymptotic_signal_singular_value",
    "subspaces",
    "simulate",
    "bicenter",
    "LargeScaleCMF",
]

from .__about__ import (
    __title__,
    __description__,
    __version__,
    __author__,
    __author_email__,
    __license__,
    __copyright__,
)

from .dist import (
    MixedContinuousDiscreteDistribution,
    MarchenkoPastur,
)
from .scaling import (
    estimate_noise_level,
    scale_noise_level,
)
from .shrinkers import (
    Shrinker,
    FrobeniusHardThreshold,
    FrobeniusShrinker,
    OperatorShrinker,
    NuclearShrinker,
    SingularValueShrinker,
)
from .graphs import (
    Properties,
    PropertyDict,
    Edge,
    UndirectedGraph,
    MultiEdge,
    UndirectedMultiGraph,
    HyperGraph,
    MatchNode,
    MatchEdge,
    MatchGraph,
    ViewGraph,
)
from .matching import (
    estimate_signal,
    estimate_angles,
    joint_matrix,
    precompute,
    match_factors,
)
from .helpers import (
    asymptotic_cosine_left,
    asymptotic_cosine_right,
    asymptotic_signal_singular_value,
)
from .functional import subspaces
from .simulate import simulate
from .preprocess import bicenter
from .lscmf import LargeScaleCMF
