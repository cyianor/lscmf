from sklearn.base import BaseEstimator
from sklearn.utils._param_validation import StrOptions, Interval
from sklearn.utils.validation import check_array, check_is_fitted
from numbers import Real
from numpy.typing import NDArray
from numpy import float64, zeros, nan, ones, sign, diag
from collections.abc import Hashable
from warnings import warn

from .graphs import ViewGraph, MultiEdge
from .matching import precompute
from .matching import match_factors
from .shrinkers import (
    FrobeniusShrinker,
    NuclearShrinker,
    OperatorShrinker,
    FrobeniusHardThreshold,
)

try:
    from typing import Unpack
except ImportError:
    from typing_extensions import Unpack


ViewDesc = tuple[Hashable, Hashable, Unpack[tuple[Hashable, ...]]]


class LargeScaleCMF(BaseEstimator):
    _parameter_constraints = {
        "shrinker": [
            StrOptions(
                {
                    "frobenius",
                    "nuclear",
                    "operator",
                    "frobenius_hard_threshold",
                }
            ),
        ],
        "use_lower_only": ["boolean"],
        "use_nonmatch_upper": ["boolean"],
        "nocomp_length1_nodes": ["boolean"],
        "require_match_lower_ge_nonmatch_upper": ["boolean"],
        "lower_min": [Interval(Real, 0, 1, closed="both")],
        "lower_max": [Interval(Real, 0, 1, closed="both")],
        "full_svd": ["boolean"],
    }

    def __init__(
        self,
        shrinker: str = "frobenius",
        use_lower_only: bool = True,
        use_nonmatch_upper: bool = True,
        nocomp_length1_nodes: bool = True,
        require_match_lower_ge_nonmatch_upper: bool = True,
        lower_min: float = 0.0,
        lower_max: float = 1.0,
        full_svd: bool = False,
    ):
        self.shrinker = shrinker
        self.use_lower_only = use_lower_only
        self.use_nonmatch_upper = use_nonmatch_upper
        self.nocomp_length1_nodes = nocomp_length1_nodes
        self.require_match_lower_ge_nonmatch_upper = (
            require_match_lower_ge_nonmatch_upper
        )
        self.lower_min = lower_min
        self.lower_max = lower_max
        self.full_svd = full_svd

    def fit(
        self,
        X: dict[ViewDesc, NDArray[float64]],
        y=None,
    ):
        self._validate_params()

        match self.shrinker:
            case "frobenius":
                base_shrinker = FrobeniusShrinker
            case "nuclear":
                base_shrinker = NuclearShrinker
            case "operator":
                base_shrinker = OperatorShrinker
            case _:
                base_shrinker = FrobeniusHardThreshold

        if self.lower_min > self.lower_max:
            raise ValueError(
                "'lower_min' must be less or equal to 'lower_max'"
            )

        layout = list(X.keys())
        for k in layout:
            X[k] = check_array(X[k])

        G = ViewGraph()
        G.add_data_from(
            [str(k) for k in X.keys()],
            X.values(),
            [k[:2] for k in X.keys()],
        )
        precompute(G, base_shrinker=base_shrinker, full_svd=self.full_svd)

        self.match_graph_ = match_factors(
            G,
            use_lower_only=self.use_lower_only,
            use_nonmatch_upper=self.use_nonmatch_upper,
            nocomp_length1_nodes=self.nocomp_length1_nodes,
            require_match_lower_ge_nonmatch_upper=self.require_match_lower_ge_nonmatch_upper,
            lower_min=self.lower_min,
            lower_max=self.lower_max,
        )

        # Match hypergraph needs to be translated to actual estimates
        # for vs and ds
        key_dict = {MultiEdge(str(k), k[:2]): k for k in layout}

        self.ds_ = {k: zeros(len(self.match_graph_.graph)) for k in layout}
        self.vs_ = {
            k: nan * ones((p, len(self.match_graph_.graph)), dtype=float64)
            for k, (p,) in G.nodes.items_filtered("p")
        }

        for i, (_, s) in enumerate(self.match_graph_.graph.items()):
            for m in s:
                sgn = 1.0
                if len(self.match_graph_.nodes[m]["node_data"]) == 2:
                    for (view, joint_factor), nd in self.match_graph_.nodes[m][
                        "node_data"
                    ].items():
                        sgn *= sign(nd["cos_match_signed"])
                        self.vs_[view][:, i] = G[view]["u"][:, joint_factor]
                elif len(self.match_graph_.nodes[m]["node_data"]) == 1:
                    for (view, joint_factor), nd in self.match_graph_.nodes[m][
                        "node_data"
                    ].items():
                        sgn *= sign(nd["cos_match_signed"])
                        self.vs_[view][:, i] = G[view]["u"][:, joint_factor]

                    if m.data_edge.n[0] == view:
                        j = 1
                    else:
                        j = 0

                    if m.data_edge.n[j] == G[m.data_edge]["views"][0]:
                        self.vs_[m.data_edge.n[j]][:, i] = G[m.data_edge]["u"][
                            :, m.factor
                        ]
                    else:
                        self.vs_[m.data_edge.n[j]][:, i] = G[m.data_edge]["v"][
                            :, m.factor
                        ]
                else:
                    warn(
                        f"Node {m} appears in "
                        f"{len(self.match_graph_.nodes[m]['node_data'])} "
                        "views/joint factors. Should match at most 2."
                    )

                self.ds_[key_dict[m.data_edge]][i] = (
                    sgn * G[m.data_edge]["s"][m.factor]
                )

        for k, s in G.edges.items_filtered("scale"):
            self.ds_[key_dict[k]] /= s

        return self

    def transform(
        self,
        X: dict[ViewDesc, NDArray[float64]],
        y=None,
    ):
        check_is_fitted(self)

        return {
            k: self.vs_[k[0]][:, self.ds_[k] != 0.0]
            @ diag(self.ds_[k][self.ds_[k] != 0.0])
            @ self.vs_[k[1]][:, self.ds_[k] != 0.0].T
            for k in X.keys()
        }

    def score(
        self,
        X: dict[ViewDesc, NDArray[float64]],
    ):
        """Computes the residual SSQ relative to the input data."""
        check_is_fitted(self)

        X_hat = self.transform(X)

        return {
            k: ((x - X_hat[k]) ** 2).sum() / (x**2).sum() for k, x in X.items()
        }

    def _more_tags(self):
        return {
            "X_types": "dict",
        }
