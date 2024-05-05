from typing import Type
from numpy import (
    float64,
    concatenate,
    sqrt,
    abs,
    arccos,
    sin,
    cos,
    asarray,
    newaxis,
    logical_and,
    nditer,
    minimum,
    zeros,
    bool_,
    sign,
)
from numpy.linalg import svd
from numpy.typing import NDArray, ArrayLike
from sklearn.decomposition import TruncatedSVD
from collections.abc import Hashable

from .shrinkers import Shrinker, SingularValueShrinker
from .graphs import MatchNode, MatchGraph, ViewGraph
from .helpers import (
    asymptotic_cosine_left,
    asymptotic_cosine_right,
    asymptotic_signal_singular_value,
)
from .scaling import scale_noise_level
from copy import deepcopy


def estimate_signal(
    x: ArrayLike,
    spectrum: ArrayLike,
    beta: float,
    base_shrinker: Type[Shrinker],
    full_svd: bool,
) -> tuple[NDArray[float64], NDArray[float64], NDArray[float64], int]:
    """Estimate low-rank signal.

    Uses ``numpy.linalg.svd`` unless ``full_svd`` is set to ``False``. In the
    latter case, ``sklearn.decomposition.TruncatedSVD`` is used instead.

    Parameters
    ----------
    x : array-like
        The noisy ``n x m`` input matrix
    base_shrinker : lscmf.shrinkers.Shrinker
        A shrinker used in the singular value shrinkage step
    full_svd : bool
        Whether to use ``numpy.linalg.svd`` (True) or
        ``sklearn.decomposition.TruncatedSVD`` (False).

    Returns
    -------
    (u, v, s, n) : ({n, k} ndarray[float64], {m, k} ndarray[float64],
                        {k} ndarray[float64], int)
        Estimates of the left and right singular values in ``u`` and ``v``,
        respectively, and singular values after shrinkage in ``s``, as well
        as the number of retained factors in ``n``.
    """
    x = asarray(x, dtype=float64)
    spectrum = asarray(spectrum, dtype=float64)
    sv_shrinker = SingularValueShrinker(base_shrinker, only_nonzero=True)

    sv_shrunk = sv_shrinker(spectrum, beta)
    n = len(sv_shrunk)
    if n > 0:
        if full_svd:
            u, s, vt = svd(x, full_matrices=False)
        else:
            tsvd = TruncatedSVD(n_components=n, algorithm="arpack")
            usigma = tsvd.fit_transform(x)
            s = tsvd.singular_values_
            u = usigma / s
            vt = tsvd.components_
    else:
        u = zeros((x.shape[0], 0))
        vt = zeros((0, x.shape[1]))

    return (
        asarray(u[:, :n], dtype=float64),
        asarray(vt.T[:, :n], dtype=float64),
        sv_shrunk,
        n,
    )


def estimate_angles(
    spectrum: NDArray[float64], rank: int, beta: float, transposed: bool
) -> tuple[NDArray[float64], NDArray[float64]]:
    """Estimate asymptotic angles from singular values.

    Note that the formulas for angles with the left and right singular
    values are slightly different. So if ``beta > 1.0`` for a matrix and
    the transpose is considered, then the role of left and right singular
    values is flipped around.

    Parameters
    ----------
    spectrum : ndarray[float64]
        Spectrum of the data matrix (without shrinkage)
    rank : int
        Low-rank of the data matrix
    beta : float
        Proportion of rows to columns.
    transposed : bool
        ``True`` if ``beta`` refers to the ratio of columns to rows instead.
    """
    angles_left = arccos(
        asymptotic_cosine_left(
            asymptotic_signal_singular_value(spectrum[:rank], beta),
            beta,
        )
    )

    angles_right = arccos(
        asymptotic_cosine_right(
            asymptotic_signal_singular_value(spectrum[:rank], beta),
            beta,
        )
    )

    if transposed:
        return angles_right, angles_left

    return angles_left, angles_right


def joint_matrix(
    G: ViewGraph, view: Hashable
) -> tuple[NDArray[float64], NDArray[float64], float, bool]:
    joint, spectrum, _ = scale_noise_level(
        concatenate(
            [
                x * sqrt(max(x.shape))
                for x in [
                    G[e]["x"] if G[e]["views"][0] == view else G[e]["x"].T
                    for e in G.incident_edges(view)
                ]
            ],
            axis=1,
        )
    )

    beta = joint.shape[0] / joint.shape[1]
    transposed = False
    if beta > 1.0:
        beta = 1.0 / beta
        transposed = True

    return (joint, spectrum, beta, transposed)


def precompute(
    G: ViewGraph, base_shrinker: Type[Shrinker], *, full_svd: bool = False
) -> None:
    G.edges.update_properties(
        scale_noise_level,
        inputs=("x",),
        outputs=("x", "spectrum", "scale"),
    )
    G.edges.update_properties(
        lambda x, spectrum, beta: estimate_signal(
            x, spectrum, beta, base_shrinker, full_svd
        ),
        inputs=("x", "spectrum", "beta"),
        outputs=("u", "v", "s", "rank"),
    )
    G.edges.update_properties(
        estimate_angles,
        inputs=("spectrum", "rank", "beta", "transposed"),
        outputs=("angles_left", "angles_right"),
    )

    # Reduction: Each incident edge of a node represents essentially another
    #            node. The data in these nodes is reduced to the joint matrix.
    for j in G.nodes:
        joint, spectrum, beta, transposed = joint_matrix(G, j)

        u, _, s, rank = estimate_signal(
            joint, spectrum, beta, base_shrinker, full_svd
        )

        angles_left, _ = estimate_angles(spectrum, rank, beta, transposed)

        G.nodes[j].update(
            # x=joint, # Takes a lot of memory and can always be recreated
            spectrum=spectrum,
            u=u,
            s=s,
            rank=rank,
            beta=beta,
            angles_left=angles_left,
        )


def match_factors(
    G: ViewGraph,
    *,
    use_lower_only: bool = True,
    use_nonmatch_upper: bool = True,
    nocomp_length1_nodes: bool = True,
    require_match_lower_ge_nonmatch_upper: bool = True,
    lower_min: float = 0.0,
    lower_max: float = 1.0,
) -> MatchGraph:
    match_graphs: list[MatchGraph] = []

    for j, (u_joint, angles_joint) in G.nodes.items_filtered(
        "u", "angles_left"
    ):
        # If we did not find any joint factors then there is nothing
        # to be matched
        if u_joint.shape[1] == 0:
            continue

        M = MatchGraph()

        edges = list(G.incident_edges(j))
        if nocomp_length1_nodes and len(edges) == 1:
            e = edges[0]
            if j == G[e]["views"][0]:
                sgn = sign(u_joint.T @ G[e]["u"])
            else:
                sgn = sign(u_joint.T @ G[e]["v"])

            for idx1, idx2 in zip(
                range(u_joint.shape[1]), range(u_joint.shape[1])
            ):
                M.add_relationship(
                    idx1,
                    MatchNode(edges[0], idx2),
                    node_data={
                        (j, idx1): {
                            "cos_match": 1.0,
                            "cos_match_signed": sgn[idx1, idx2],
                            "lower_bound": 1.0,
                            "upper_bound": 1.0,
                            "nonmatch_upper_bound": 0.0,
                        }
                    },
                )
        else:
            for e in edges:
                if j == G[e]["views"][0]:
                    # Edge is irrelevant if no factors were found
                    if G[e]["u"].shape[1] == 0:
                        continue
                    angles: NDArray[float64] = G[e]["angles_left"]
                    cos_match_signed: NDArray[float64] = u_joint.T @ G[e]["u"]
                else:
                    # Edge is irrelevant if no factors were found
                    if G[e]["v"].shape[1] == 0:
                        continue
                    angles: NDArray[float64] = G[e]["angles_right"]
                    cos_match_signed: NDArray[float64] = u_joint.T @ G[e]["v"]

                cos_match: NDArray[float64] = abs(cos_match_signed)

                # For technical reasons: Numerical errors can sometimes lead
                # to cos_match slightly overshooting 1
                cos_match = minimum(cos_match, 1.0)

                lower_bound_angles: NDArray[float64] = (
                    angles_joint[:, newaxis] + angles[newaxis, :]
                )
                # No need to adjust for overshooting of angles
                # lower_bound_angles[lower_bound_angles > pi / 2] = (
                #     pi - lower_bound_angles[lower_bound_angles > pi / 2]
                # )
                lower_bound: NDArray[float64] = cos(lower_bound_angles)
                lower_bound[lower_bound < lower_min] = lower_min
                lower_bound[lower_bound > lower_max] = lower_max

                # No need to take absolute value of angles since
                # cos(-x) = cos(x)
                upper_bound: NDArray[float64] = cos(
                    angles_joint[:, newaxis] - angles[newaxis, :]
                )

                nonmatch_upper_bound: NDArray[float64] = sin(
                    lower_bound_angles
                ) + sin(angles_joint[:, newaxis]) * sin(angles[newaxis, :])

                if use_lower_only:
                    matches = lower_bound <= cos_match
                else:
                    matches: NDArray[bool_] = logical_and(
                        lower_bound <= cos_match,
                        upper_bound >= cos_match,
                    )

                if use_nonmatch_upper:
                    matches = logical_and(
                        matches, nonmatch_upper_bound <= cos_match
                    )

                if require_match_lower_ge_nonmatch_upper:
                    matches = logical_and(
                        matches, nonmatch_upper_bound <= lower_bound
                    )

                it = nditer(matches, flags=["multi_index"])
                for m in it:
                    if m:
                        idx1 = it.multi_index[0]
                        idx2 = it.multi_index[1]
                        node_data = {
                            (j, idx1): {
                                "cos_match": cos_match[idx1, idx2],
                                "cos_match_signed": cos_match_signed[
                                    idx1, idx2
                                ],
                                "lower_bound": lower_bound[idx1, idx2],
                                "upper_bound": upper_bound[idx1, idx2],
                                "nonmatch_upper_bound": nonmatch_upper_bound[
                                    idx1, idx2
                                ],
                            }
                        }

                        M.add_relationship(
                            idx1,
                            MatchNode(e, idx2),
                            node_data=node_data,
                        )

        match_graphs.append(M)

    while len(match_graphs) > 1:
        m1 = match_graphs.pop()
        m2 = match_graphs.pop()

        # print("m1:", m1.graph)
        # print("m2:", m2.graph)

        intersected_nodes = set(m1.nodes).intersection(set(m2.nodes))
        # print(intersected_nodes)

        for n in intersected_nodes:
            # print(n)
            e1 = m1.incident_edges(n)
            # We might have removed n from m2 already if it was included in a
            # previously changed hyperedge
            if n in m2.nodes:
                e2 = m2.incident_edges(n)
            else:
                continue

            if len(e1) != 1 or len(e2) != 1:
                raise ValueError(
                    "Expected node to be present in exactly one hyperedge. "
                    f"Node {n} is in {len(e1)} edges in the first hypergraph "
                    f"and in {len(e2)} edges in the second hypergraph."
                )
            e1 = e1[0]
            e2 = e2[0]

            # Merge hyperedges
            for key in m2[e2]:
                if key not in m1.nodes:
                    m1.add_node(key, **m2.nodes[key])
                else:
                    m1.nodes[key]["node_data"].update(
                        m2.nodes[key]["node_data"]
                    )

            m1.edges[e1].update(m2.edges[e2])

            m1[e1] = m1[e1].union(m2[e2])

            # e2 might overlap with hyperedges in m1 other than e1
            merge_edges = []
            for e in m1.edges:
                if e != e1 and len(m1[e].intersection(m2[e2])) > 0:
                    # e == e1 in that case
                    merge_edges.append(e)

            for e in merge_edges:
                m1.edges[e1].update(m1.edges[e])
                m1[e1] = m1[e1].union(m1[e])

                del m1[e]
                del m1.edges[e]

            # Cleanup
            for key in m2[e2]:
                del m2.nodes[key]
            del m2[e2]
            del m2.edges[e2]

            # print("m1:", m1.graph)
            # print("m2:", m2.graph)

        for n, prop in m2.nodes.items():
            if n not in m1.nodes:
                m1.add_node(n, **prop)

        new_edge = max(m1.edges.keys()) + 1
        for e, prop in m2.edges.items():
            m1.add_edge(new_edge, **prop)
            if len(m2[e]) > 0:
                m1[new_edge] = m2[e]
            new_edge += 1

        m2.graph.clear()
        m2.edges.clear()
        m2.nodes.clear()

        match_graphs.append(m1)

    return match_graphs[0]
