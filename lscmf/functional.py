from numpy import (
    arccos,
    arcsin,
    zeros,
    diag,
    concatenate,
    float64,
    ndim,
    logical_not,
)
from numpy.linalg import svd
from numpy.typing import NDArray


def subspaces(
    qf: NDArray[float64], qg: NDArray[float64]
) -> tuple[NDArray[float64], NDArray[float64], NDArray[float64]]:
    """Find principal angles and vectors.

    Implementation of Algorithm 3.2 in Knyazev AV, Argentati ME "Principal
    Angles between Subspaces in an A-based Scalar Product: Algorithms and
    Perturbation Estimates", SIAM J. Sci. Comput. 23(6):2008-2040,
    DOI 10.1137/S1064827500377332

    Parameters
    ----------
    qf : ndarray[float64]
        Orthonormal basis of first subspace (n x p1)
    qg : ndarray[float64]
        Orthonormal basis of second subspace (n x p2)

    Returns
    -------
    u : ndarray[float64]
        Left principal vectors
    theta : ndarray[float64]
        Principal angles
    vt : ndarray[float64]
        Right principal vectors

    Raises
    ------
    ValueError
        If qf or qg are not matrices.
    """
    if ndim(qf) != 2 or ndim(qg) != 2:
        raise ValueError("qf and qg need to be matrices")
    q = min(qf.shape[1], qg.shape[1])
    y, s, zt = svd(qf.T @ qg, full_matrices=False)
    u_cos = qf @ y
    v_cos = qg @ zt.T
    idx = s**2 < 0.5
    theta = zeros((q,), dtype=float64)
    theta[idx] = arccos(s[idx])
    u = u_cos[:, idx]
    v = v_cos[:, idx]
    rf = u_cos[:, logical_not(idx)]
    rg = v_cos[:, logical_not(idx)]
    s_r = s[logical_not(idx)]

    b = rg - qf @ (qf.T @ rg)
    y, mu, zt = svd(b, full_matrices=False)
    v_sin = rg @ zt.T
    u_sin = rf @ (rf.T @ v_sin) @ diag(1.0 / s_r)
    idx = mu**2 <= 0.5
    theta[: len(idx)][idx] = arcsin(mu[idx])[::-1]
    u = concatenate([u_sin[:, idx][:, ::-1], u], axis=1)
    v = concatenate([v_sin[:, idx][:, ::-1], v], axis=1)

    return u, theta, v.T
