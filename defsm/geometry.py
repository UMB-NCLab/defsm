"""Planar depth, geometric median, convex lifting and the safety deficit."""
import math
from itertools import combinations
import numpy as np
from scipy.optimize import linprog, minimize

EPS = 1e-9


def depth_many(C, U):
    """Exact Tukey depth of each row of C (m x 2) with respect to the points U (n x 2)."""
    V = U[None, :, :] - C[:, None, :]
    ang = np.arctan2(V[..., 1], V[..., 0])
    base = np.concatenate([ang + np.pi / 2, ang - np.pi / 2], axis=1)
    cand = np.concatenate([base + 1e-7, base - 1e-7], axis=1)
    Y = np.stack([np.cos(cand), np.sin(cand)], axis=-1)
    proj = np.einsum("mnk,mdk->mnd", V, Y)
    at = np.abs(V).sum(-1) < EPS
    return ((proj >= -EPS) | at[..., None]).sum(1).min(1)


def deepest(U, q=12):
    """A deepest point of U: best of the data points and the intersections of lines through
    pairs of the q deepest data points. Returns (point, depth)."""
    U = np.asarray(U, float); n = len(U)
    keep = np.argsort(-depth_many(U, U), kind="stable")[:min(q, n)]
    lines = [(U[i], U[j] - U[i]) for i, j in combinations(keep, 2)]
    P = np.array([l[0] for l in lines]); R = np.array([l[1] for l in lines])
    ia, ib = np.triu_indices(len(lines), 1)
    den = R[ia, 0] * R[ib, 1] - R[ia, 1] * R[ib, 0]
    ok = np.abs(den) > 1e-12
    diff = P[ib] - P[ia]
    t = (diff[:, 0] * R[ib, 1] - diff[:, 1] * R[ib, 0]) / np.where(ok, den, 1)
    X = (P[ia] + t[:, None] * R[ia])[ok]
    C = np.unique(np.round(np.r_[U, X], 10), axis=0)
    D = np.concatenate([depth_many(C[s:s + 2000], U) for s in range(0, len(C), 2000)])
    u = C[D == D.max()].mean(0)
    return u, int(depth_many(u[None], U)[0])


def geomed(W, it=200):
    """Geometric median by Weiszfeld iterations."""
    z = W.mean(0)
    for _ in range(it):
        q = 1 / np.maximum(np.linalg.norm(W - z, axis=1), 1e-9)
        z = (q[:, None] * W).sum(0) / q.sum()
    return z


def safegm_point(U):
    """SafeGM decision in the plane: the geometric median if its depth is at least ceil(n/3),
    otherwise a deepest point. Returns (point, depth, used_gm)."""
    g = geomed(U); dg = int(depth_many(g[None], U)[0])
    if dg >= math.ceil(len(U) / 3):
        return g, dg, 1
    u, D = deepest(U)
    return u, D, 0


def _min_norm_weights(U, u, cap):
    n = len(U)
    cons = [{"type": "eq", "fun": lambda l: U.T @ l - u, "jac": lambda l: U.T},
            {"type": "eq", "fun": lambda l: l.sum() - 1.0, "jac": lambda l: np.ones((1, n))}]
    res = minimize(lambda l: l @ l, np.full(n, 1.0 / n), jac=lambda l: 2 * l, method="SLSQP",
                   bounds=[(0.0, cap)] * n, constraints=cons, options={"ftol": 1e-12, "maxiter": 500})
    lam = np.clip(res.x, 0, None)
    return lam / lam.sum()


def capped_lift(U, u):
    """Convex weights lam with sum_j lam_j U_j = u whose largest weight is minimal (min-norm among those)."""
    U = np.asarray(U, float); n, k = U.shape
    c = np.r_[np.zeros(n), 1.0]
    A_eq = np.r_[np.c_[U.T, np.zeros((k, 1))], [np.r_[np.ones(n), 0.0]]]
    A_ub = np.c_[np.eye(n), -np.ones((n, 1))]
    res = linprog(c, A_ub=A_ub, b_ub=np.zeros(n), A_eq=A_eq, b_eq=np.r_[u, 1.0],
                  bounds=[(0, None)] * (n + 1), method="highs")
    if res.status != 0:
        raise ValueError("point outside the convex hull")
    return _min_norm_weights(U, u, res.x[-1] * (1 + 1e-7) + 1e-12)


def basic_lift(Z, z, cap):
    """Basic solution of {lam : sum_j lam_j Z_j = z, sum lam = 1, 0 <= lam <= cap} (sparse), or None."""
    m = len(Z)
    res = linprog(np.zeros(m), A_eq=np.r_[Z.T, np.ones((1, m))], b_eq=np.r_[z, 1.0],
                  bounds=[(0, cap)] * m, method="highs-ds")
    if res.status != 0:
        return None
    lam = np.clip(res.x, 0, None)
    return lam / lam.sum()


def safety_deficit(w, G, r):
    """Smallest a such that w = sum_g mu_g g + q with mu >= 0, sum mu = 1 - a, q in a*P (exact LP)."""
    w, G = np.asarray(w, float), np.asarray(G, float)
    m, d = G.shape; nv = m + d + 1
    c = np.zeros(nv); c[-1] = 1.0
    A_eq = np.zeros((d + 1, nv))
    A_eq[:d, :m] = G.T; A_eq[:d, m:m + d] = np.eye(d)
    A_eq[d, :m] = 1.0; A_eq[d, -1] = 1.0
    A_ub = np.zeros((d + 1, nv))
    A_ub[:d, m:m + d] = np.eye(d); A_ub[:d, -1] = -1.0
    A_ub[d, m:m + d] = 1.0; A_ub[d, -1] = -float(r)
    res = linprog(c, A_ub=A_ub, b_ub=np.zeros(d + 1), A_eq=A_eq, b_eq=np.r_[w, 1.0],
                  bounds=[(0, None)] * nv, method="highs")
    return float(res.x[-1]) if res.status == 0 else 1.0
