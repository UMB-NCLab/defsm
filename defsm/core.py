"""FSM problem: data, facility-location objectives, client oracle, attacks, rounding."""
import numpy as np

_DATA = {}


def load_dataset(name, folder="."):
    """ResNet18 features and labels produced by make_features.py, standardized per dimension."""
    if name not in _DATA:
        X = np.load(f"{folder}/{name}_features.npy").astype(float)
        y = np.load(f"{folder}/{name}_labels.npy").astype(int)
        _DATA[name] = ((X - X.mean(0)) / (X.std(0) + 1e-9), y)
    return _DATA[name]


def make_problem(seed, feats, labels, d, n, alpha, pts=40, bandwidth=0.5):
    """Ground set of d images; n clients with pts images each from a Dirichlet(alpha) class mixture.
    Client i's objective is f_i(S) = sum_{p in D_i} max_{s in S} exp(-||p - s||^2 / (bandwidth * h)^2)."""
    rng = np.random.default_rng(seed)
    n_cls = int(labels.max()) + 1
    E = feats[rng.choice(len(feats), d, replace=False)]
    by_cls = [np.flatnonzero(labels == c) for c in range(n_cls)]
    h = np.median(np.linalg.norm(E[rng.integers(0, d, 500)] - E[rng.integers(0, d, 500)], axis=1))
    sims = []
    for _ in range(n):
        cls = rng.choice(n_cls, pts, p=rng.dirichlet(alpha * np.ones(n_cls)))
        idx = np.array([rng.choice(by_cls[c]) for c in cls])
        D2 = ((feats[idx][:, None, :] - E[None]) ** 2).sum(-1)
        sims.append(np.exp(-D2 / (bandwidth * h) ** 2))
    return sims


def f_val(S, sim):
    S = list(S)
    return float(sim[:, S].max(1).sum()) if S else 0.0


def grad_est(x, sim, rng, m):
    """Unbiased estimate of the gradient of the multilinear extension at x from m random sets."""
    d = len(x); g = np.zeros(d); npts = sim.shape[0]
    for _ in range(m):
        R = np.flatnonzero(rng.random(d) < x)
        if len(R) == 0:
            g += sim.sum(0); continue
        sR = sim[:, R]
        o = np.argsort(-sR, axis=1)
        b1 = sR[np.arange(npts), o[:, 0]]
        b2 = sR[np.arange(npts), o[:, 1]] if len(R) > 1 else np.zeros(npts)
        gain = np.maximum(sim - b1[:, None], 0).sum(0)
        inR = np.zeros(d)
        np.add.at(inR, R[o[:, 0]], b1 - b2)
        gain[R] = inR[R]
        g += gain
    return g / m


def vertex(idx, d):
    v = np.zeros(d); v[np.asarray(idx, int)] = 1.0
    return v


def topr(v, r):
    return np.argsort(-v, kind="stable")[:r]


def lowest(x, k):
    return np.argsort(x, kind="stable")[:k]


def near_dup(v, pool, rng):
    """Swap one of the r ones of v for an element of pool (the result is still binary with r ones)."""
    v = v.copy(); off = [e for e in pool if v[e] == 0]
    if off:
        v[rng.choice(np.flatnonzero(v))] = 0; v[rng.choice(off)] = 1
    return v


def bad_updates(attack, x, Wg, hw, nb, r, rng):
    """Updates of the nb bad clients (Section VI-A). hw: an honest gradient estimate."""
    d = len(x)
    if attack == "include":
        return np.array([vertex(lowest(x, r), d) for _ in range(nb)])
    if attack == "camouflage":
        h = topr(hw, r)[: r // 2]
        rest = [e for e in lowest(x, d) if e not in set(h)][: r - len(h)]
        return np.array([vertex(np.r_[h, rest], d) for _ in range(nb)])
    if attack == "random":
        return np.array([vertex(rng.choice(d, r, replace=False), d) for _ in range(nb)])
    if attack == "near_include":
        base, pool = vertex(lowest(x, r), d), lowest(x, 2 * r)
        return np.array([near_dup(base, pool, rng) for _ in range(nb)])
    if attack == "mimic":
        worst, out = lowest(x, r), []
        for _ in range(nb):
            v = Wg[rng.integers(len(Wg))].copy()
            off = [e for e in worst if v[e] == 0]; k = min(r // 2, len(off))
            v[rng.choice(np.flatnonzero(v), k, replace=False)] = 0; v[rng.choice(off, k, replace=False)] = 1
            out.append(v)
        return np.array(out)
    raise ValueError(attack)


def in_P(w, r, tol=1e-6):
    return w.min() >= -tol and w.max() <= 1 + tol and w.sum() <= r + 1e-4 * max(r, 1)


def to_valid(w, r, rng):
    """Keep w if it lies in P; otherwise map it to the vertex of its r largest coordinates."""
    if in_P(w, r):
        return w
    return vertex(np.argsort(-(w + rng.random(len(w)) * 1e-9), kind="stable")[:r], len(w))


def uniq(V):
    return np.unique(np.round(V, 9), axis=0)


def pipage_round(x, r, rng):
    """Randomized pipage rounding of x (rescaled to sum r) to a set of r elements."""
    x = np.clip(x.astype(float), 0, 1); x *= r / x.sum()
    x = np.clip(x, 0, 1)
    for _ in range(10 * len(x)):
        frac = np.flatnonzero((x > 1e-9) & (x < 1 - 1e-9))
        if len(frac) < 2:
            break
        i, j = frac[0], frac[1]
        a, b = min(1 - x[i], x[j]), min(x[i], 1 - x[j])
        if rng.random() < b / (a + b):
            x[i] += a; x[j] -= a
        else:
            x[i] -= b; x[j] += b
    return np.argsort(-x, kind="stable")[:r]
