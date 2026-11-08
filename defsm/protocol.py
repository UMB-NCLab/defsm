"""One DeFSM run: clients, agents, contract (Sections IV-C, IV-D), and the baselines of Section VI."""
import numpy as np
from . import core as C
from . import geometry as G

DEFAULT = dict(dataset="cifar10", d=200, n=60, r=10, alpha=0.5, eta=0.01, m=8, max_rounds=300, patience=15,
               M=20, L=20, beta_c=0.2, beta_a=0.3, told=0.0, log_every=5, rounds_samples=10)

METHODS = ("defsm", "defsm_agentmean", "agent_gm", "agent_trim", "agent_mean", "fsm_gm", "ref")
RULES = {"agent_gm": lambda V, k: G.geomed(V),
         "agent_trim": lambda V, k: np.sort(V, axis=0)[k:len(V) - k].mean(0) if len(V) > 2 * k else np.sort(V, axis=0).mean(0),
         "agent_mean": lambda V, k: V.mean(0)}


def _nnz(A):
    return int((np.abs(A) > 1e-9).sum())


def safegm_local(W, Pi, log):
    """An agent's local aggregate: SafeGM on the projections of its distinct updates, capped lift."""
    Wu = C.uniq(W); U = Wu @ Pi.T
    pt, _, used = G.safegm_point(U); log["gm_used"].append(used)
    return G.capped_lift(U, pt) @ Wu


def aggregate(method, W, bad_client, x, r, cfg, rng, log):
    """Global aggregate of one round. W: client updates (good first, then bad)."""
    n, d = W.shape; M, L = cfg["M"], cfg["L"]
    nba = int(round(cfg["beta_a"] * M))
    bad_agents = set(rng.permutation(M)[:nba].tolist())
    samples = [rng.choice(n, L, replace=False) for _ in range(M)]
    Pi = rng.normal(0, 1 / np.sqrt(2), (2, d))
    Pi2 = rng.normal(0, 1 / np.sqrt(2), (2, d))
    if method == "ref":
        return W[~bad_client].mean(0)
    if method == "fsm_gm":
        return C.to_valid(G.geomed(C.uniq(W)), r, rng)

    junk, pool = C.vertex(C.lowest(x, r), d), C.lowest(x, 2 * r)
    tc = cfg["told"]
    k_local = max(0, int(round((cfg["beta_c"] + tc) * L)))
    k_global = max(0, int(round((cfg["beta_a"] + tc) * M)))
    V = np.zeros((M, d))
    for j in range(M):
        if j in bad_agents:
            V[j] = C.near_dup(junk, pool, rng)
        elif method in RULES:
            V[j] = RULES[method](C.uniq(W[samples[j]]), k_local)
        else:
            V[j] = safegm_local(W[samples[j]], Pi, log)
    Vu = C.uniq(V); m = len(Vu)

    if method in RULES or method == "defsm_agentmean":
        w = RULES[method](Vu, k_global) if method in RULES else Vu.mean(0)
        log["cost"].append((_nnz(w), _nnz(Vu), m, m))
        return C.to_valid(w, r, rng)

    Z = Vu @ Pi2.T
    z, D, used = G.safegm_point(Z); log["gm_used"].append(used)
    lam = G.basic_lift(Z, z, 1.0 / D)
    if lam is None:
        lam = G.capped_lift(Z, z)
    w = lam @ Vu; sup = lam > 1e-9
    log["cost"].append((_nnz(w), _nnz(Vu[sup]), int(sup.sum()), m))
    return C.to_valid(w, r, rng)


def run(cfg, seed, method, attack):
    """One run. Returns final qualities, average safety deficit, rounds, curve and per-round costs."""
    X, y = C.load_dataset(cfg["dataset"], cfg.get("data_dir", "data"))
    sims = C.make_problem(seed, X, y, cfg["d"], cfg["n"], cfg["alpha"])
    rng = np.random.default_rng(20_000 + seed)
    d, n, r = cfg["d"], cfg["n"], cfg["r"]
    nb = int(round(cfg["beta_c"] * n)); good = np.arange(n - nb)
    bad_client = np.r_[np.zeros(n - nb, bool), np.ones(nb, bool)]
    fg = lambda S: np.mean([C.f_val(S, sims[i]) for i in good])
    oracle = lambda x, i: C.vertex(C.topr(C.grad_est(x, sims[i], rng, cfg["m"]), r), d)
    log = {"gm_used": [], "cost": []}
    x = np.zeros(d); defs, curve = [], []
    last, stable, min_rounds = None, 0, int(round(1 / cfg["eta"]))
    for t in range(cfg["max_rounds"]):
        Wg = np.array([oracle(x, i) for i in good])
        if nb == 0:
            Wb = np.zeros((0, d))
        elif attack == "none":
            Wb = np.array([oracle(x, i) for i in range(n - nb, n)])
        else:
            hw = C.grad_est(x, sims[rng.choice(good)], rng, cfg["m"])
            Wb = C.bad_updates(attack, x, Wg, hw, nb, r, rng)
        w = aggregate(method, np.r_[Wg, Wb], bad_client, x, r, cfg, rng, log)
        if t % cfg["log_every"] == 0 and method != "ref":
            defs.append(G.safety_deficit(w, Wg, r))
        x = x + cfg["eta"] * w
        cur = frozenset(C.topr(x, r).tolist())
        stable = stable + 1 if cur == last else 0; last = cur
        if t % cfg["log_every"] == 0:
            curve.append((t, fg(sorted(cur))))
        if t + 1 >= min_rounds and stable >= cfg["patience"]:
            break
    out = dict(f_top=fg(C.topr(x, r)),
               f_pipage=np.mean([fg(C.pipage_round(x, r, rng)) for _ in range(cfg["rounds_samples"])]) if x.sum() > 0 else 0.0,
               f_rand=np.mean([fg(rng.choice(d, r, replace=False)) for _ in range(50)]),
               deficit=float(np.mean(defs)) if defs else 0.0, rounds=t + 1, curve=curve)
    if log["cost"]:
        cost = np.array(log["cost"], float).mean(0)
        out.update(client_nnz_reconstruct=cost[1], client_nnz_vote=cost[0], support=cost[2], n_aggregates=cost[3])
    if log["gm_used"]:
        out["gm_share"] = float(np.mean(log["gm_used"]))
    return out


_REF = {}


def normalized(cfg, seed, method, attack):
    """run() plus the normalized quality Q: 1 = averaging the good clients' updates only, 0 = random subset."""
    key = (seed,) + tuple(cfg[k] for k in ("dataset", "beta_c", "eta", "m", "alpha", "d", "n", "M", "L"))
    if key not in _REF:
        _REF[key] = run(cfg, seed, "ref", "none")
    ref, o = _REF[key], run(cfg, seed, method, attack)
    o["Q"] = (o["f_pipage"] - ref["f_rand"]) / max(ref["f_pipage"] - ref["f_rand"], 1e-9)
    o["ref_f_rand"], o["ref_f_pipage"] = ref["f_rand"], ref["f_pipage"]
    return o
