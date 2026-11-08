"""Run the experiments of Section VI. Results go to results/<experiment>.jsonl (resumable).

    python experiments.py                  # all experiments
    python experiments.py main hetero      # selected experiments
    python experiments.py --quick          # one seed per configuration, for a fast check
    python experiments.py --workers 8
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import argparse, json
from multiprocessing import Pool
from defsm import protocol as P

ATTACKS = ["include", "camouflage", "near_include", "mimic", "random"]
BETAS = [(bc, ba) for bc in (0.2, 0.3) for ba in (0.1, 0.3)]
DATASETS = ["cifar10", "pathmnist"]
METHODS = ["defsm", "agent_gm", "agent_trim", "agent_mean"]


def job(exp, dataset, method, attack, seed, **cfg):
    return dict(exp=exp, dataset=dataset, method=method, attack=attack, seed=seed, **cfg)


def experiments():
    E = {}
    E["main"] = [job("main", ds, m, a, s, beta_c=bc, beta_a=ba)
                 for s in range(3) for ds in DATASETS for a in ATTACKS for bc, ba in BETAS for m in METHODS]
    E["fsmgm"] = [job("fsmgm", ds, "fsm_gm", a, s, beta_c=bc, beta_a=0.0)
                  for s in range(3) for ds in DATASETS for a in ATTACKS for bc in (0.2, 0.3)]
    E["told"] = [job("told", ds, "agent_trim", a, s, beta_c=bc, beta_a=ba, told=tc)
                 for s in range(3) for ds in DATASETS for a in ATTACKS for bc, ba in BETAS for tc in (-0.1, 0.1)]
    E["hetero"] = [job("hetero", ds, m, a, s, beta_c=0.3, beta_a=0.3, alpha=0.01)
                   for s in range(10) for ds in DATASETS for a in ("include", "near_include", "mimic") for m in METHODS]
    E["scale"] = [job("scale", "pathmnist", m, a, s, beta_c=0.3, beta_a=0.3, n=n)
                  for s in range(10) for n in (60, 200, 500) for a in ("near_include", "mimic") for m in METHODS]
    E["agents"] = [job("agents", ds, m, a, s, beta_c=0.3, beta_a=0.3, M=M, L=L)
                   for s in range(5) for ds in DATASETS for a in ("near_include", "mimic")
                   for M, L in ((10, 20), (20, 20), (40, 20), (20, 10), (20, 40)) for m in ("defsm", "agent_gm")]
    E["ablation"] = [job("ablation", ds, "defsm_agentmean", a, s, beta_c=0.3, beta_a=0.3)
                     for s in range(3) for ds in DATASETS for a in ATTACKS]
    E["curves"] = [job("curves", ds, m, a, s, beta_c=0.3, beta_a=0.3)
                   for s in range(3) for ds in DATASETS for a in ATTACKS for m in METHODS]
    return E


def key(j):
    return json.dumps(j, sort_keys=True)


def task(j):
    cfg = dict(P.DEFAULT, **{k: v for k, v in j.items() if k in P.DEFAULT})
    if j["exp"] == "curves":
        o = P.run(cfg, j["seed"], j["method"], j["attack"])
        out = dict(deficit=o["deficit"], rounds=o["rounds"], curve=o["curve"])
    else:
        o = P.normalized(cfg, j["seed"], j["method"], j["attack"])
        out = {k: v for k, v in o.items() if k != "curve"}
    return j, {k: (float(v) if not isinstance(v, list) else v) for k, v in out.items()}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    E = experiments(); names = a.names or list(E)
    os.makedirs("results", exist_ok=True)
    todo = []
    for name in names:
        fn = f"results/{name}.jsonl"
        done = {key({k: v for k, v in json.loads(l).items() if k in E[name][0]}) for l in open(fn)} if os.path.exists(fn) else set()
        jobs = [j for j in E[name] if key(j) not in done and (not a.quick or j["seed"] == 0)]
        todo += jobs
        print(f"{name}: {len(jobs)} runs to do", flush=True)
    with Pool(a.workers) as pool:
        for i, (j, o) in enumerate(pool.imap_unordered(task, todo, chunksize=1), 1):
            with open(f"results/{j['exp']}.jsonl", "a") as f:
                f.write(json.dumps(dict(j, **o)) + "\n")
            print(f"[{i}/{len(todo)}] {j['exp']} {j['dataset']} {j['method']} {j['attack']} seed={j['seed']}"
                  + (f" Q={o['Q']:.3f}" if "Q" in o else "") + f" deficit={o['deficit']:.3f}", flush=True)
