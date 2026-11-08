"""Figures, tables and key numbers of Section VI from results/*.jsonl. Output: figures/.

    python figures.py
"""
import json, os, collections, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import spearmanr

plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#888", "axes.linewidth": 0.6, "xtick.color": "#444", "ytick.color": "#444",
                     "legend.frameon": False, "pdf.fonttype": 42})
OUT = "figures/"
os.makedirs(OUT, exist_ok=True)
METH = ["defsm", "agent_gm", "agent_trim", "agent_mean"]
NAME = {"defsm": "DeFSM", "agent_gm": "Agent-GM", "agent_trim": "Agent-Trim", "agent_mean": "Agent-Mean", "fsm_gm": "FSM-GM"}
TEX = dict(NAME, defsm=r"$\mathsf{DeFSM}$")
COL = {"defsm": "#2a78d6", "agent_gm": "#1baf7a", "agent_trim": "#e87ba4", "agent_mean": "#eb6834", "fsm_gm": "#555555"}
HATCH = {"defsm": "", "agent_gm": "\\\\", "agent_trim": "xx", "agent_mean": "//"}
ATT = ["include", "camouflage", "near_include", "mimic", "random"]
ALAB = {"include": "Include", "camouflage": "Camoufl.", "near_include": "Near-Incl.", "mimic": "Mimic", "random": "Random"}
DS = [("cifar10", "CIFAR10"), ("pathmnist", "PathMNIST")]
BET = [(bc, ba) for bc in (0.2, 0.3) for ba in (0.1, 0.3)]
LR = 20 * 10  # an agent downloads its L binary client updates with r ones each


def load(name):
    fn = f"results/{name}.jsonl"
    return [json.loads(l) for l in open(fn)] if os.path.exists(fn) else []


R = {name: load(name) for name in ("main", "fsmgm", "told", "hetero", "scale", "agents", "ablation", "curves")}


def group(rows, *fields):
    g = collections.defaultdict(list)
    for r in rows:
        g[tuple(r[f] for f in fields)].append(r)
    return g


main = group(R["main"], "dataset", "method", "attack", "beta_c", "beta_a")
setting_q = lambda ds, m: [np.mean([r["Q"] for r in main[(ds, m, a) + b]]) for a in ATT for b in BET]
runs_of = lambda ds, m: [r for a in ATT for b in BET for r in main[(ds, m, a) + b]]


def savefig(fig, name):
    fig.savefig(OUT + name + ".pdf", bbox_inches="tight"); fig.savefig(OUT + name + ".png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def cost(m, field):
    """Average download per round over the default matrix (baselines: seed 0, DeFSM: all seeds, as in the paper)."""
    rows = [r for r in R["main"] if r["method"] == m and (m == "defsm" or r["seed"] == 0)]
    return np.mean([r[field] for r in rows])


# ------------------------------------------------------------------ H1: accuracy summary (Fig. 2)
fsm = group(R["fsmgm"], "dataset", "attack", "beta_c")
fig, axs = plt.subplots(2, 1, figsize=(3.45, 2.75), sharex=True)
rng = np.random.default_rng(0)
order = METH + ["fsm_gm"]
for ax, (ds, dl) in zip(axs, DS):
    for i, m in enumerate(order):
        ys = [np.mean([r["Q"] for r in v]) for k, v in fsm.items() if k[0] == ds] if m == "fsm_gm" else setting_q(ds, m)
        ax.scatter(i + rng.uniform(-0.18, 0.18, len(ys)), ys, s=11, color=COL[m], alpha=0.45, linewidths=0)
        ax.scatter([i], [np.mean(ys)], s=60, marker="D", color=COL[m], edgecolors="k", linewidths=0.7, zorder=5)
        ax.scatter([i], [min(ys)], s=40, marker="v", facecolor="w", edgecolors=COL[m], linewidths=1.2, zorder=5)
    ax.axvline(len(METH) - 0.5, color="#bbb", lw=0.6, ls=":")
    ax.set_xticks(range(len(order))); ax.set_xticklabels([NAME[m].replace("-", "-\n") for m in order], fontsize=6.5)
    ax.text(0.01, 0.04, dl, transform=ax.transAxes, fontsize=7.5, va="bottom"); ax.grid(axis="y", alpha=0.25, lw=0.4)
    ax.set_ylim(0.42, 1.08); ax.set_ylabel("$Q$ per setting", fontsize=7.5); ax.tick_params(labelsize=7)
fig.legend(handles=[Line2D([], [], marker="D", ls="", mfc="w", mec="k", label="avg"),
                    Line2D([], [], marker="v", ls="", mfc="w", mec="k", label="worst setting")],
           loc="upper center", ncol=2, fontsize=7, bbox_to_anchor=(0.5, 1.04), handletextpad=0.3, columnspacing=1.0)
fig.subplots_adjust(hspace=0.08)
savefig(fig, "h1_accuracy")

# ------------------------------------------------------------------ Table III: reliability summary
def stats(rows_by_setting):
    allr = [r for v in rows_by_setting for r in v]
    return (max(r["deficit"] for r in allr), min(r["Q"] for r in allr),
            min(np.mean([r["Q"] for r in v]) for v in rows_by_setting), np.mean([r["Q"] for r in allr]))

S = {(ds, m): stats([main[(ds, m, a) + b] for a in ATT for b in BET]) for ds, _ in DS for m in METH}
best = {(ds, j): (min if j == 0 else max)(S[(ds, m)][j] for m in METH) for ds, _ in DS for j in range(4)}
fmt = lambda ds, m, j: (r"\textbf{%.2f}" if abs(S[(ds, m)][j] - best[(ds, j)]) < 1e-9 else "%.2f") % S[(ds, m)][j]
rows = []
for m in METH:
    rows.append(TEX[m] + " & " + " & ".join(fmt(ds, m, j) for ds, _ in DS for j in range(4)) + r"\\")
    if m == "defsm":
        rows.append(r"\midrule")
cells = []
for ds, _ in DS:
    cells += ["%.2f" % v for v in stats([v for k, v in fsm.items() if k[0] == ds])]
rows += [r"\midrule\multicolumn{9}{l}{\emph{Centralized ideal reference}}\\", "FSM-GM & " + " & ".join(cells) + r"\\",
         r"\midrule\multicolumn{9}{l}{\emph{Agent-Trim told a wrong Byzantine rate (true $\pm0.1$)}}\\"]
told = group(R["told"], "dataset", "told", "attack", "beta_c", "beta_a")
for tc in (-0.1, 0.1):
    cells = []
    for ds, _ in DS:
        cells += ["%.2f" % v for v in stats([told[(ds, tc, a) + b] for a in ATT for b in BET])]
    rows.append(f"Agent-Trim, ${'-' if tc < 0 else '+'}0.1$ & " + " & ".join(cells) + r"\\")
head = r"\makecell{worst\\$\sigma$} & \makecell{worst\\run} & \makecell{worst\\setting} & \makecell{avg\\$Q$}"
open(OUT + "tab_summary.tex", "w").write(r"""\begin{table}[t]
\centering\small
\caption{Reliability over all 20 attack settings (5 attacks $\times$ $\beta_c\in\{0.2,0.3\}$ $\times$ $\beta_a\in\{0.1,0.3\}$), 3 seeds, $\alpha=0.5$. All methods except FSM-GM run in the $\mathsf{DeFSM}$ architecture; Agent-Trim is told the true Byzantine rates unless stated. FSM-GM is the centralized ideal reference ($\beta_c$ only, 10 settings). Worst $\sigma$: largest deficit of any run. Worst run / worst setting / avg: smallest $Q$ of any run, smallest average $Q$ of any setting, average $Q$. Bold: best in column.}
\label{tab:summary}
\setlength{\tabcolsep}{2.2pt}
\begin{tabular}{lcccccccc}
\toprule
& \multicolumn{4}{c}{CIFAR10} & \multicolumn{4}{c}{PathMNIST}\\
\cmidrule(lr){2-5}\cmidrule(lr){6-9}
Method & """ + head + " & " + head + r"""\\
\midrule
""" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")

# ------------------------------------------------------------------ hardest setting (Fig. 3)
fig, axes = plt.subplots(1, 2, figsize=(5.6, 1.75), sharey=True)
x, w = np.arange(len(ATT)), 0.2
for ax, (ds, dl) in zip(axes, DS):
    for i, m in enumerate(METH):
        v = [np.array([r["Q"] for r in main[(ds, m, a, 0.3, 0.3)]]) for a in ATT]
        ax.bar(x + (i - 1.5) * w, [q.mean() for q in v], w * 0.88, yerr=[q.std() for q in v], color=COL[m], hatch=HATCH[m],
               edgecolor="white", linewidth=0.4, error_kw=dict(lw=0.6, capsize=1.5, ecolor="#555"), label=TEX[m])
    ax.set_xticks(x); ax.set_xticklabels([ALAB[a] for a in ATT], fontsize=6.5, rotation=20); ax.set_title(dl, fontsize=9); ax.set_ylim(0, 1.15)
    ax.yaxis.grid(True, color="#e5e5e5", lw=0.5); ax.set_axisbelow(True)
axes[0].set_ylabel("Normalized quality $Q$")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, ncol=4, loc="upper center", fontsize=7.5, handlelength=1.6, columnspacing=1.4)
fig.subplots_adjust(top=0.78, bottom=0.12, left=0.07, right=0.995, wspace=0.06)
savefig(fig, "quality_hard")

# ------------------------------------------------------------------ convergence (Fig. 4)
cv = group(R["curves"], "dataset", "method", "attack")
fig, axes = plt.subplots(2, 5, figsize=(6.4, 2.6))
for row, (ds, dl) in enumerate(DS):
    for col, a in enumerate(ATT):
        ax = axes[row, col]
        for i, m in enumerate(METH):
            cs = [r["curve"] for r in cv[(ds, m, a)]]; n = min(len(c) for c in cs)
            ax.plot([p[0] for p in cs[0][:n]], np.mean([[p[1] for p in c[:n]] for c in cs], 0), color=COL[m], lw=1.3,
                    ls=["-", "--", "-.", ":"][i], label=TEX[m])
        if row == 0: ax.set_title(ALAB[a], fontsize=8)
        if row == 1: ax.set_xlabel("Round", fontsize=7.5)
        if col == 0: ax.set_ylabel(dl + "\nobjective", fontsize=7.5)
        ax.tick_params(labelsize=6.5); ax.set_xlim(0, 100); ax.yaxis.grid(True, color="#e5e5e5", lw=0.5)
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, ncol=4, loc="upper center", fontsize=7.5, handlelength=2.2, columnspacing=1.4, bbox_to_anchor=(0.5, 1.07))
fig.subplots_adjust(top=0.88, bottom=0.12, left=0.08, right=0.995, wspace=0.32, hspace=0.25)
savefig(fig, "convergence")

# ------------------------------------------------------------------ H2: cost plane (Fig. 5) (the cost table is not in the paper)
fig, ax = plt.subplots(figsize=(3.45, 2.5))
off = {("defsm", ""): (8, -2), ("agent_gm", ", clients"): (8, 4), ("agent_mean", ", clients"): (8, -9),
       ("agent_trim", ", clients"): (8, -2), ("agent_gm", ", vote"): (-2, 9), ("agent_mean", ", vote"): (-30, -12),
       ("agent_trim", ", vote"): (6, 6)}
for m in METH:
    cb, ca = cost(m, "client_nnz_reconstruct"), cost(m, "client_nnz_vote")
    pts = [(LR, cb, "")] if m == "defsm" else [(LR, cb, ", clients"), (LR + cb, ca, ", vote")]
    for px, py, suf in pts:
        ax.scatter([px], [py], s=70 if m == "defsm" else 45, marker="*" if m == "defsm" else ("o" if suf == ", clients" else "s"),
                   color=COL[m], edgecolors="k", linewidths=0.6, zorder=5)
        ax.annotate(NAME[m] + suf + (" (no vote)" if m == "defsm" else ""), (px, py), xytext=off[(m, suf)],
                    textcoords="offset points", fontsize=6.5, color=COL[m], va="center")
ax.set_xlabel("download per agent (nonzero entries/round)"); ax.set_ylabel("download per client")
ax.set_xlim(0, 1500); ax.set_ylim(-60, 1200); ax.grid(alpha=0.25, lw=0.4)
savefig(fig, "h2_cost")

rule = {"agent_gm": "GM", "agent_trim": "Trim", "agent_mean": "Mean"}
rows = [r"$\mathsf{DeFSM}$ & %d & $\mathsf{SafeGM}$ on $L$ & --- & %d (%.1f vectors)\\" % (LR, cost("defsm", "client_nnz_reconstruct"), cost("defsm", "support")),
        r"$\mathsf{DeFSM}$, vote & %d & $\mathsf{SafeGM}$, sum & yes & %d\\" % (LR + cost("defsm", "client_nnz_reconstruct"), cost("defsm", "client_nnz_vote")),
        r"\midrule"]
for m in METH[1:]:
    rows.append(r"%s, vote & %d & %s twice & yes & %d\\" % (NAME[m], LR + cost(m, "client_nnz_reconstruct"), rule[m], cost(m, "client_nnz_vote")))
    rows.append(r"%s, clients & %d & %s on $L$ & --- & %d (%.1f vectors)\\" % (NAME[m], LR, rule[m], cost(m, "client_nnz_reconstruct"), cost(m, "n_aggregates")))
open(OUT + "tab_comm.tex", "w").write(r"""\begin{table}[t]
\centering\footnotesize
\caption{Cost per aggregation round in the $\mathsf{DeFSM}$ architecture ($M=20$, $L=20$, $r=10$, $d=200$), in nonzero entries downloaded, averaged over both datasets and all settings. In $\mathsf{DeFSM}$, the contract computes the 2D point and the weights from $2M$ numbers on-chain, and each client computes the weighted sum. Vote: the agents compute the global aggregate and vote on its hash (for $\mathsf{DeFSM}$, an option for bandwidth-limited clients). Clients: each client computes it from all local aggregates. The other rules cannot run in the contract.}
\label{tab:comm}
\setlength{\tabcolsep}{2pt}
\begin{tabular}{lcccc}
\toprule
Rule & \makecell{Agent\\download} & \makecell{Agent\\work} & Vote & \makecell{Client\\download}\\
\midrule
""" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")

# ------------------------------------------------------------------ H3: stability (Fig. 6), deficit vs quality (Fig. 7), Table IV
het = group(R["hetero"], "dataset", "method", "attack")
fig, axs = plt.subplots(1, 2, figsize=(3.45, 2.0), sharey=True)
for ax, (ds, dl) in zip(axs, DS):
    for m in METH:
        for a in ATT:
            qs = [r["Q"] for b in BET for r in main[(ds, m, a) + b]]
            ax.scatter(np.mean(qs), min(qs), s=26, color=COL[m], alpha=0.8, linewidths=0)
        for a in ("include", "near_include", "mimic"):
            qs = [r["Q"] for r in het[(ds, m, a)]]
            ax.scatter(np.mean(qs), min(qs), s=30, marker="^", facecolor="w", edgecolors=COL[m], linewidths=1.1)
    ax.plot([-0.2, 1.1], [-0.2, 1.1], color="#bbb", lw=0.6, ls="--")
    ax.text(0.97, 0.04, dl, transform=ax.transAxes, fontsize=7.5, ha="right", va="bottom"); ax.set_xlabel("avg $Q$"); ax.grid(alpha=0.25, lw=0.4)
    ax.set_xlim(0.3, 1.05); ax.set_ylim(-0.1, 1.05); ax.set_xticks([0.4, 0.6, 0.8, 1.0]); ax.tick_params(labelsize=7)
axs[0].set_ylabel("worst run $Q$")
h = [Line2D([], [], marker="s", ls="", color=COL[m], label=NAME[m]) for m in METH]
h += [Line2D([], [], marker="o", ls="", color="#999", label=r"$\alpha=0.5$, all $\beta$"),
      Line2D([], [], marker="^", ls="", mfc="w", mec="#666", label=r"$\alpha=0.01$")]
fig.legend(handles=h, loc="upper center", ncol=3, fontsize=6.5, bbox_to_anchor=(0.5, 1.08), handletextpad=0.3, columnspacing=0.8)
fig.subplots_adjust(wspace=0.08)
savefig(fig, "h3_stability")

fig, ax = plt.subplots(figsize=(3.45, 2.5))
pd_, pq_ = [], []
for m in METH:
    for ds, mk in (("cifar10", "o"), ("pathmnist", "^")):
        v = runs_of(ds, m)
        ax.scatter([r["deficit"] for r in v], [r["Q"] for r in v], s=9, marker=mk, color=COL[m], alpha=0.5, linewidths=0)
        if m != "agent_trim":
            pd_ += [r["deficit"] for r in v]; pq_ += [r["Q"] for r in v]
pd_, pq_ = np.array(pd_), np.array(pq_)
bins = [(lo, (pd_ >= lo) & (pd_ < lo + 0.1)) for lo in np.arange(0, 0.5, 0.1)]
ax.plot([lo + 0.05 for lo, s in bins], [pq_[s].mean() for _, s in bins], color="k", lw=1.2, marker="D", ms=3.5)
ax.plot([lo + 0.05 for lo, s in bins], [pq_[s].min() for _, s in bins], color="k", lw=1.0, ls=":", marker="v", ms=3.5, mfc="w")
ax.annotate("Agent-Trim:\nnear-zero\naggregates", (0.97, 0.45), fontsize=6.5, color="#b0507a", ha="center")
ax.set_xlabel(r"safety deficit $\sigma$ (avg over rounds)"); ax.set_ylabel("quality $Q$")
ax.set_xlim(0, 1.05); ax.set_ylim(0.2, 1.13); ax.grid(alpha=0.25, lw=0.4)
h = [Line2D([], [], marker="o", ls="", color=COL[m], label=NAME[m]) for m in METH]
h += [Line2D([], [], color="k", marker="D", ms=3.5, label="avg in bin"),
      Line2D([], [], color="k", ls=":", marker="v", ms=3.5, mfc="w", label="worst in bin")]
ax.legend(handles=h, fontsize=6, loc="lower left", bbox_to_anchor=(0.5, 0.0), ncol=1, handletextpad=0.3)
savefig(fig, "h3_deficit_quality")

rows = []
for ds, dl in DS:
    for j, a in enumerate(("include", "near_include", "mimic")):
        q = {m: [r["Q"] for r in het[(ds, m, a)]] for m in METH}
        bm = max(min(v) for v in q.values())
        cells = [("%.2f / " % np.mean(q[m])) + ((r"\textbf{%.2f}" if abs(min(q[m]) - bm) < 1e-9 else "%.2f") % min(q[m])) for m in METH]
        rows.append((dl if j == 0 else "") + " & " + {"include": "Include", "near_include": "Near-I.", "mimic": "Mimic"}[a]
                    + " & " + " & ".join(cells) + r"\\")
    rows.append(r"\midrule")
open(OUT + "tab_hetero.tex", "w").write(r"""\begin{table}[t]
\centering\footnotesize
\setlength{\tabcolsep}{2.5pt}
\caption{Extreme heterogeneity (Dirichlet $\alpha=0.01$, $\beta_c=\beta_a=0.3$), 10 seeds: avg / worst normalized quality $Q$. Bold: best worst case.}
\label{tab:hetero}
\begin{tabular}{llcccc}
\toprule
Dataset & Attack & $\mathsf{DeFSM}$ & Agent-GM & Agent-Trim & Agent-Mean\\
\midrule
""" + "\n".join(rows[:-1]) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")

# ------------------------------------------------------------------ H4: design parameters (Fig. 8)
sc = group(R["scale"], "n", "method")
ag = group(R["agents"], "M", "L", "method")
fig, axes = plt.subplots(1, 4, figsize=(7.16, 1.9))
def panel(ax, xs, get, xlabel, title):
    for m, mk in (("defsm", "o"), ("agent_gm", "s")):
        st = [(np.mean(get(x, m)), np.min(get(x, m))) for x in xs]
        ax.plot(range(len(xs)), [s[0] for s in st], color=COL[m], lw=2, marker=mk, ms=4, label=NAME[m] + ", avg")
        ax.plot(range(len(xs)), [s[1] for s in st], color=COL[m], lw=1.2, ls=":", marker=mk, ms=3.5, mfc="white", label=NAME[m] + ", worst run")
    ax.set_xticks(range(len(xs))); ax.set_xticklabels([str(x) for x in xs]); ax.set_xlim(-0.3, len(xs) - 0.7)
    ax.set_xlabel(xlabel); ax.set_title(title, fontsize=8.5); ax.set_ylim(0.35, 0.95)
    ax.yaxis.grid(True, color="#e5e5e5", lw=0.5); ax.set_axisbelow(True)
panel(axes[0], [60, 200, 500], lambda n, m: [r["Q"] for r in sc[(n, m)]], "Number of clients $N$", "(a) Clients (PathMNIST)")
panel(axes[1], [10, 20, 40], lambda M, m: [r["Q"] for r in ag[(M, 20, m)]], "Number of agents $M$", "(b) Agents")
panel(axes[2], [10, 20, 40], lambda L, m: [r["Q"] for r in ag[(20, L, m)]], "Sample size $L$", "(c) Sample size")
axes[0].set_ylabel("Normalized quality $Q$")
Ms = [10, 20, 40]
sup = [np.mean([r["support"] for r in ag[(M, 20, "defsm")]]) for M in Ms]
alla = [np.mean([r["n_aggregates"] for r in ag[(M, 20, "defsm")]]) for M in Ms]
ax = axes[3]
ax.plot(Ms, sup, color=COL["defsm"], lw=2, marker="o", ms=4); ax.plot(Ms, alla, color=COL["agent_gm"], lw=2, marker="s", ms=4)
ax.annotate(f"{sup[-1]:.1f}", (Ms[-1], sup[-1]), textcoords="offset points", xytext=(5, -2), fontsize=6.5)
ax.annotate(f"{alla[-1]:.0f}", (Ms[-1], alla[-1]), textcoords="offset points", xytext=(5, -2), fontsize=6.5)
ax.text(11, 27, "Agent-GM\n(clients)", fontsize=6.5, ha="left"); ax.text(26, 6, "DeFSM", fontsize=6.5)
ax.set_xticks(Ms); ax.set_xlim(7, 47); ax.set_xlabel("Number of agents $M$"); ax.set_title("(d) Client download", fontsize=8.5)
ax.set_ylabel("Aggregates per round"); ax.yaxis.grid(True, color="#e5e5e5", lw=0.5)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, ncol=4, loc="upper center", fontsize=7, handlelength=2.4, columnspacing=1.2)
fig.subplots_adjust(top=0.74, bottom=0.22, left=0.06, right=0.995, wspace=0.42)
savefig(fig, "ablation")

# ------------------------------------------------------------------ overall picture (Fig. 9)
fsm_avg = np.mean([r["Q"] for r in R["fsmgm"]])
tw = group(R["told"], "dataset", "told", "attack", "beta_c", "beta_a")
trim_wrong = min(np.mean([r["Q"] for r in v]) for k, v in tw.items() if k[1] == 0.1)
fig, axs = plt.subplots(1, 2, figsize=(6.0, 2.15), sharey=True)
titles = ["(a) Download per client (no vote for any method)", "(b) Download per agent (baselines need a vote)"]
for p, ax in enumerate(axs):
    means = {}
    for m in METH:
        xs_all, ys_all = [], []
        for ds, mk in (("cifar10", "o"), ("pathmnist", "^")):
            keys = [k for k in main if k[0] == ds and k[1] == m]
            cb = [np.mean([r["client_nnz_reconstruct"] for r in main[k] if m == "defsm" or r["seed"] == 0]) for k in keys]
            xs = cb if p == 0 else [LR if m == "defsm" else LR + c for c in cb]
            ys = [np.mean([r["Q"] for r in main[k]]) for k in keys]
            ax.scatter(xs, ys, s=12, marker=mk, color=COL[m], alpha=0.35, linewidths=0)
            xs_all += xs; ys_all += ys
        mx, my, wy = np.mean(xs_all), np.mean(ys_all), min(ys_all)
        means[m] = (mx, wy)
        ax.plot([mx, mx], [wy, my], color=COL[m], lw=1.2, zorder=4)
        ax.scatter([mx], [my], s=70, marker="D", color=COL[m], edgecolors="k", linewidths=0.7, zorder=5)
        ax.scatter([mx], [wy], s=45, marker="v", facecolor="w", edgecolors=COL[m], linewidths=1.2, zorder=5)
    tx = means["agent_trim"][0]
    ax.plot([tx, tx], [trim_wrong, means["agent_trim"][1]], color=COL["agent_trim"], lw=1.0, ls=":", zorder=4)
    ax.scatter([tx], [trim_wrong], s=45, marker="v", facecolor="w", edgecolors=COL["agent_trim"], linewidths=1.2, zorder=5)
    ax.annotate("Agent-Trim, rate misestimated", (tx, trim_wrong), xytext=(8, 0), textcoords="offset points", fontsize=6.5, color="#b0507a", va="center")
    ax.axhline(fsm_avg, color="#555", ls="--", lw=0.8)
    ax.text(0.99, fsm_avg + 0.012, "centralized ideal", fontsize=7, color="#444", ha="right", transform=ax.get_yaxis_transform())
    ax.set_title(titles[p], fontsize=8); ax.set_xlabel("nonzero entries downloaded per round"); ax.grid(alpha=0.25, lw=0.4)
    ax.set_xlim(left=0)
axs[0].set_ylabel("quality $Q$ (one point per setting)"); axs[0].set_ylim(0.3, 1.1)
h = [Line2D([], [], marker="s", ls="", color=COL[m], label=NAME[m]) for m in METH]
h += [Line2D([], [], marker="o", ls="", color="#999", label="CIFAR10"), Line2D([], [], marker="^", ls="", color="#999", label="PathMNIST"),
      Line2D([], [], marker="D", ls="", mfc="w", mec="k", label="avg"), Line2D([], [], marker="v", ls="", mfc="w", mec="k", label="worst setting")]
fig.legend(handles=h, loc="upper center", ncol=4, fontsize=7, bbox_to_anchor=(0.5, 1.02), handletextpad=0.2, columnspacing=0.9)
fig.subplots_adjust(top=0.74, bottom=0.17, left=0.07, right=0.99, wspace=0.08)
savefig(fig, "tradeoff")

# ------------------------------------------------------------------ key numbers quoted in the text
ab = group(R["ablation"], "dataset"); hard = lambda ds, m: np.mean([r["Q"] for a in ATT for r in main[(ds, m, a, 0.3, 0.3)]])
L = []
for ds, dl in DS:
    L.append(f"{dl}: avg Q  DeFSM {S[(ds, 'defsm')][3]:.2f}  Agent-GM {S[(ds, 'agent_gm')][3]:.2f}  "
             f"FSM-GM {np.mean([r['Q'] for r in R['fsmgm'] if r['dataset'] == ds]):.2f}")
d_avg, g_avg = np.mean([S[(ds, 'defsm')][3] for ds, _ in DS]), np.mean([S[(ds, 'agent_gm')][3] for ds, _ in DS])
L.append(f"DeFSM retains {100 * d_avg / fsm_avg:.1f}% of FSM-GM; Agent-GM retains {100 * g_avg / fsm_avg:.1f}%")
L.append(f"client download: DeFSM {cost('defsm', 'client_nnz_reconstruct'):.0f} entries ({cost('defsm', 'support'):.1f} aggregates), "
         f"Agent-GM {cost('agent_gm', 'client_nnz_reconstruct'):.0f}; agent download with vote: Agent-GM {LR + cost('agent_gm', 'client_nnz_reconstruct'):.0f} vs DeFSM {LR}")
L.append(f"deficit vs quality (DeFSM, Agent-GM, Agent-Mean): Spearman {spearmanr(pd_, pq_)[0]:.2f}; "
         + ", ".join(f"{NAME[m]} {spearmanr([r['deficit'] for ds, _ in DS for r in runs_of(ds, m)], [r['Q'] for ds, _ in DS for r in runs_of(ds, m)])[0]:.2f}" for m in METH))
for ds, dl in DS:
    L.append(f"{dl} at beta_c=beta_a=0.3: DeFSM {hard(ds, 'defsm'):.2f}, agent-level mean {np.mean([r['Q'] for r in ab[(ds,)]]):.2f}")
L.append(f"client aggregates vs M=10,20,40: DeFSM {', '.join(f'{s:.1f}' for s in sup)}; all {', '.join(f'{s:.0f}' for s in alla)}")
open(OUT + "numbers.txt", "w").write("\n".join(L) + "\n")
print("\n".join(L))
