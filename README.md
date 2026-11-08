# DeFSM simulation code

Simulator for the paper *DeFSM: Robust and Efficient Decentralization of Federated Submodular Maximization*. It reproduces every figure and table of Section VI (Numerical Evaluation).

## Layout

| File | Contents |
|---|---|
| `defsm/core.py` | FSM problem: features, facility-location objectives, client oracle, attacks, pipage rounding |
| `defsm/geometry.py` | Tukey depth, deepest point, geometric median, the SafeGM decision, capped and basic lifts, safety deficit |
| `defsm/protocol.py` | One run: clients, agents and contract (Sections IV-C, IV-D), the baselines, the normalized quality Q |
| `experiments.py` | All experiments of Section VI; writes `results/<experiment>.jsonl` |
| `figures.py` | All figures and tables of Section VI, plus `figures/numbers.txt` with the numbers quoted in the text |
| `make_features.py` | Builds the ResNet18 feature files from the raw datasets |
| `results/` | The results used in the paper |

Methods: `defsm` (DeFSM with SafeGM), `agent_gm`, `agent_trim`, `agent_mean` (the rule run by the agents in place of SafeGM), `fsm_gm` (FSM-GM, the centralized ideal reference), `defsm_agentmean` (ablation: mean at the agent level), `ref` (averaging the good clients only; defines Q = 1).

## Setup

```
pip install -r requirements.txt
```

Put the four feature files in `data/`:

| File | Source | md5 |
|---|---|---|
| `cifar10_features.npy`, `cifar10_labels.npy` | `cifar10_1.zip` (ResNet18 features) | `445233db9d767bef013f08b9a20643e7`, `eb77e55fd91712f9543e280e89841350` |
| `pathmnist_features.npy`, `pathmnist_labels.npy` | `data.zip` | `29e0595bfeeaecf826cb444491f6b1fa`, `d52b7cfc25f972b2fcdcaa541995fd82` |

Do not use the `cifar10_*` files in `data.zip`: they hold older colour-histogram features. To rebuild the features from the raw datasets instead, run `python ../make_features.py` inside `data/` (needs torch, torchvision and medmnist); results then differ slightly from the paper.

## Reproduce

Figures and tables from the shipped results (seconds):

```
python figures.py
```

Re-run the simulations (results are appended to `results/`; delete or move that folder first to start from scratch):

```
python experiments.py --quick           # one seed per configuration, a fast check
python experiments.py main hetero        # selected experiments
python experiments.py --workers 8        # everything: about 1,600 runs
```

A single run takes about 5 to 15 seconds on one core (longer for N = 500). With the paper's feature files, every run is deterministic and reproduces the shipped results exactly.

| Experiment | Paper | Runs |
|---|---|---|
| `main` | Default matrix: Figs. 2, 3, 5, 6, 7, 9, Table III | 4 methods x 2 datasets x 5 attacks x 4 Byzantine-rate pairs x 3 seeds |
| `fsmgm` | FSM-GM reference | 2 datasets x 5 attacks x 2 client rates x 3 seeds |
| `told` | Agent-Trim told a wrong rate (Table III, Fig. 9) | default matrix, rate off by -0.1 and +0.1 |
| `hetero` | Extreme heterogeneity, alpha = 0.01 (Table IV, Fig. 6) | 4 methods x 2 datasets x 3 attacks x 10 seeds |
| `scale` | Number of clients N (Fig. 8a) | PathMNIST, N in {60, 200, 500}, 10 seeds |
| `agents` | Number of agents M and sample size L (Fig. 8b-d) | DeFSM and Agent-GM, 5 seeds |
| `ablation` | Mean instead of the SafeGM point at the agent level | 2 datasets x 5 attacks x 3 seeds |
| `curves` | Convergence (Fig. 4) | 4 methods x 2 datasets x 5 attacks x 3 seeds |

Communication costs in Figs. 5 and 9 average the baselines over seed 0 and DeFSM over all seeds, as in the paper. `figures.py` also writes `tab_comm.tex`, a per-method cost table that is no longer in the paper.
