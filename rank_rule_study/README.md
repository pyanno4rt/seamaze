# Rank-rule study (DLR-CMA-ES)

Compares rank proposal rules of DLR-CMA-ES with the problems and solver
settings of the seamaze cross-benchmarking script, over n = 10 … 500.

- Phase 1: **matrix Bernstein with ε**
- Phase 2: **current (0.02 soft filter)** and **soft filter with ε**
- Phase 3: **matrix Bernstein with age-dependent ε**

Only seamaze code is used: the benchmarks come from `seamaze.benchmarks`, and
the optimizer is `seamaze/optimizers/_dlrcmaes.py` of this copy of the
repository. `current` is that file unchanged; the other rules are copies of it
in which only the rank proposal is replaced (`variants.py`; the age-dependent
rule additionally stores its noise state per direction). The copies are
written to `results_benchmark/optimizer_variants/`.

## Setup on the workstation

Copy the whole `seamaze` folder, then in it (Python ≥ 3.11):

```bash
python3 -m venv .venv-study
source .venv-study/bin/activate
pip install numpy scipy numba matplotlib seaborn
```

## Run

```bash
python rank_rule_study/run_study.py --quick --workers 4    # test, a few minutes
bash rank_rule_study/start.sh --workers 32                 # full study in the background
tail -f rank_rule_study/results_decay_fixed/log.txt        # progress (folder: RESULTS_DIR)
```

To use more cores with a limited number of workers per process, split the
study into parts that run in parallel and do not overlap (each takes every
N-th run):

```bash
bash rank_rule_study/start_all.sh 4 8      # 4 processes with 8 workers each
bash rank_rule_study/stop.sh               # stops all of them
```

Use the same Python environment for every start. `start.sh` uses `python3`
(set `PYTHON=...` for another interpreter) and keeps running after logout.
Each run uses one core; `--workers` sets how many run in parallel.

Finished runs are kept in `results_benchmark/runs/` and skipped on restart;
crashed runs are redone. Options:

- `--plots-only`: redraw all plots from the saved runs
- `--phase 3_age_noise`: run only this phase (also next to a running study)
- `--problems Sphere "Rotated Ellipsoid"`: run only these problems

Results are written to the folder `RESULTS_DIR` in `study_config.py`, currently
`results_decay_fixed/` (runs with the covariance decay 1 − c₁ − c_μ Σw of
Hansen 2016, eq. 47). `results_benchmark/` holds the earlier runs with the
decay 1 − c₁ − c_μ, kept for comparison.
Every plot update first empties its plot folder, so only current figures are
in it.

### Benchmark-script settings

`study_config_benchmark.py` uses the dimensions and population sizes of the
benchmark script (n = 10 and 100 with the provided start points, default λ
and λ = 100), without gradient; all other settings as in `study_config.py`.
Results go to `results_benchmark_settings/`. It runs independently of the main
study (also at the same time):

```bash
bash rank_rule_study/start_benchmark.sh 4 8
tail -f rank_rule_study/results_benchmark_settings/log.txt
STUDY_CONFIG=study_config_benchmark python rank_rule_study/run_study.py --plots-only
```

`stop.sh` stops all running studies.

## Settings (`study_config.py`)

As in the benchmark script:

- Problems (seamaze names): Ackley, Bent-Cigar, Griewank, Rosenbrock,
  Rotated Ellipsoid, Sphere. Any other seamaze benchmark can be added by its
  name.
- Population size and gradient: the script uses default and 100, with and
  without gradient; currently only default λ without gradient
  (`POP_SIZES`, `USE_GRADIENT`).
- Start points: `starting_points_dim{n}.npz` (5 points each). For dimensions
  without such a file, 5 points are drawn once uniformly from [-5, 5]^n and
  saved as `starting_points_dim{n}_generated.npz`; a provided file always
  takes priority.
- Solver: σ0 = 3, energy tolerance 1e-2, at most 10^7 generations and 12 h,
  no fitness target, fitness window 50, tolerance 1e-6, σ threshold 1e-8,
  update interval 1, `random_state=43`.

Differences to the script: dimensions n = 10, 20, 50, 100, 200, 500, and the
global numpy generator is seeded with 43 (the script leaves it unseeded), so
runs are reproducible and the rules see the same random numbers.

A run counts as solved in plots and tables if its final best f is below
1e-8 (`SUCCESS_F`).

Runs per rule: 6 problems × 6 dimensions × 5 start points × settings
(currently 1 setting: 180 runs per rule, 720 in total).
n ≤ 100 finishes within hours; n = 200 and 500 dominate, especially with
λ = 100 and in phase 2 (rank up to several hundred at n = 500).

## Output (`results_benchmark/plots/<phase>/`)

| File | Content |
|---|---|
| `success_table.png` | runs with final f < 1e-8, median generations, mean rank; per setting and rule |
| `scaling_generations.png`, `scaling_rank.png`, `scaling_runtime.png` | generations, mean rank, runtime vs. n; one axis per problem |
| `final_f_vs_runtime_pop<λ>.png`, `final_f_vs_generations_pop<λ>.png` | final best f of every run; one axis per problem, colour = rule, marker = n, filled/hollow = without/with gradient |
| `traj_rank_pop<λ>.png`, `traj_f_pop<λ>.png` | rank / best f over generations, one panel per problem and n |
| `summary_runs.csv` | one row per run |
| `trajectories.csv` | thinned trajectories incl. low-rank contribution |

Plots of a phase include all rules of that and earlier phases.
