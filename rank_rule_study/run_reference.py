"""Reference runs with the full-rank optimizers of seamaze (CMA-ES, LM-MA-ES)
on Rosenbrock and Griewank, default population size, without gradient.

Same dimensions, start points, solver settings and global seed as the
rank-rule study (study_config.py, or STUDY_CONFIG), so the local minima found
by DLR-CMA-ES can be compared with those of the full-rank methods.
The optimizers are used unchanged from seamaze/optimizers.

Usage:
    python rank_rule_study/run_reference.py --workers 8 [--part K/N]
    python rank_rule_study/run_reference.py --summary-only
Results: rank_rule_study/results_reference/ (runs/, summary.csv, final_f.png).
The summary also includes the DLR-CMA-ES runs of the study (RESULTS_DIR) with
the same settings, if they exist.
"""
import argparse
import importlib
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[_v] = '1'
import glob
import pickle
import sys
import time
import traceback
from datetime import datetime
from multiprocessing import get_context
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
C = importlib.import_module(os.environ.get('STUDY_CONFIG', 'study_config'))    # settings file, see README

PROBLEMS = ['Rosenbrock', 'Griewank']
ALGOS = ['CMAES', 'LMMAES']
OUT = os.path.join(HERE, 'results_reference')
N_KEEP = 2000    # stored points per trajectory


def solver_kwargs(algo):
    kw = {k: v for k, v in C.SOLVER.items() if not k.startswith('low_rank_')}
    if algo == 'LMMAES':
        kw.pop('update_interval')    # no eigendecomposition in LM-MA-ES
    return kw


def run(job):
    from runner import BENCHMARKS
    import seamaze.optimizers as O
    t_start = time.time()
    try:
        problem = BENCHMARKS[job['problem']](job['ndim'])
        f_hist = []
        if C.GLOBAL_SEED is not None:
            np.random.seed(C.GLOBAL_SEED)
        s = getattr(O, job['algo'])(number_of_variables=job['ndim'], objective=problem.__call__,
                                     number_of_individuals=None, min_log_level='critical',
                                     callback=lambda s: f_hist.append(s._result['optimal_value']),
                                     **job['solver'])
        res = s.optimize(np.array(job['x0'], dtype=float))
        m = len(f_hist)
        keep = np.unique(np.concatenate([np.geomspace(1, max(m, 1), N_KEEP // 2).astype(int) - 1,
                                         np.linspace(0, max(m - 1, 0), N_KEEP // 2).astype(int)]))[:m]
        out = dict(job, gen=np.arange(1, m + 1)[keep], f=np.array(f_hist)[keep], iters=res['iterations'],
                   fbest=float(res['optimal_value']), info=res['solver_info'], wall=time.time() - t_start)
    except Exception as e:    # keep the sweep going; the failure is recorded
        out = dict(job, gen=np.array([]), f=np.array([]), iters=-1, fbest=float('inf'),
                   info=f'CRASH {type(e).__name__}: {e}', wall=time.time() - t_start,
                   traceback=traceback.format_exc())
    tmp = job['out'] + '.tmp'
    with open(tmp, 'wb') as fh:
        pickle.dump(out, fh)
    os.replace(tmp, job['out'])
    return job['out'], out['info'], out['fbest'], out['iters'], out['wall']


def load_all():
    """Reference runs and the matching DLR-CMA-ES runs of the study (default λ, no gradient)."""
    rows = []
    for fn in glob.glob(os.path.join(OUT, 'runs', '*.pkl')):
        rows.append(pickle.load(open(fn, 'rb')))
    for fn in glob.glob(os.path.join(HERE, C.RESULTS_DIR, 'runs', '*_popdef_nograd.pkl')):
        r = pickle.load(open(fn, 'rb'))
        if r['problem'] in PROBLEMS:
            rows.append(dict(r, algo=f"DLR {r['rule']}"))
    return rows


def summary():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = load_all()
    algos = ALGOS + sorted({r['algo'] for r in rows} - set(ALGOS))
    lines = ['problem,n,algorithm,runs,solved,local_minima,median_fbest,median_generations,stop_reasons']
    print(f"{'problem':11s} {'n':>4s} {'algorithm':22s} solved  median f   median gen  stop reasons")
    for p in PROBLEMS:
        for n in C.DIMS:
            for a in algos:
                rr = [r for r in rows if r['problem'] == p and r['ndim'] == n and r['algo'] == a]
                if not rr:
                    continue
                fb = np.array([r['fbest'] for r in rr])
                infos = {}
                for r in rr:
                    infos[r['info']] = infos.get(r['info'], 0) + 1
                ok = int(np.sum(fb < C.SUCCESS_F))
                gens = np.median([r['iters'] for r in rr])
                print(f"{p:11s} {n:4d} {a:22s} {ok}/{len(rr)}   {np.median(fb):9.2e}  {gens:9.0f}  {infos}")
                lines.append(f"{p},{n},{a},{len(rr)},{ok},{len(rr) - ok},{np.median(fb):.6e},{gens:.0f},"
                             + '"' + '; '.join(f'{k}: {v}' for k, v in infos.items()) + '"')
    with open(os.path.join(OUT, 'summary.csv'), 'w') as fh:
        fh.write('\n'.join(lines) + '\n')

    # final best f of every run; one axis per problem, x = n (shifted per algorithm)
    fig, axes = plt.subplots(1, len(PROBLEMS), figsize=(6 * len(PROBLEMS), 4.5), squeeze=False)
    colors = plt.cm.tab10(np.arange(len(algos)))
    for ax, p in zip(axes[0], PROBLEMS):
        for k, a in enumerate(algos):
            rr = [r for r in rows if r['problem'] == p and r['algo'] == a]
            if not rr:
                continue
            x = [C.DIMS.index(r['ndim']) + (k - (len(algos) - 1) / 2) * 0.12 for r in rr]
            y = [max(r['fbest'], 1e-16) for r in rr]
            ax.scatter(x, y, color=colors[k], s=22, label=a, alpha=0.8)
        ax.axhline(C.SUCCESS_F, color='0.6', lw=0.8, ls='--')
        ax.set_yscale('log')
        ax.set_xticks(range(len(C.DIMS)), [str(n) for n in C.DIMS])
        ax.set_xlabel('n')
        ax.set_ylabel('final best f')
        ax.set_title(f'{p} (default λ, no gradient)')
    axes[0][0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'final_f.png'), dpi=130)
    print(f"summary: {os.path.join(OUT, 'summary.csv')}, plot: {os.path.join(OUT, 'final_f.png')}")


def main():
    from run_study import start_points
    from runner import slug
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--part', default='1/1', help='K/N: run every N-th job, starting with the K-th')
    ap.add_argument('--summary-only', action='store_true')
    args = ap.parse_args()
    os.makedirs(os.path.join(OUT, 'runs'), exist_ok=True)
    if args.summary_only:
        summary()
        return
    part, nparts = map(int, args.part.split('/'))
    jobs = []
    for n in C.DIMS:    # small dimensions first
        for p in PROBLEMS:
            for i, x0 in enumerate(start_points(n)):
                for a in ALGOS:
                    out = os.path.join(OUT, 'runs', f'{a}_{slug(p)}_n{n}_x{i}.pkl')
                    if not os.path.exists(out) or pickle.load(open(out, 'rb'))['info'].startswith('CRASH'):
                        jobs.append(dict(algo=a, problem=p, ndim=n, start=i, x0=x0, solver=solver_kwargs(a), out=out))
    jobs = jobs[part - 1::nparts]
    logf = os.path.join(OUT, 'log.txt')
    with open(logf, 'a') as fh:
        fh.write(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] [part {part}/{nparts}] {len(jobs)} runs, {args.workers} workers\n")
    with get_context('spawn').Pool(args.workers, maxtasksperchild=50) as pool:
        for k, (out, info, fb, it, wall) in enumerate(pool.imap_unordered(run, jobs), 1):
            with open(logf, 'a') as fh:
                fh.write(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] [part {part}/{nparts}] {k}/{len(jobs)} "
                         f"{os.path.basename(out)}: f={fb:.2e}, {it} gen, {wall:.0f}s, {info}\n")
    summary()


if __name__ == '__main__':
    main()
