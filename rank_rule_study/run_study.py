"""Rank-rule study for DLR-CMA-ES with the settings of the seamaze benchmark
script: matrix Bernstein first, then the comparison rules; plots after each
phase and regularly in between.

Usage (from anywhere):
    python rank_rule_study/run_study.py                # all phases, default workers
    python rank_rule_study/run_study.py --workers 16
    python rank_rule_study/run_study.py --quick        # short test run (a few minutes)
    python rank_rule_study/run_study.py --plots-only   # only redo the plots
    python rank_rule_study/run_study.py --phase 3_age_noise          # only this phase
    python rank_rule_study/run_study.py --problems Sphere "Rotated Ellipsoid"   # only these problems
    python rank_rule_study/run_study.py --part 2/4 --workers 8   # every 4th run (see start_all.sh)
Settings: study_config.py. Finished runs are kept; restarting continues and
reruns only crashed runs.
"""
import argparse
import fcntl
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[_v] = '1'
import pickle
import shutil
import sys
import time
from datetime import datetime
from multiprocessing import get_context
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import importlib    # noqa: E402
C = importlib.import_module(os.environ.get('STUDY_CONFIG', 'study_config'))    # settings file, see README


def log(msg, path):
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    with open(path, 'a') as fh:
        fh.write(line + '\n')


def start_points(n):
    """Provided start points, else generated ones (uniform in [-5, 5]^n, saved once)."""
    provided = os.path.join(HERE, f'starting_points_dim{n}.npz')
    if os.path.exists(provided):
        return np.load(provided)['starting_points']
    generated = os.path.join(HERE, f'starting_points_dim{n}_generated.npz')
    if not os.path.exists(generated):
        pts = np.random.default_rng(C.START_SEED + n).uniform(-5.0, 5.0, (C.N_GENERATED_STARTS, n))
        tmp = f'{generated}.{os.getpid()}.npz'
        np.savez(tmp, starting_points=pts)
        os.replace(tmp, generated)
    return np.load(generated)['starting_points']


def jobs_for(rules, modules, run_dir, quick, problems):
    from runner import slug
    dims = [min(C.DIMS)] if quick else C.DIMS
    solver = dict(C.SOLVER)
    if quick:
        solver['maximum_iterations'] = 2000
    jobs = []
    for n in dims:    # small dimensions first, so results arrive early
        x0s = start_points(n)[:2] if quick else start_points(n)
        for p in problems:
            for i, x0 in enumerate(x0s):
                for pop in C.POP_SIZES:
                    for grad in C.USE_GRADIENT:
                        for v in rules:
                            name = (f"{v}_{slug(p)}_n{n}_x{i + 1}_pop{pop or 'def'}_"
                                    f"{'grad' if grad else 'nograd'}.pkl")
                            jobs.append(dict(rule=v, module=modules[v], problem=p, ndim=n, start=i + 1,
                                             x0=x0.tolist(), pop=pop, grad=grad, solver=solver,
                                             global_seed=C.GLOBAL_SEED, out=os.path.join(run_dir, name)))
    return jobs


def plot_locked(plots, out, rules, name, gradient='both'):
    """Redraw the plots of a phase; a file lock keeps parallel parts from
    drawing into the same folder at the same time."""
    with open(os.path.join(out, 'plots.lock'), 'w') as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            plots.make_all(out, rules, name, gradient)
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


PLATEAU = ('ABSOLUTE_FITNESS_PLATEAU_REACHED', 'RELATIVE_FITNESS_PLATEAU_REACHED')


def needs_run(job):
    """Run if there is no result, it crashed, or it stopped on a fitness plateau
    above SUCCESS_F with a larger tolerance than the current one (runs are
    seeded, so the rerun follows the same path and only continues longer)."""
    if not os.path.exists(job['out']):
        return True
    with open(job['out'], 'rb') as fh:
        r = pickle.load(fh)
    return (r['info'].startswith('CRASH')
            or (r['info'] in PLATEAU and r['fbest'] >= C.SUCCESS_F
                and r['solver']['tolerance'] > job['solver']['tolerance']))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) // 2),
                    help='parallel runs (default: half of the CPU cores)')
    ap.add_argument('--out', default=os.path.join(HERE, C.RESULTS_DIR), help='results directory')
    ap.add_argument('--quick', action='store_true',
                    help='short test run (smallest n, 2 start points, 2000 generations)')
    ap.add_argument('--plots-only', action='store_true', help='only redo the plots')
    ap.add_argument('--phase', nargs='+', metavar='NAME',
                    help='run only these phases (e.g. 3_age_noise); plots still compare with all earlier phases')
    ap.add_argument('--problems', nargs='+', metavar='NAME',
                    help='run only these problems (seamaze names, e.g. Sphere "Rotated Ellipsoid")')
    ap.add_argument('--gradient', choices=('both', 'no', 'yes'), default='both',
                    help="with --plots-only: plot all runs (both), only those without (no) or with (yes) gradient")
    ap.add_argument('--rules', nargs='+', metavar='RULE',
                    help='with --plots-only: plot only these rules, into plots/only_<rules>')
    ap.add_argument('--part', default='1/1', metavar='K/N',
                    help='run only part K of N (every N-th run); the parts do not overlap')
    args = ap.parse_args()
    try:
        part, nparts = (int(x) for x in args.part.split('/'))
        assert 1 <= part <= nparts
    except (ValueError, AssertionError):
        ap.error('--part must be K/N with 1 <= K <= N, e.g. 2/4')
    tag = f'[part {part}/{nparts}] ' if nparts > 1 else ''
    names = [name for name, _ in C.PHASES]
    if args.phase and not set(args.phase) <= set(names):
        ap.error(f'unknown phase; available: {", ".join(names)}')
    if args.problems and not set(args.problems) <= set(C.PROBLEMS):
        ap.error(f'unknown problem; available: {", ".join(C.PROBLEMS)}')
    problems = args.problems or C.PROBLEMS
    known = [v for _, rules in C.PHASES for v in rules]
    if args.rules and not set(args.rules) <= set(known):
        ap.error(f'unknown rule; available: {", ".join(known)}')

    from runner import BENCHMARKS
    unknown = [p for p in C.PROBLEMS if p not in BENCHMARKS]
    if unknown:
        ap.error(f'not a seamaze benchmark: {unknown}; available: {", ".join(sorted(BENCHMARKS))}')
    out = args.out + ('_quick' if args.quick else '')
    run_dir = os.path.join(out, 'runs')
    os.makedirs(run_dir, exist_ok=True)
    logf = os.path.join(out, 'log.txt')

    import plots    # after the thread settings
    from variants import build
    all_rules = []
    if args.plots_only:
        if args.rules:
            plot_locked(plots, out, args.rules, 'only_' + '_'.join(args.rules), args.gradient)
        for name, rules in ([] if args.rules else C.PHASES):
            all_rules += rules
            plot_locked(plots, out, list(all_rules), name, args.gradient)
            if len(all_rules) > len(rules):    # later phases also by themselves
                plot_locked(plots, out, rules, name + '_only', args.gradient)
        log(f'plots written to {os.path.join(out, "plots")}', logf)
        return

    modules = build(os.path.join(out, 'optimizer_variants'))
    # fixed optimizer copies, e.g. from another branch (EXTRA_VARIANTS in the settings file)
    modules.update({r: os.path.join(HERE, p) for r, p in getattr(C, 'EXTRA_VARIANTS', {}).items()})
    shutil.copy(C.__file__, os.path.join(out, f'{C.__name__}_used.py'))
    log(f'{tag}results in {out}, {args.workers} workers', logf)
    for n in C.DIMS:
        src = 'provided' if os.path.exists(os.path.join(HERE, f'starting_points_dim{n}.npz')) else 'generated'
        log(f'{tag}n = {n}: {len(start_points(n))} start points ({src})', logf)

    for name, rules in C.PHASES:
        all_rules += rules
        if args.phase and name not in args.phase:
            continue
        jobs = jobs_for(rules, modules, run_dir, args.quick, problems)[part - 1::nparts]
        todo = [j for j in jobs if needs_run(j)]
        log(f'{tag}phase {name}: rules {rules}, {len(jobs)} runs, {len(todo)} to do', logf)
        last_plot, done = time.time(), 0
        if todo:
            from runner import run
            with get_context('spawn').Pool(args.workers, maxtasksperchild=50) as pool:
                for path, info, fbest, iters, wall in pool.imap_unordered(run, todo):
                    done += 1
                    log(f'{tag}{done}/{len(todo)} {os.path.basename(path)[:-4]}: {info}, f = {fbest:.2e}, '
                        f'{iters} generations, {wall / 60:.1f} min', logf)
                    if part == 1 and time.time() - last_plot > 60 * C.PLOT_EVERY_MIN:
                        plot_locked(plots, out, list(all_rules), name)
                        last_plot = time.time()
        plot_locked(plots, out, list(all_rules), name)
        if len(all_rules) > len(rules):
            plot_locked(plots, out, rules, name + '_only')
        log(f'{tag}phase {name} done; plots in {os.path.join(out, "plots", name)}', logf)
    log(tag + ('ALL DONE' if not (args.phase or args.problems)
        else f'DONE: phases {" ".join(args.phase or names)}, problems {", ".join(problems)}'), logf)


if __name__ == '__main__':
    main()
