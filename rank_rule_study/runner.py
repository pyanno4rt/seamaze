"""Single run of DLR-CMA-ES with one rank rule (executed in worker processes)."""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[_v] = '1'    # one thread per run; parallelism comes from the workers
import importlib.util
import pickle
import sys
import time
import traceback
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import seamaze.benchmarks as B    # noqa: E402

N_KEEP = 2000    # stored points per trajectory

# seamaze benchmarks by their names (problem.name)
BENCHMARKS = {cls(2).name: cls for cls in (getattr(B, c) for c in B.__all__)
              if isinstance(cls, type) and issubclass(cls, B.BenchmarkFunction) and cls is not B.BenchmarkFunction}


def slug(name):
    return name.replace(' ', '_')


class Recorder:
    """Best f so far, rank, elapsed time and low-rank contribution per generation."""

    def __init__(self):
        self.k, self.t0 = 0, time.perf_counter()
        self.data = np.zeros((4, 1 << 16))

    def __call__(self, s):
        if self.k == self.data.shape[1]:
            self.data = np.concatenate([self.data, np.zeros_like(self.data)], axis=1)
        # ||U S U^T||_F / ||U S U^T + diag(psi)||_F with orthonormal U, in O(n r)
        core, psi = s._core, s._psi
        lr2 = np.sum(core**2)
        lr_diag = (s._basis**2) @ core
        cov2 = lr2 + 2.0 * psi @ lr_diag + psi @ psi
        self.data[:, self.k] = (s._result['optimal_value'], s.rank, time.perf_counter() - self.t0,
                                np.sqrt(lr2 / cov2) if cov2 > 0 else np.nan)
        self.k += 1


_MODULES = {}    # optimizer variants loaded (and numba-compiled) once per worker


def load_variant(rule, path):
    if rule not in _MODULES:
        spec = importlib.util.spec_from_file_location(f'dlr_{rule}', path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _MODULES[rule] = mod
    return _MODULES[rule]


def run(job):
    """job: dict(rule, module, problem, ndim, start, x0, pop, grad, solver,
    global_seed, out). Writes a pickle with thinned trajectories."""
    t_start = time.time()
    try:
        mod = load_variant(job['rule'], job['module'])
        problem = BENCHMARKS[job['problem']](job['ndim'])
        rec = Recorder()
        if job['global_seed'] is not None:
            np.random.seed(job['global_seed'])
        s = mod.DLRCMAES(number_of_variables=job['ndim'], objective=problem.__call__,
                         gradient=problem.gradient if job['grad'] else None,
                         number_of_individuals=job['pop'], min_log_level='critical', callback=rec,
                         **job['solver'])
        rec.t0 = time.perf_counter()
        try:
            res = s.optimize(np.array(job['x0'], dtype=float))
            info = res['solver_info']
        except np.linalg.LinAlgError as e:
            # The covariance became numerically singular (condition number near
            # 1/eps) and a linear solve failed before the condition-number check
            # stopped the run; counted as a stop, the trajectory is kept.
            res, info = None, f'SINGULAR_COVARIANCE ({e})'
        m = rec.k
        if m == 0:
            raise RuntimeError(f'no generation completed ({info})')
        gen = np.arange(1, m + 1)
        keep = np.unique(np.concatenate([np.geomspace(1, m, N_KEEP // 2).astype(int) - 1,
                                         np.linspace(0, m - 1, N_KEEP // 2).astype(int)]))
        f, rank, t, lrc = rec.data[:, :m]
        out = dict(job, gen=gen[keep], f=f[keep], rank=rank[keep].astype(int), time=t[keep], lr_contrib=lrc[keep],
                   iters=res['iterations'] if res else m,
                   fbest=float(res['optimal_value']) if res else float(f[-1]),
                   info=info, popsize=s._pop_size, wall=time.time() - t_start)
    except Exception as e:    # keep the sweep going; the failure is recorded
        out = dict(job, gen=np.array([]), f=np.array([]), rank=np.array([]), time=np.array([]),
                   lr_contrib=np.array([]), iters=-1, fbest=float('inf'), info=f'CRASH {type(e).__name__}: {e}',
                   popsize=0, wall=time.time() - t_start, traceback=traceback.format_exc())
    tmp = job['out'] + '.tmp'
    with open(tmp, 'wb') as fh:
        pickle.dump(out, fh)
    os.replace(tmp, job['out'])
    return job['out'], out['info'], out['fbest'], out['iters'], out['wall']
