"""Plots and CSV tables of the rank-rule study.

Usage (also called by run_study.py): python plots.py <results_dir> [rules ...]
Reads the run pickles in <results_dir>/runs and writes into <results_dir>/plots/<name>.
Problems are labelled with their seamaze names; the start points of the
benchmark script act as repetitions.
"""
import csv
import glob
import os
import pickle
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt    # noqa: E402
from matplotlib.lines import Line2D    # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib    # noqa: E402
C = importlib.import_module(os.environ.get('STUDY_CONFIG', 'study_config'))    # settings file, see README
from variants import LABEL    # noqa: E402

COLOR = {'current': '#2a78d6', 'soft_eps': '#eb6834', 'matrix_bernstein': '#eda100',
         'bernstein_age': '#1baf7a', 'mp_hysteresis': '#7a4fc2',
         'bootstrap': '#c2417e', 'bootstrap_nohyst': '#5b5b56'}
LSTYLE = ['-', '--', ':', '-.']
SET_MARK = ['o', 's', '^', 'D']
DIM_MARK = ['o', 's', '^', 'D', 'v', 'P', 'X', '*']
INK, INK2 = '#1f1f1e', '#5f5e58'
plt.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True,
                     'grid.color': '#e4e3dd', 'legend.frameon': False, 'font.size': 8})


def pop_label(pop):
    return 'λ default' if pop is None else f'λ = {pop}'


def setting_label(pop, grad):
    return f"{pop_label(pop)}, {'with' if grad else 'no'} gradient"


SETTINGS = [(pop, grad) for pop in C.POP_SIZES for grad in C.USE_GRADIENT]


def load(results_dir, rules):
    runs = []
    for f in sorted(glob.glob(os.path.join(results_dir, 'runs', '*.pkl'))):
        with open(f, 'rb') as fh:
            r = pickle.load(fh)
        if r['rule'] in rules:
            runs.append(r)
    return runs


def summarize(r):
    g, k, f = r['gen'], r['rank'].astype(float), r['f']
    ok = len(g) > 0
    mean_rank = float(np.sum(0.5 * (k[1:] + k[:-1]) * np.diff(g)) / (g[-1] - g[0])) if len(g) > 1 else \
        (float(k[0]) if ok else np.nan)
    return dict(rule=r['rule'], problem=r['problem'], ndim=r['ndim'], start=r['start'],
                pop=r['pop'], grad=r['grad'], setting=setting_label(r['pop'], r['grad']),
                popsize=r['popsize'], generations=int(g[-1]) if ok else 0,
                evaluations=int(g[-1]) * r['popsize'] if ok else 0,
                time_s=float(r['time'][-1]) if ok else np.nan,
                final_best_f=float(f[-1]) if ok else np.inf,
                solved=int(ok and f[-1] < C.SUCCESS_F),
                mean_rank=mean_rank, max_rank=int(k.max()) if ok else 0, final_rank=int(k[-1]) if ok else 0,
                final_low_rank_contribution=float(r['lr_contrib'][-1]) if ok else np.nan,
                final_info=r['info'])


def write_csv(runs, out):
    rows = sorted((summarize(r) for r in runs),
                  key=lambda d: (d['problem'], d['ndim'], d['setting'], d['rule'], d['start']))
    with open(os.path.join(out, 'summary_runs.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(out, 'trajectories.csv'), 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['rule', 'problem', 'ndim', 'start', 'setting', 'popsize', 'generation',
                    'evaluations', 'time_s', 'best_f', 'rank', 'low_rank_contribution'])
        for r in runs:
            g = r['gen']
            if not len(g):
                continue
            t = np.unique(np.concatenate([np.geomspace(g[0], g[-1], 150), np.linspace(g[0], g[-1], 150)]))
            for i in np.unique(np.append(np.searchsorted(g, t).clip(0, len(g) - 1), len(g) - 1)):
                w.writerow([r['rule'], r['problem'], r['ndim'], r['start'], setting_label(r['pop'], r['grad']),
                            r['popsize'], int(g[i]), int(g[i]) * r['popsize'], f"{r['time'][i]:.4f}",
                            f"{r['f'][i]:.6e}", int(r['rank'][i]), f"{r['lr_contrib'][i]:.4f}"])
    return rows


def grid(nplots, ncols=3, w=5.0, h=3.6):
    nrows = int(np.ceil(nplots / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(w * ncols, h * nrows), constrained_layout=True,
                            squeeze=False)
    for ax in axs.ravel()[nplots:]:
        ax.axis('off')
    return fig, axs.ravel()


def problems_in(rows):
    return [p for p in C.PROBLEMS if any(r['problem'] == p for r in rows)]


def scaling(rows, rules, key, ylabel, fname, title, logy=True):
    """Value vs. dimension per problem; markers: runs (x: not solved), lines: median."""
    fig, axs = grid(len(problems_in(rows)))
    for ax, p in zip(axs, problems_in(rows)):
        for v in rules:
            for i, (pop, grad) in enumerate(SETTINGS):
                rr = [r for r in rows if r['problem'] == p and r['rule'] == v and r['pop'] == pop and r['grad'] == grad]
                if not rr:
                    continue
                dims = sorted({r['ndim'] for r in rr})
                med = []
                for n in dims:
                    xs = [r for r in rr if r['ndim'] == n]
                    jit = n * (1 + 0.05 * (i - 1.5))
                    ok = [r[key] for r in xs if r['solved']]
                    bad = [r[key] for r in xs if not r['solved']]
                    ax.plot([jit] * len(ok), ok, ls='none', marker=SET_MARK[i], ms=4, color=COLOR[v], alpha=0.8)
                    ax.plot([jit] * len(bad), bad, ls='none', marker='x', ms=5, color=COLOR[v])
                    med.append(np.median(ok + bad))
                ax.plot(dims, med, ls=LSTYLE[i], color=COLOR[v], lw=1.2, alpha=0.8)
        ax.set_xscale('log')
        if logy:
            ax.set_yscale('log')
        ax.set_xticks(C.DIMS, [str(n) for n in C.DIMS])
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_title(p, fontweight='bold')
        ax.set_xlabel('dimension n')
        ax.set_ylabel(ylabel)
    h = [Line2D([], [], color=COLOR[v], lw=2, label=LABEL[v]) for v in rules] + \
        [Line2D([], [], color=INK2, marker=SET_MARK[i], ls=LSTYLE[i], label=setting_label(*s))
         for i, s in enumerate(SETTINGS)] + \
        [Line2D([], [], color=INK2, marker='x', ls='none', label=f'final f ≥ {C.SUCCESS_F:g}')]
    fig.legend(handles=h, loc='outside lower center', ncol=4, fontsize=8)
    fig.suptitle(title, fontweight='bold', color=INK)
    fig.savefig(fname, dpi=110)
    plt.close(fig)


def success(rows, rules, fname):
    """Solved runs / runs, median generations of solved runs and mean rank."""
    probs = problems_in(rows)
    dims = sorted({r['ndim'] for r in rows})
    fig, axs = plt.subplots(len(SETTINGS), len(rules),
                            figsize=(1.1 * len(dims) * len(rules) + 3, 0.45 * len(probs) * len(SETTINGS) + 1.5),
                            constrained_layout=True, squeeze=False)
    for i, (pop, grad) in enumerate(SETTINGS):
        for j, v in enumerate(rules):
            ax = axs[i, j]
            M = np.full((len(probs), len(dims)), np.nan)
            for a, p in enumerate(probs):
                for b, n in enumerate(dims):
                    rr = [r for r in rows if r['problem'] == p and r['ndim'] == n and r['rule'] == v
                          and r['pop'] == pop and r['grad'] == grad]
                    if not rr:
                        continue
                    ok = [r['generations'] for r in rr if r['solved']]
                    M[a, b] = len(ok) / len(rr)
                    txt = f"{len(ok)}/{len(rr)}  " + (f"{np.median(ok) / 1000:.1f}k" if ok else '–') + \
                        f"  r {np.mean([r['mean_rank'] for r in rr]):.0f}"
                    ax.text(b, a, txt, ha='center', va='center', fontsize=6.5, color=INK)
            cm = plt.get_cmap('RdYlGn').copy()
            cm.set_bad('#eeeeee')
            ax.imshow(np.ma.masked_invalid(M), cmap=cm, vmin=0, vmax=1, aspect='auto', alpha=0.55)
            ax.set_xticks(range(len(dims)), [f'n={n}' for n in dims])
            ax.set_yticks(range(len(probs)), probs)
            ax.grid(False)
            ax.set_title(f'{LABEL[v]}\n{setting_label(pop, grad)}', fontsize=8.5, fontweight='bold')
    fig.suptitle(f'Runs with final f < {C.SUCCESS_F:g} / runs (start points), median generations of these '
                 f'runs, mean rank (r)', fontweight='bold', color=INK)
    fig.savefig(fname, dpi=110)
    plt.close(fig)


def final_scatter(rows, rules, pop, xkey, xlabel, fname):
    """Final best f of every run vs. its cost; one axis per problem.
    Colour = rule, marker = dimension, filled = no gradient, hollow = with gradient."""
    sel = [r for r in rows if r['pop'] == pop and r['rule'] in rules and np.isfinite(r[xkey]) and r[xkey] > 0]
    dims = sorted({r['ndim'] for r in rows})
    fig, axs = grid(len(problems_in(rows)))
    for ax, p in zip(axs, problems_in(rows)):
        for r in sel:
            if r['problem'] != p:
                continue
            c = COLOR[r['rule']]
            ax.plot(r[xkey], max(r['final_best_f'], 1e-18), ls='none',
                    marker=DIM_MARK[dims.index(r['ndim']) % len(DIM_MARK)], ms=6, color=c,
                    markerfacecolor='none' if r['grad'] else c, markeredgewidth=1.2 if r['grad'] else 0.5,
                    markeredgecolor=c if r['grad'] else 'white', alpha=0.85)
        ax.set_xscale('log')
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_yscale('log')
        ax.axhline(C.SUCCESS_F, color=INK2, ls=':', lw=0.8)
        ax.set_title(p, fontweight='bold')
        ax.set_xlabel(xlabel)
        ax.set_ylabel('final best f')
    h = [Line2D([], [], color=COLOR[v], marker='o', ls='none', ms=7, label=LABEL[v]) for v in rules] + \
        [Line2D([], [], color=INK2, marker=DIM_MARK[i % len(DIM_MARK)], ls='none', ms=7, label=f'n = {n}')
         for i, n in enumerate(dims)] + \
        [Line2D([], [], color=INK2, marker='o', ls='none', ms=7, label='filled: no gradient'),
         Line2D([], [], color=INK2, marker='o', ls='none', ms=7, markerfacecolor='none', markeredgewidth=1.2,
                label='hollow: with gradient')]
    fig.legend(handles=h, loc='outside lower center', ncol=4, fontsize=8)
    fig.suptitle(f'Final best f vs. {xlabel}, {pop_label(pop)} (one marker per run; dotted: f = {C.SUCCESS_F:g})',
                 fontweight='bold', color=INK)
    fig.savefig(fname, dpi=110)
    plt.close(fig)


def trajectories(runs, rules, pop, ykey, fname):
    """Rank or best f over generations; one panel per problem and n, one line per
    run; solid: no gradient, dashed: with gradient."""
    probs = [p for p in C.PROBLEMS if any(r['problem'] == p for r in runs)]
    dims = sorted({r['ndim'] for r in runs})
    fig, axs = plt.subplots(len(probs), len(dims), figsize=(3.6 * len(dims), 2.2 * len(probs)),
                            constrained_layout=True, squeeze=False)
    for a, p in enumerate(probs):
        for b, n in enumerate(dims):
            ax = axs[a, b]
            for r in runs:
                if r['problem'] == p and r['ndim'] == n and r['pop'] == pop and len(r['gen']):
                    y = np.maximum(r[ykey], 1e-18) if ykey == 'f' else r[ykey]
                    ax.plot(r['gen'], y, color=COLOR[r['rule']], lw=0.8, alpha=0.6, ls='--' if r['grad'] else '-')
            if ykey == 'f':
                ax.set_yscale('log')
                ax.axhline(C.SUCCESS_F, color=INK2, ls=':', lw=0.8)
            ax.ticklabel_format(axis='x', style='sci', scilimits=(-3, 4))
            ax.tick_params(labelsize=6.5)
            ax.set_title(f'{p}, n={n}', fontsize=8, fontweight='bold')
            if b == 0:
                ax.set_ylabel('rank' if ykey == 'rank' else 'best f')
            if a == len(probs) - 1:
                ax.set_xlabel('generations')
    h = [Line2D([], [], color=COLOR[v], lw=2, label=LABEL[v]) for v in rules] + \
        [Line2D([], [], color=INK2, ls='-', label='no gradient'), Line2D([], [], color=INK2, ls='--', label='with gradient')]
    fig.legend(handles=h, loc='outside lower center', ncol=len(h), fontsize=8)
    what = 'Rank' if ykey == 'rank' else 'Best f'
    fig.suptitle(f'{what} over generations, {pop_label(pop)} (one line per run)', fontweight='bold', color=INK)
    fig.savefig(fname, dpi=100)
    plt.close(fig)


def make_all(results_dir, rules, name, gradient='both'):
    """gradient: 'both', 'no' or 'yes' -- plot all runs, only those without or
    only those with gradient (the latter two into <name>_no_gradient / _with_gradient)."""
    global SETTINGS
    keep = {'both': (False, True), 'no': (False,), 'yes': (True,)}[gradient]
    SETTINGS = [(pop, grad) for pop in C.POP_SIZES for grad in C.USE_GRADIENT if grad in keep]
    runs = [r for r in load(results_dir, rules) if r['grad'] in keep]
    if not runs:
        return 0
    if gradient != 'both':
        name += '_no_gradient' if gradient == 'no' else '_with_gradient'
    out = os.path.join(results_dir, 'plots', name)
    os.makedirs(out, exist_ok=True)
    for f in glob.glob(os.path.join(out, '*.png')) + glob.glob(os.path.join(out, '*.csv')):
        os.remove(f)    # start each update from an empty folder, so no stale figures remain
    rules = [v for v in rules if any(r['rule'] == v for r in runs)]
    rows = write_csv(runs, out)
    success(rows, rules, os.path.join(out, 'success_table.png'))
    scaling(rows, rules, 'generations', 'generations used', os.path.join(out, 'scaling_generations.png'),
            'Generations used vs. dimension (markers: runs, x: final f not below threshold; lines: median)')
    scaling(rows, rules, 'mean_rank', 'mean rank over the run', os.path.join(out, 'scaling_rank.png'),
            'Mean rank vs. dimension (markers: runs; lines: median)', logy=False)
    scaling(rows, rules, 'time_s', 'runtime [s]', os.path.join(out, 'scaling_runtime.png'),
            'Runtime vs. dimension (markers: runs, x: final f not below threshold; lines: median)')
    for pop in C.POP_SIZES:
        tag = 'default' if pop is None else str(pop)
        final_scatter(rows, rules, pop, 'time_s', 'runtime [s]', os.path.join(out, f'final_f_vs_runtime_pop{tag}.png'))
        final_scatter(rows, rules, pop, 'generations', 'generations used',
                      os.path.join(out, f'final_f_vs_generations_pop{tag}.png'))
        trajectories(runs, rules, pop, 'rank', os.path.join(out, f'traj_rank_pop{tag}.png'))
        trajectories(runs, rules, pop, 'f', os.path.join(out, f'traj_f_pop{tag}.png'))
    return len(runs)


if __name__ == '__main__':
    d = sys.argv[1]
    rules = sys.argv[2:] or [v for _, vs in C.PHASES for v in vs]
    print('plotted', make_all(d, rules, 'manual_' + '_'.join(rules)), 'runs')
