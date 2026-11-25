"""Final best f vs. generations and vs. runtime of the reference runs (CMA-ES,
LM-MA-ES), in the style of the rank-rule plots (plots.final_scatter).

Usage: python rank_rule_study/plot_reference.py
Plots: rank_rule_study/results_reference/plots/final_f_vs_{generations,runtime}.png
"""
import glob
import os
import pickle
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plots import C, DIM_MARK, INK, INK2, Line2D, grid, matplotlib, plt    # noqa: E402  (style of the rank-rule plots)
from run_reference import ALGOS, OUT, PROBLEMS    # noqa: E402

COLOR = {'CMAES': '#7a4fc2', 'LMMAES': '#c2417e'}
LABEL = {'CMAES': 'CMA-ES', 'LMMAES': 'LM-MA-ES'}


def final_scatter(runs, xkey, xlabel, fname):
    """Final best f of every run vs. its cost; one axis per problem.
    Colour = algorithm, marker = dimension."""
    dims = sorted({r['ndim'] for r in runs})
    problems = [p for p in PROBLEMS if any(r['problem'] == p for r in runs)]
    fig, axs = grid(len(problems), ncols=len(problems))
    for ax, p in zip(axs, problems):
        for r in runs:
            if r['problem'] != p or not r[xkey] > 0:
                continue
            c = COLOR[r['algo']]
            ax.plot(r[xkey], max(r['fbest'], 1e-18), ls='none', marker=DIM_MARK[dims.index(r['ndim']) % len(DIM_MARK)],
                    ms=6, color=c, markeredgewidth=0.5, markeredgecolor='white', alpha=0.85)
        ax.set_xscale('log')
        x = [r[xkey] for r in runs if r['problem'] == p and r[xkey] > 0]
        if x and max(x) / min(x) > 10:    # minor labels only if less than one decade is shown
            ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_yscale('log')
        ax.axhline(C.SUCCESS_F, color=INK2, ls=':', lw=0.8)
        ax.set_title(p, fontweight='bold')
        ax.set_xlabel(xlabel)
        ax.set_ylabel('final best f')
    h = [Line2D([], [], color=COLOR[a], marker='o', ls='none', ms=7, label=LABEL[a]) for a in ALGOS] + \
        [Line2D([], [], color=INK2, marker=DIM_MARK[i % len(DIM_MARK)], ls='none', ms=7, label=f'n = {n}')
         for i, n in enumerate(dims)]
    fig.legend(handles=h, loc='outside lower center', ncol=4, fontsize=8)
    fig.suptitle(f'Final best f vs. {xlabel}, λ default, no gradient '
                 f'(one marker per run; dotted: f = {C.SUCCESS_F:g})', fontweight='bold', color=INK)
    fig.savefig(fname, dpi=110)
    plt.close(fig)


def main():
    runs = []
    for fn in sorted(glob.glob(os.path.join(OUT, 'runs', '*.pkl'))):
        with open(fn, 'rb') as fh:
            r = pickle.load(fh)
        if not r['info'].startswith('CRASH'):
            runs.append(r)
    if not runs:
        print(f"no runs in {os.path.join(OUT, 'runs')}")
        return
    out = os.path.join(OUT, 'plots')
    os.makedirs(out, exist_ok=True)
    for f in glob.glob(os.path.join(out, '*.png')):
        os.remove(f)
    final_scatter(runs, 'iters', 'generations', os.path.join(out, 'final_f_vs_generations.png'))
    final_scatter(runs, 'wall', 'runtime [s]', os.path.join(out, 'final_f_vs_runtime.png'))
    print(f'{len(runs)} runs; plots written to {out}')


if __name__ == '__main__':
    main()
