# Unit tests for the plotting functions and the result plotter
# Authors: Tim Ortkamp, Chinmay Patwardhan, Pia Stammer

import re
from functools import cache

import matplotlib
import matplotlib.pyplot as plt
import pytest
from numpy import arange, full

from seamaze.benchmarks import Sphere
from seamaze.diagnostics import MonitorCMAES, MonitorDLRCMAES
from seamaze.optimizers import CMAES, DLRCMAES
from seamaze.plotting import (
    ResultPlotter, plot_bound_violations, plot_fitness, plot_matrix_slices,
    plot_series)

SERIES = arange(1.0, 11.0)

# Plot function, keyword arguments and expected number of plotted lines
PLOT_FUNCTIONS = {
    'series': (plot_series, {'series': SERIES}, 1),
    'fitness': (
        plot_fitness, {'fitness': [SERIES, 2 * SERIES, 3 * SERIES]}, 3),
    'bound_violations': (
        plot_bound_violations, {'violation': [SERIES, SERIES / 2]}, 2),
    # With axis=0, one line is plotted per matrix column
    'matrix_slices': (
        plot_matrix_slices,
        {'matrix': arange(1.0, 13.0).reshape(4, 3), 'axis': 0}, 3),
    }


@pytest.fixture
def shown(monkeypatch):
    """
    Replace plt.show so that no window opens and nothing blocks, recording
    the properties of each figure that would have been shown.
    """
    figures = []

    def fake_show(**kwargs):
        axes = plt.gcf().axes[0]
        figures.append({
            'title': axes.get_title(),
            'lines': len(axes.get_lines()),
            'yscale': axes.get_yscale(),
            'block': kwargs.get('block')})

    monkeypatch.setattr(plt, 'show', fake_show)
    return figures


# %% Tests of the plotting functions

# Check that calling a plotting function without a save path shows the
# figure (blocking) with the expected content and closes it afterwards
@pytest.mark.parametrize('name', PLOT_FUNCTIONS)
def test_plotShowsFigure(name, shown):
    function, kwargs, num_lines = PLOT_FUNCTIONS[name]
    function(**kwargs, title='Test title')

    assert shown == [{'title': 'Test title', 'lines': num_lines,
                      'yscale': 'linear', 'block': True}]
    assert plt.get_fignums() == []


# Check that a given save path writes the figure to a file 
@pytest.mark.parametrize('name', PLOT_FUNCTIONS)
def test_plotSavesFigure(name, shown, tmp_path):
    function, kwargs, _ = PLOT_FUNCTIONS[name]
    path = tmp_path / f'{name}.pdf'
    function(**kwargs, save_path=str(path))

    assert path.read_bytes().startswith(b'%PDF')
    assert shown == []
    assert plt.get_fignums() == []


@pytest.mark.parametrize('semilog, yscale', [(False, 'linear'), (True, 'log')])
def test_semilogScale(semilog, yscale, shown):
    plot_series(SERIES, semilog=semilog)

    assert shown[0]['yscale'] == yscale


# An empty series is skipped without creating a figure
def test_emptySeriesIsSkipped(shown):
    plot_series([])

    assert shown == []
    assert plt.get_fignums() == []


# Check that showing works without a display
@pytest.mark.filterwarnings('ignore:.*non-interactive:UserWarning')
def test_showWithoutDisplay():
    assert matplotlib.get_backend().lower() == 'agg'

    plot_series(SERIES)

    assert plt.get_fignums() == []


# %% Tests of the result plotter with real monitor data

# Plot files written by plot_all (after the date prefix) for label 'test'
CMAES_FILES = {
    f'test_{name}.pdf' for name in (
        'optimal_value', 'fitness', 'squared_bound_viols', 'gamma', 'sigma',
        'mean_change_norm', 'path_sigma_norm', 'path_cov_norm',
        'cov_svs_start', 'cov_svs_mid', 'cov_svs_end', 'cov_svs', 'cov_cn',
        'cov_spectr_norm')}
DLRCMAES_FILES = CMAES_FILES | {
    f'test_{name}.pdf' for name in (
        'rank', 'low_rank_contribution', 'low_rank_offdiag_contribution',
        'low_rank_correlation_strength')}

PAIRS = {
    'CMAES': (CMAES, MonitorCMAES, CMAES_FILES),
    'DLRCMAES': (DLRCMAES, MonitorDLRCMAES, DLRCMAES_FILES),
    }


@cache
def monitor_data(pair):
    """Record the full monitor data of a short optimization run."""
    solver_class, monitor_class, _ = PAIRS[pair]
    with monitor_class() as monitor:
        solver = solver_class(
            number_of_variables=5, objective=Sphere(5), maximum_iterations=10,
            min_log_level='critical', callback=monitor.full, random_state=42)
        solver.optimize(full(5, 2.0))
    return monitor.data


# Check that all plots are saved into a (newly created) folder with a date
# prefix and the label in the filename, without showing any figure
@pytest.mark.parametrize('pair', PAIRS)
def test_resultPlotterSavesAll(pair, shown, tmp_path):
    folder = tmp_path / 'new' / 'plots'
    ResultPlotter(monitor_data(pair), 'test', save_folder=folder).plot_all()
    names = [path.name for path in folder.iterdir()]

    assert all(re.match(r'\d{4}-\d{2}-\d{2}_test_', name) for name in names)
    assert {name.split('_', 1)[1] for name in names} == PAIRS[pair][2]
    assert shown == []
    assert plt.get_fignums() == []


# Without a save folder, the same plots are shown and nothing is written
@pytest.mark.parametrize('pair', PAIRS)
def test_resultPlotterShowsAll(pair, shown, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ResultPlotter(monitor_data(pair), 'test').plot_all()

    assert len(shown) == len(PAIRS[pair][2])
    assert all(figure['title'].endswith('(test)') for figure in shown)
    assert list(tmp_path.iterdir()) == []


# Only the plots whose flag is enabled are created
def test_resultPlotterFlags(shown):
    plotter = ResultPlotter(monitor_data('DLRCMAES'), 'test')
    for flag in [name for name in vars(plotter) if name.startswith('show_')]:
        setattr(plotter, flag, False)
    plotter.plot_all()

    assert shown == []

    plotter.show_step_size = True
    plotter.plot_all()

    assert [figure['title'] for figure in shown] == ['Step size (test)']


# Metrics missing from the data are skipped
def test_resultPlotterMissingData(shown):
    ResultPlotter({}, 'test').plot_all()

    assert shown == []
