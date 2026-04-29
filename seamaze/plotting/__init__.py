"""
Plotting module.

==================================================================

This module aims to provide methods and classes for plotting.
"""

# Author: Tim Ortkamp, Chinmay Patwardhan, Pia Stammer

from seamaze.plotting._plot_fitness import plot_fitness
from seamaze.plotting._plot_matrix_slices import plot_matrix_slices
from seamaze.plotting._plot_scatter import plot_scatter
from seamaze.plotting._plot_series import plot_series
from seamaze.plotting._plot_results import plot_results

from seamaze.plotting._visualizer_2d import Visualizer2D

__all__ = [
    'plot_fitness',
    'plot_matrix_slices',
    'plot_scatter',
    'plot_series',
    'plot_results',
    'Visualizer2D']
