"""2D interactive visualizer."""

# Authors: Tim Ortkamp, Chinmay Patwardhan, Pia Stammer

# %% External package import

import matplotlib
def enable_interactive_backend():
    for backend in ['TkAgg', 'Qt5Agg', 'MacOSX']:
        try:
            matplotlib.use(backend)
            return True
        except ImportError:
            continue
    return False
enable_interactive_backend()

import matplotlib.pyplot as plt
from numpy import (
    argmin, cos, linspace, maximum, meshgrid, pi, sin, sqrt, stack,
    unravel_index, zeros, zeros_like)
from scipy.linalg import eigh

# %% Class definition


class Visualizer2D:
    """
    2D interactive visualizer class.

    Parameters
    ----------
    ...

    Attributes
    ----------
    ...
    """

    def __init__(
            self,
            objective,
            bounds,
            dimensions,
            grid_points=100):

        #
        plt.rcParams.update({
            "text.usetex": False,
            "font.family": 'sans-serif',
            "font.serif": [
                'Helvetica', 'Arial', 'DejaVu Sans', 'Liberation Sans'],
            "axes.labelsize": 10,
            "font.size": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.labelweight": 'normal',
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.grid": False,
            "figure.dpi": 100
            })

        #
        self.objective = objective
        self.dimensions = dimensions

        # Extract scalar limits
        self.x_min, self.x_max = bounds[0][0], bounds[1][0]
        self.y_min, self.y_max = bounds[0][1], bounds[1][1]

        # Create the figure
        self.fig, self.ax = plt.subplots(figsize=(10, 8))

        # Manual adjustment: reserve 15% space at the top for the header
        self.fig.subplots_adjust(top=0.85, bottom=0.1, left=0.1, right=0.95)

        # Precompute landscape
        x = linspace(self.x_min, self.x_max, grid_points)
        y = linspace(self.y_min, self.y_max, grid_points)
        X, Y = meshgrid(x, y)
        Z = zeros_like(X)

        #
        for i in range(grid_points):

            #
            for j in range(grid_points):

                #
                p = zeros(dimensions)
                p[0], p[1] = X[i, j], Y[i, j]
                Z[i, j] = self.objective(p)

        # Plot background contours
        self.cnt = self.ax.contourf(
            X, Y, Z, levels=50, cmap='viridis', alpha=0.7)
        self.ax.contour(
            X, Y, Z, levels=20, colors='white', alpha=0.15, linewidths=0.5)

        # Mark Global Minimum
        min_idx = unravel_index(argmin(Z), Z.shape)
        self.ax.scatter(
            X[min_idx], Y[min_idx], color='orange', marker='*', s=100,
            linewidths=2, label='Global minimum (grid)', zorder=12
            )

        # Colorbar setup
        cbar = self.fig.colorbar(self.cnt, ax=self.ax, shrink=0.8, pad=0.05)
        cbar.set_label(r'$f$', rotation=0, labelpad=15)

        #
        self.ax.set_xlabel(r'$x_1$')
        self.ax.set_ylabel(r'$x_2$')

        #
        self.mean_dot = None
        self.cov_line = None

        #
        self.ax.set_xlim(self.x_min, self.x_max)
        self.ax.set_ylim(self.y_min, self.y_max)
        self.ax.legend(loc='upper right', fontsize=8, framealpha=0.6)

        # Initial draw
        self.fig.canvas.draw()

    def update(
            self,
            iteration,
            mean,
            cov,
            sigma,
            fitness,
            delay=0.001):
        """
        Update the plot with current iteration data.

        Parameters
        ----------
        ...
        """

        #
        mu_2d = mean[:2]

        #
        if self.mean_dot: self.mean_dot.remove()
        if self.cov_line: self.cov_line.remove()

        # Distribution geometry
        vals, vecs = eigh(cov[:2, :2])
        t = linspace(0, 2 * pi, 100)
        circle = stack([cos(t), sin(t)])
        scaling = sigma * sqrt(maximum(vals, 1e-12))
        ellipse_2d = (vecs @ (scaling[:, None] * circle))

        # Plot search distribution and mean
        self.cov_line, = self.ax.plot(
            ellipse_2d[0, :] + mu_2d[0], ellipse_2d[1, :] + mu_2d[1],
            color='white', lw=1.8, zorder=10)

        #
        self.mean_dot = self.ax.scatter(
            mu_2d[0], mu_2d[1], color='white', marker='*', s=100,
            edgecolors='white', zorder=11)

        # UPDATE HEADER using suptitle for guaranteed visibility
        self.fig.suptitle(
            f"Generation {iteration:d} | $f$ = {fitness:.2e} | "
            f"$\sigma$ = {sigma:.2e} ", fontsize=12, fontweight='normal',
            y=0.9)

        #
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

        #
        plt.pause(delay)
