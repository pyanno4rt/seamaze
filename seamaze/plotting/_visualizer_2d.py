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
            grid_points=300):

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
        self.fig, (self.ax, self.ax_hist) = plt.subplots(
            1, 2, figsize=(14, 8), gridspec_kw={'width_ratios': [2.5, 1]}
            )

        # Manual adjustment: reserve 15% space at the top for the header
        self.fig.subplots_adjust(
            top=0.83, bottom=0.1, left=0.08, right=0.95, wspace=0.25
            )

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
        self.ax_hist.set_title('Singular value spectrum', fontsize=10, pad=10)
        self.ax_hist.set_xlabel('Index')
        self.ax_hist.set_ylabel('Value')
        self.ax_hist.grid(True, linestyle='--', alpha=0.5)

        #
        self.ax_hist.set_xlim(-0.5, dimensions - 0.5)

        #
        self.pop_dots = None
        self.mean_dot = None
        self.cov_line = None
        self.spec_dots = None

        #
        self.ax.set_xlim(self.x_min, self.x_max)
        self.ax.set_ylim(self.y_min, self.y_max)
        self.ax.legend(loc='upper right', fontsize=8, framealpha=0.6)

        # Initial draw
        self.fig.canvas.draw()

    def update(
            self,
            iteration,
            population,
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
        cov_2d = cov[:2, :2]
        vals, vecs = eigh(cov_2d)

        #
        t = linspace(0, 2 * pi, 100)
        circle = stack([cos(t), sin(t)])
        scaling = sigma * sqrt(maximum(vals, 1e-12))
        ellipse_2d = vecs @ (scaling[:, None] * circle)

        #
        global_eigenvalues, _ = eigh(cov)
        singular_values = sigma * sqrt(maximum(global_eigenvalues, 1e-12))
        singular_values_sorted = sorted(singular_values, reverse=True)
        indices = linspace(0, self.dimensions - 1, self.dimensions)

        #
        if self.pop_dots: self.pop_dots.remove()
        if self.mean_dot: self.mean_dot.remove()
        if self.cov_line: self.cov_line.remove()
        if self.spec_dots: self.spec_dots.remove()

        #
        self.pop_dots = self.ax.scatter(
            population[:, 0], population[:, 1],
            color='cyan', alpha=0.6, s=15, zorder=5)

        self.cov_line, = self.ax.plot(
            ellipse_2d[0, :] + mu_2d[0], ellipse_2d[1, :] + mu_2d[1],
            color='white', lw=1.8, zorder=10)

        self.mean_dot = self.ax.scatter(
            mu_2d[0], mu_2d[1], color='white', marker='*', s=100,
            edgecolors='black', zorder=11)

        #
        self.ax.set_title(
            f"Objective space (x1, x2)\n"
            f"Generation {iteration:d} | $f$ = {fitness:.2e} | "
            f"$\sigma$ = {sigma:.2e} ", fontsize=10, fontweight='normal',
            pad=10, loc='center'
            )

        #
        self.spec_dots = self.ax_hist.scatter(
            indices, singular_values_sorted, color='crimson',
            edgecolors='black', alpha=0.8, s=25, zorder=5
            )

        #
        max_val = max(singular_values_sorted)
        min_val = min(singular_values_sorted)
        self.ax_hist.set_ylim(0.99 * min_val, max_val * 1.01)

        #
        x_offset = 4.0
        self.ax_hist.set_xlim(-x_offset, (self.dimensions - 1) + x_offset)

        #
        self.fig.suptitle(
            "Visualizer2D", fontsize=16, fontweight='bold'
            )

        #
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(delay)
