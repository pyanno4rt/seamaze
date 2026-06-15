"""2D interactive visualizer."""

# Authors: Tim Ortkamp, Chinmay Patwardhan, Pia Stammer

# %% External package import

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from numpy import (
    apply_along_axis, arange, argmin, argsort, cos, linspace, maximum,
    meshgrid, pi, sin, sqrt, stack, unravel_index, zeros)
from scipy.linalg import eigh

def enable_interactive_backend():
    """Select the interactive matplotlib backend"""
    for backend in ['TkAgg', 'Qt5Agg', 'MacOSX']:
        try:
            matplotlib.use(backend)
            return True
        except ImportError:
            continue
    return False

enable_interactive_backend()
plt.ion()

# %% Class definition


class Visualizer2D:
    """
    2D interactive visualizer class.

    This class implements a side-by-side interactive visualization: a 2D
    contour plot of the objective function landscape (with population and
    search distribution ellipse) and additional analysis plots (e.g., showing
    the dynamic singular value spectrum).

    Parameters
    ----------
    objective : Callable[[ndarray], float]
        The objective function to be minimized. Must accept a 1D ``ndarray``
        and return a scalar ``float``.

    bounds : list
        Boundaries for the 2D contour plot.

    dimensions : int
        Dimensionality of the search space.

    pop_size : int
        Size of the sample population.
    """

    def __init__(
            self,
            objective,
            bounds,
            dimensions,
            pop_size):

        # Configure the global matplotlib parameters
        plt.rcParams.update({
            "text.usetex": False,
            "font.family": 'sans-serif',
            "font.serif": [
                'Helvetica', 'Arial', 'DejaVu Sans', 'Liberation Sans'],
            "axes.labelsize": 9,
            "font.size": 10,
            "legend.fontsize": 8,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.labelweight": 'normal',
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.grid": False,
            "figure.dpi": 100
            })

        # Get the arguments
        self.objective = objective
        self.bounds = bounds
        self.dimensions = dimensions
        self.pop_size = pop_size

        # Get the index sets
        self.indices_dim = arange(1, self.dimensions + 1, 1)
        self.indices_pop = arange(1, self.pop_size + 1, 1)

        # Get the plotting step size for the ticks
        step_size = max(1, self.dimensions // 5)

        # -------------------------------------------------
        # GENERAL FIGURE SETUP ----------------------------
        # -------------------------------------------------

        # Initialize the figure
        self.fig = plt.figure(figsize=(12, 7.5))

        # Adjust the subplots
        self.fig.subplots_adjust(
            top=0.78, bottom=0.08, left=0.08, right=0.95, wspace=0.3,
            hspace=0.7
            )

        # Define a 3x2 grid layout
        plot_grid = self.fig.add_gridspec(
            3, 2, width_ratios=[2.5, 1], height_ratios=[1.0, 1.0, 1.0]
            )

        # Assign the subplots
        self.ax_2d = self.fig.add_subplot(plot_grid[0:2, 0])
        self.ax_coord = self.fig.add_subplot(plot_grid[2, 0])
        self.ax_fit = self.fig.add_subplot(plot_grid[0, 1])
        self.ax_err = self.fig.add_subplot(plot_grid[1, 1])
        self.ax_svs = self.fig.add_subplot(plot_grid[2, 1])

        # Get the plotting bounds
        x_min, x_max = self.bounds[0][0], self.bounds[1][0]
        y_min, y_max = self.bounds[0][1], self.bounds[1][1]

        # -------------------------------------------------
        # 2D LANDSCAPE PLOT INITIALIZATION ----------------
        # -------------------------------------------------

        # Set the number of grid points
        grid_points = 250

        # Create a meshgrid
        grid_x, grid_y = meshgrid(
            linspace(x_min, x_max, grid_points),
            linspace(y_min, y_max, grid_points)
            )

        # Initialize a 3D tensor for high-dimensional search spaces
        grid_3d = zeros((grid_points, grid_points, dimensions))
        grid_3d[..., 0] = grid_x
        grid_3d[..., 1] = grid_y

        # Calculate and flatten the fitness values
        flat_z = apply_along_axis(
            self.objective, 1, grid_3d.reshape(-1, dimensions)
            )

        # Reshape the fitness values into a 2D grid
        grid_z = flat_z.reshape(grid_points, grid_points)

        # Compute the contour landscape and overlay subtle white lines
        self.contour = self.ax_2d.contourf(
            grid_x, grid_y, grid_z, levels=50, cmap='viridis', alpha=0.7)
        self.ax_2d.contour(
            grid_x, grid_y, grid_z, levels=20, colors='white', alpha=0.15,
            linewidths=0.5
            )

        # Plot the minimum point on the grid
        best_idx = unravel_index(argmin(grid_z), grid_z.shape)
        self.ax_2d.scatter(
            grid_x[best_idx], grid_y[best_idx], color='orange', marker='*',
            s=100, linewidths=2, label='Global minimum (grid)', zorder=12
            )

        # Configure the colorbar
        colorbar = self.fig.colorbar(
            self.contour, ax=self.ax_2d, location='bottom',
            orientation='horizontal', shrink=0.7, pad=0.16, aspect=40
            )
        colorbar.set_label(r'$f$', rotation=0, labelpad=5, fontsize=8)
        colorbar.ax.tick_params(labelsize=8)

        # Set plot title, axis labels, limits, and legend
        self.ax_2d.set_title(
            r'Objective space ($x_1$, $x_2$)', fontsize=10, loc='center', pad=6
            )
        self.ax_2d.set_xlabel(r'$x_1$')
        self.ax_2d.set_ylabel(r'$x_2$')
        self.ax_2d.set_xlim(x_min, x_max)
        self.ax_2d.set_ylim(y_min, y_max)

        # Initialize the mean/population dots and the covariance line
        self.mean_dot = self.ax_2d.scatter(
            [], [], color='white', marker='*', s=100, edgecolors='black',
            zorder=11, label='Mean'
            )
        self.cov_line, = self.ax_2d.plot(
            [], [], color='white', lw=1.8, zorder=10, label='Covariance'
            )
        self.pop_dots = self.ax_2d.scatter(
            [], [], color='cyan', alpha=0.6, s=15, zorder=5,
            label='Population'
            )

        # Set the legend
        self.ax_2d.legend(loc='upper right', fontsize=8, framealpha=0.8)

        # -------------------------------------------------
        # PARALLEL COORDINATES PLOT INITIALIZATION --------
        # -------------------------------------------------

        # Configure the parallel coordinates subplot
        self.ax_coord.set_title('Parallel coordinates', fontsize=10, pad=6)
        self.ax_coord.set_xlabel('Dimension')
        self.ax_coord.set_ylabel('Value')
        self.ax_coord.set_xlim(
            1 - 0.02 * self.dimensions, 1.02 * self.dimensions
            )
        self.ax_coord.grid(True, linestyle=':', alpha=0.6)

        # Place evenly-spaced x-ticks
        self.ax_coord.set_xticks(arange(1, self.dimensions + 1, step_size))

        # Initialize the mean line and the population line collection
        self.coord_mean_line, = self.ax_coord.plot(
            [], [], color='darkblue', lw=2.2, zorder=10, label='Mean'
            )
        self.coord_collection = LineCollection(
            [], color='cyan', alpha=0.3, lw=0.8, zorder=4
            )

        # Label the population line collection and add it to the axis
        self.coord_collection.set_label('Population')
        self.ax_coord.add_collection(self.coord_collection)

        # Set the legend
        self.ax_coord.legend(loc='upper right', fontsize=8, framealpha=0.8)

        # -------------------------------------------------
        # FITNESS PLOT INITIALIZATION ---------------------
        # -------------------------------------------------

        # Configure the fitness subplot
        self.ax_fit.set_title('Fitness', fontsize=10, pad=6)
        self.ax_fit.set_xlabel('Individual')
        self.ax_fit.set_ylabel('Value')
        self.ax_fit.set_xlim(
            1 - 0.02 * self.pop_size, 1.02 * self.pop_size
            )
        self.ax_fit.grid(True, linestyle=':', alpha=0.6)

        # Place evenly-spaced ticks
        self.ax_fit.set_xticks(arange(1, self.pop_size + 1, step_size))

        # Initialize the fitness vector line
        self.fit_vector_line, = self.ax_fit.plot(
            [], [], color='red', lw=1.8, zorder=10
            )

        # -------------------------------------------------
        # SQUARED BOUND ERRORS PLOT INITIALIZATION --------
        # -------------------------------------------------

        # Configure the squared bound error subplot
        self.ax_err.set_title('Squared bound error', fontsize=10, pad=6)
        self.ax_err.set_xlabel('Individual')
        self.ax_err.set_ylabel('Value')
        self.ax_err.set_xlim(
            1 - 0.02 * self.pop_size, 1.02 * self.pop_size
            )
        self.ax_err.grid(True, linestyle=':', alpha=0.6)

        # Place evenly-spaced ticks
        self.ax_err.set_xticks(arange(1, self.pop_size + 1, step_size))

        # Initialize the error vector line
        self.err_vector_line, = self.ax_err.plot(
            [], [], color='orange', lw=1.8, zorder=10
            )

        # -------------------------------------------------
        # SINGULAR VALUES PLOT INITIALIZATION -------------
        # -------------------------------------------------

        # Configure the singular values subplot
        self.ax_svs.set_title('Singular values', fontsize=10, pad=6)
        self.ax_svs.set_xlabel('Dimension')
        self.ax_svs.set_ylabel('Value')
        self.ax_svs.set_xlim(
            1 - 0.02 * self.dimensions, 1.02 * self.dimensions
            )
        self.ax_svs.grid(True, linestyle=':', alpha=0.6)

        # Place evenly-spaced ticks
        self.ax_svs.set_xticks(arange(1, self.dimensions + 1, step_size))

        # Initialize the singular value vector line
        self.svs_vector_line, = self.ax_svs.plot(
            [], [], color='green', lw=1.8, zorder=10
            )

        # -------------------------------------------------
        # TITLE/SUBTITLE AND INITIAL DRAWING -----
        # -------------------------------------------------

        # Add the main title
        self.fig.suptitle(
            'Visualizer2D', fontsize=16, fontweight='bold', y=0.96
            )

        # Add the subtitle
        self.suptitle_text = self.fig.text(
            0.5, 0.90, "", fontsize=11, fontweight='normal', ha='center',
            va='center'
            )

        # Draw the canvas initially
        plt.show(block=False)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def update(
            self,
            iteration,
            population,
            mean,
            cov,
            sigma,
            fitness,
            squared_bound_errors,
            best_fitness,
            delay=0.001):
        """
        Update the plot with current iteration data.

        Parameters
        ----------
        iteration : int
            Index of the current generation.

        population : ndarray
            Array with the current generation.

        mean : ndarray
            Mean vector of the search distribution.

        cov : ndarray
            Covariance matrix of the search distribution.

        sigma : float
            Step-size parameter.

        fitness : ndarray
            Fitness values of the current generation.

        squared_bound_errors : ndarray
            Squared errors (bound violations) of the current generation.

        best_fitness : float
            Fitness value (best) of the current generation.

        delay : float, default=0.001
            Delay parameter (in seconds) to control update speed.
        """

        # -------------------------------------------------
        # 2D LANDSCAPE PLOT UPDATE ------------------------
        # -------------------------------------------------

        # Extract mean subvector and covariance submatrix
        mu_2d = mean[:2]
        cov_2d = cov[:2, :2]

        # Perform eigenvalue decomposition on the covariance submatrix
        evs, evecs = eigh(cov_2d)

        # Calculate the 2D ellipse
        angles = linspace(0, 2 * pi, 100)
        scaling = sigma * sqrt(maximum(evs, 1e-12))
        ellipse_2d = evecs @ (
            scaling[:, None] * stack([cos(angles), sin(angles)])
            )

        # Update the mean/population dots and the covariance line
        self.pop_dots.set_offsets(population[:, :2])
        self.mean_dot.set_offsets(mu_2d)
        self.cov_line.set_data(
            ellipse_2d[0, :] + mu_2d[0], ellipse_2d[1, :] + mu_2d[1]
            )

        # -------------------------------------------------
        # PARALLEL COORDINATES PLOT UPDATE ----------------
        # -------------------------------------------------

        # Expand the dimension indices
        indices_dim_exp = self.indices_dim[None, :]

        # Stack dimension indices and individuals
        segments = stack(
            [indices_dim_exp.repeat(len(population), axis=0), population],
            axis=-1
            )

        # Update the mean line and the population line collection
        self.coord_mean_line.set_data(self.indices_dim, mean)
        self.coord_collection.set_segments(segments)

        # Compute maximum, minimum and range of the population values
        coord_max = max(population.max(), mean.max())
        coord_min = min(population.min(), mean.min())
        coord_range = coord_max - coord_min

        # Check if the range is small
        if abs(coord_range) < 1e-9:

            # Set range-independent limits
            self.ax_coord.set_ylim(coord_min - 0.1, coord_max + 0.1)

        else:

            # Set range-dependent limits
            self.ax_coord.set_ylim(
                coord_min - 0.05 * coord_range, coord_max + 0.05 * coord_range
                )

        # -------------------------------------------------
        # FITNESS PLOT UPDATE -----------------------------
        # -------------------------------------------------

        # Sort the fitness values
        sort_indices = argsort(fitness)
        fitness = fitness[sort_indices]

        # Update the fitness vector line
        self.fit_vector_line.set_data(self.indices_pop, fitness)

        # Compute maximum, minimum and range of the fitness values
        fit_max = fitness.max()
        fit_min = fitness.min()
        fit_range = fit_max - fit_min

        # Check if the range is small
        if abs(fit_range) < 1e-9:

            # Set range-independent limits
            self.ax_fit.set_ylim(fit_min - 0.1, fit_max + 0.1)

        else:

            # Set range-dependent limits
            self.ax_fit.set_ylim(
                fit_min - 0.05 * fit_range, fit_max + 0.05 * fit_range
                )

        # -------------------------------------------------
        # SQUARED BOUND ERRORS PLOT INITIALIZATION --------
        # -------------------------------------------------

        # Sort the errors by the fitness values
        squared_bound_errors = squared_bound_errors[sort_indices]

        # Update the error vector line
        self.err_vector_line.set_data(self.indices_pop, squared_bound_errors)

        # Compute maximum, minimum and range of the errors
        err_max = squared_bound_errors.max()
        err_min = squared_bound_errors.min()
        err_range = err_max - err_min

        # Check if the range is small
        if abs(err_range) < 1e-9:

            # Set range-independent limits
            self.ax_err.set_ylim(err_min - 0.1, err_max + 0.1)

        else:

            # Set range-dependent limits
            self.ax_err.set_ylim(
                err_min - 0.05 * err_range, err_max + 0.05 * err_range
                )

        # -------------------------------------------------
        # SINGULAR VALUES PLOT INITIALIZATION -------------
        # -------------------------------------------------

        # Calculate the full set of singular values
        full_evs, _ = eigh(cov)
        svs = sorted(full_evs, reverse=True)

        # Update the singular value vector line
        self.svs_vector_line.set_data(self.indices_dim, svs)

        # Compute maximum, minimum and range of the singular values
        svs_max = max(svs)
        svs_min = min(svs)
        svs_range = svs_max - svs_min

        # Check if the range is small
        if abs(svs_range) < 1e-9:

            # Set range-independent limits
            self.ax_svs.set_ylim(svs_min - 0.1, svs_max + 0.1)

        else:

            # Set range-dependent limits
            self.ax_svs.set_ylim(
                svs_min - 0.05 * svs_range, svs_max + 0.05 * svs_range
                )

        # -------------------------------------------------
        # SUBTITLE UPDATE AND REFRESH ---------------------
        # -------------------------------------------------

        # Update the subtitle
        self.suptitle_text.set_text(
            f"Generation {iteration:d}  |  f = {best_fitness:.2e}  |  "
            fr"$\sigma$ = {sigma:.2e}"
            )

        # Refresh the canvas
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        # Pause the update
        self.fig.canvas.start_event_loop(delay)
