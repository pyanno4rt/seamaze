"""Dynamic visualizer."""

# Authors: Tim Ortkamp, Chinmay Patwardhan, Pia Stammer

# %% External package import

from time import time

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import (
    BoundaryNorm, LinearSegmentedColormap, LogNorm, Normalize
    )
from matplotlib.legend_handler import HandlerTuple
from matplotlib.patches import Patch
from matplotlib.ticker import (
    FixedFormatter, FixedLocator, NullLocator, ScalarFormatter
    )
from matplotlib.transforms import ScaledTranslation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numpy import (
    arange, argmin, argsort, array, ceil, clip, cos, cumsum, digitize, divide,
    float64, floor, full, full_like, hstack, inf, isnan, linspace, logspace,
    log1p, log10, maximum, meshgrid, minimum, nan, nanmax, nanmin, percentile,
    pi, sin, sort, sqrt, stack, unique, where, zeros, zeros_like
    )
from numpy import all as nall
from numpy import mean as nmean
from scipy.linalg import eigh
from scipy.ndimage import gaussian_filter

matplotlib.rcParams['path.simplify'] = True
matplotlib.rcParams['path.simplify_threshold'] = 1.0
matplotlib.rcParams['agg.path.chunksize'] = 10000

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


class Visualizer:
    """
    Dynamic visualizer class.

    This class implements a side-by-side dynamic visualization: a 2D
    reconstruction of the objective function landscape (with population and
    search distribution ellipse) and additional analysis plots (coordinate
    space, fitness history, feasibility spectrum, and covariance rank).

    Parameters
    ----------
    bounds : list
        Boundaries for the landscape plot.

    dimensions : int
        Dimensionality of the search space.

    pop_size : int
        Size of the sample population.
    """

    def __init__(
            self,
            bounds,
            dimensions,
            pop_size):

        # Configure the global matplotlib parameters
        plt.rcParams.update({
            "axes.grid": False,
            "axes.labelsize": 9,
            "axes.labelweight": 'normal',
            "figure.dpi": 100,
            "font.family": 'sans-serif',
            "font.serif": [
                'Helvetica', 'Arial', 'DejaVu Sans', 'Liberation Sans'
                ],
            "font.size": 10,
            "legend.fontsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "text.usetex": False,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9
            })

        # Get the arguments
        self.dimensions = dimensions
        self.pop_size = pop_size

        # Specify the index set for the dimensions
        self.indices_dim = arange(1, self.dimensions + 1, 1)

        # Check if the space is low-dimensional
        if self.dimensions <= 6:

            # Generate dense ticks
            custom_dim_ticks = list(range(1, self.dimensions + 1))

        else:

            # Get the approximate step size for 5 equidistant ticks
            approx_step = self.dimensions // 5

            # Determine the base magnitude for rounding
            magnitude = 10 ** max(0, len(str(approx_step)) - 2)

            # Round the step size down
            step = max(1, (approx_step // magnitude) * magnitude)

            # Generate intermediate tick marks
            middle_ticks = list(range(step, self.dimensions, step))

            # Check if the largest middle tick crowds the upper bound
            if (middle_ticks and
                (self.dimensions - middle_ticks[-1]) < (step * 0.3)):

                # Drop the largest middle tick
                middle_ticks.pop()

            # Get the custom ticks by including explicit bound ticks
            custom_dim_ticks = [1] + middle_ticks + [self.dimensions]

        # Calculate the marker size depending on the dimensionality
        marker_size_dim = (
            0.0 if self.dimensions > 150
            else float(clip(15.0 / log1p(self.dimensions), 1.0, 5.0))
            )

        # =====================================================================
        # FIGURE SETUP
        # =====================================================================

        # Create the figure
        self.fig = plt.figure(figsize=(12, 7.5))

        # Adjust the subplot spacings
        self.fig.subplots_adjust(
            top=0.78, bottom=0.08, left=0.07, right=0.92, wspace=0.5,
            hspace=0.7
            )

        # Construct a 3x2 plot grid
        plot_grid = self.fig.add_gridspec(
            3, 2, width_ratios=[2.2, 1], height_ratios=[1.0, 1.0, 1.0]
            )

        # Add the subplots to the grid
        self.ax_rec = self.fig.add_subplot(plot_grid[0:2, 0])
        self.ax_coord = self.fig.add_subplot(plot_grid[2, 0])
        self.ax_fit = self.fig.add_subplot(plot_grid[0, 1])
        self.ax_err = self.fig.add_subplot(plot_grid[1, 1])
        self.ax_svs = self.fig.add_subplot(plot_grid[2, 1])

        # =====================================================================
        # RECONSTRUCTED OBJECTIVE SPACE
        # =====================================================================

        # Pre-allocate buffers for tracking histories
        self.max_size = 100000
        self._hist_x = zeros(self.max_size, dtype=float64)
        self._hist_y = zeros(self.max_size, dtype=float64)
        self._hist_f = zeros(self.max_size, dtype=float64)

        # Initialize the history size counter
        self._hist_count = 0

        # Determine the bounds for the landscape plot
        bounds = bounds if bounds is not None else [(-1, -1), (1, 1)]

        # Get the plot limits
        x_min, x_max = bounds[0][0], bounds[1][0]
        y_min, y_max = bounds[0][1], bounds[1][1]

        # Set the number of points in each dimension
        points = 100

        # Create evenly spaced node points
        self.x_nodes = linspace(x_min, x_max, points + 1)
        self.y_nodes = linspace(y_min, y_max, points + 1)

        # Compute the center points for interpolation
        x_centers = 0.5 * (self.x_nodes[:-1] + self.x_nodes[1:])
        y_centers = 0.5 * (self.y_nodes[:-1] + self.y_nodes[1:])

        # Build coordinate matrices from the center points
        self.grid_x, self.grid_y = meshgrid(x_centers, y_centers)

        # Initialize the background grid
        self.grid_memory = full((points, points), nan, dtype=float64)

        # Precompute the baseline values for ellipse generation
        angles = linspace(0, 2 * pi, 100)
        self._ellipse_base = stack([cos(angles), sin(angles)])

        # Initialize the contours
        self.contours = None

        # Initialize the mappable for the colorbar
        initial_mappable_rec = ScalarMappable(
            norm=Normalize(vmin=0, vmax=1), cmap='coolwarm'
            )

        # Initialize the colorbar
        self.rec_colorbar = self.fig.colorbar(
            initial_mappable_rec, ax=self.ax_rec, location='bottom',
            orientation='horizontal', shrink=0.5, pad=0.16, aspect=50
            )

        # Configure the colorbar formatter
        formatter = ScalarFormatter(useMathText=False)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-3, 3))
        formatter.set_useOffset(False)

        # Configure the colorbar
        self.rec_colorbar.ax.xaxis.set_major_formatter(formatter)
        self.rec_colorbar.set_label(
            r'$f$', rotation=0, labelpad=5, fontsize=7
            )
        self.rec_colorbar.ax.tick_params(labelsize=8)

        # Initialize the dynamic scatter points and lines
        self.best_x_star = self.ax_rec.scatter(
            [], [], color='#FFD700', marker='*', s=180, linewidths=0.5,
            edgecolors='#1A1A1A', zorder=14, label='Best'
            )
        self.mean_dot = self.ax_rec.scatter(
            [], [], color='#FFFFFF', marker='*', s=160, linewidths=0.5,
            edgecolors='#1A1A1A', zorder=13, label='Mean'
            )
        self.cov_line_outer, = self.ax_rec.plot(
            [], [], color='#1A1A1A', lw=2.0, zorder=11, label='Covariance'
            )
        self.cov_line_inner, = self.ax_rec.plot(
            [], [], color='#FFFFFF', lw=1.0, zorder=11, label='_nolegend_'
            )
        self.pop_dots = self.ax_rec.scatter(
            [], [], color='#FF9F1C', s=30, linewidths=0.4, alpha=0.8,
            edgecolors='#1A1A1A', zorder=5, label='Population'
            )

        # Initialize and configure the legend
        rec_legend = self.ax_rec.legend(
            handles=[
                self.best_x_star, self.mean_dot,
                (self.cov_line_outer, self.cov_line_inner), self.pop_dots
                ],
            labels=['Best', 'Mean', 'Covariance', 'Population'],
            handler_map={tuple: HandlerTuple(ndivide=None, pad=0.0)},
            loc='upper left', bbox_to_anchor=(1.01, 1.0), ncol=1, fontsize=7,
            framealpha=1.0
            )
        rec_legend.set_zorder(100)

        # Set title, axis labels, and limits
        self.ax_rec.set_title(
            r'Reconstructed Objective Space ($x_1$, $x_2$)', fontsize=10,
            loc='center', pad=6
            )
        self.ax_rec.set_xlabel(r'$x_1$')
        self.ax_rec.set_ylabel(r'$x_2$')
        self.ax_rec.set_xlim(x_min, x_max)
        self.ax_rec.set_ylim(y_min, y_max)

        # Allow layout stretching
        self.ax_rec.set_aspect('auto')

        # =====================================================================
        # COORDINATE SPACE
        # =====================================================================

        # Initialize the coordinate mean line
        self.coord_mean_line, = self.ax_coord.plot(
            [], [], color='#03045E', lw=1.0, marker='o',
            markersize=marker_size_dim, zorder=10, label='Mean'
            )

        # Initialize the coordinate sigma-band
        self.coord_fill_band = None

        # Initialize and configure the legend
        coord_legend = self.ax_coord.legend(
            handles=[self.coord_mean_line, Patch(color='#4EA8DE', alpha=0.6)],
            labels=['Mean', r'$\sigma$-band'],
            loc='upper left', bbox_to_anchor=(1.01, 1.0), ncol=1, fontsize=7,
            framealpha=1.0
            )
        coord_legend.set_zorder(100)

        # Set title, axis labels, limits, and ticks
        self.ax_coord.set_title('Coordinate Space', fontsize=10, pad=6)
        self.ax_coord.set_xlabel('Dimension')
        self.ax_coord.set_ylabel('Value')
        self.ax_coord.set_xlim(
            1 - 0.02 * self.dimensions, 1.02 * self.dimensions
            )
        self.ax_coord.grid(True, linestyle=':', alpha=0.6)
        self.ax_coord.set_xticks(custom_dim_ticks)

        # =====================================================================
        # FITNESS HISTORY
        # =====================================================================

        # Initialize histories for statistical measures
        self._hist_gens = []
        self._hist_min = []
        self._hist_q25 = []
        self._hist_mean = []
        self._hist_q75 = []
        self._hist_max = []

        # Initialize the plot lines (max, mean, min)
        self.fit_max_line, = self.ax_fit.plot(
            [], [], color='#ADB5BD', linestyle='-', lw=0.5, alpha=0.5, zorder=3
            )
        self.fit_mean_line, = self.ax_fit.plot(
            [], [], color='#495057', linestyle='-', lw=1.0, zorder=5
            )
        self.fit_min_line, = self.ax_fit.plot(
            [], [], color='red', linestyle='-', lw=2.0, zorder=10
            )

        # Initialize the IQR and range bands
        self.fit_inner_band = None
        self.fit_outer_band = None

        # Create a layout buffer for right-side alignment
        plot_divider_fit = make_axes_locatable(self.ax_fit)
        right_buffer_fit = plot_divider_fit.append_axes(
            "right", size="4%", pad=0.04
            )
        right_buffer_fit.axis('off')

        # Initialize and configure the legend
        fit_legend = self.ax_fit.legend(
            handles=[
                self.fit_min_line, self.fit_mean_line,
                Patch(color='red', alpha=0.35), Patch(color='red', alpha=0.22)
                ],
            labels=['Best', 'Mean', 'IQR', 'Range'],
            loc='upper left', bbox_to_anchor=(1.0, 1.0), ncol=1, fontsize=7,
            framealpha=1.0
            )
        fit_legend.set_zorder(100)

        # Set title, axis labels, and grid
        self.ax_fit.set_title('Fitness History', fontsize=10, pad=6)
        self.ax_fit.set_xlabel('Generation')
        self.ax_fit.set_ylabel('Fitness')
        self.ax_fit.grid(True, linestyle=':', alpha=0.4)

        # =====================================================================
        # FEASIBILITY SPECTRUM
        # =====================================================================

        # Initialize the dynamic spectral matrix
        self._err_spec_matrix = full((self.pop_size, 0), nan, dtype=float64)

        # Initialize the mesh handle
        self.err_density_mesh = None

        # Set the log limits and boundary steps for errors
        self._err_log_floor = -10.0
        self._err_log_ceil = 5.0
        err_boundaries = arange(self._err_log_floor, self._err_log_ceil + 1, 1)

        # Set the colorbar ticks
        self._cbar_err_ticks = arange(
            self._err_log_floor, self._err_log_ceil + 1, 3
            )

        # Define the colors
        high_contrast_colors = [
            "#FFF0F5", "#FAD2E1", "#F77F00", "#D90429", "#3F000B"
            ]

        # Construct the custom colormap and and hide zero errors as white
        self._err_cmap = LinearSegmentedColormap.from_list(
            "infeasibility_fade", high_contrast_colors
            )
        self._err_cmap.set_bad("#FFFFFF")

        # Create a layout buffer for right-side alignment
        plot_divider_err = make_axes_locatable(self.ax_err)
        colorbar_axis_err = plot_divider_err.append_axes(
            "right", size="2%", pad=0.04
            )

        # Discretize log values into color increments
        self._err_norm = BoundaryNorm(
            boundaries=err_boundaries, ncolors=self._err_cmap.N, clip=True
            )

        # Initialize the mappable for the colorbar
        initial_err_mappable = ScalarMappable(
            norm=self._err_norm, cmap=self._err_cmap
            )

        # Configure the colorbar
        self.err_density_cbar = self.fig.colorbar(
            initial_err_mappable, cax=colorbar_axis_err,
            orientation='vertical', ticks=self._cbar_err_ticks
            )
        self.err_density_cbar.set_label('Violation', fontsize=7, labelpad=5)
        self.err_density_cbar.ax.tick_params(labelsize=7)

        # Initialize the interior text field
        self.err_status_text = self.ax_err.text(
            0.97, 0.93, "", color='#38B000', fontsize=8, fontweight='bold',
            ha='right', va='top', transform=self.ax_err.transAxes, zorder=100,
            bbox={
                'facecolor': '#FFFFFF', 'edgecolor': '#38B000', 'alpha': 0.95,
                'boxstyle': 'round,pad=0.4', 'lw': 1.0}
            )
        self.err_status_text.set_visible(False)

        # Set title, axis labels, limits and ticks
        self.ax_err.set_title('Feasibility Spectrum', fontsize=10, pad=6)
        self.ax_err.set_xlabel('Generation')
        self.ax_err.set_ylabel('Individual')
        self.ax_err.set_ylim(0.5, self.pop_size + 0.5)
        self.ax_err.set_yticks([1, self.pop_size])

        # =====================================================================
        # COVARIANCE RANK
        # =====================================================================

        # Initialize histories for statistical measures
        self._svs_hist_gens = []
        self._svs_hist_80 = []
        self._svs_hist_95 = []
        self._svs_hist_99 = []

        # Initialize the level line plots
        self.svs_80_line, = self.ax_svs.plot(
            [], [], color='#4A5568', linestyle='-', lw=1.0, zorder=8
            )
        self.svs_95_line, = self.ax_svs.plot(
            [], [], color='#D90429', linestyle='-', lw=1.0, zorder=10
            )
        self.svs_99_line, = self.ax_svs.plot(
            [], [], color='#2A9D8F', linestyle='-', lw=1.0, zorder=7
            )

        # Initialize IQR and range bands
        self.svs_inner_band = None
        self.svs_outer_band = None

        # Create a layout buffer for right-side alignment
        plot_divider_svs = make_axes_locatable(self.ax_svs)
        right_buffer_svs = plot_divider_svs.append_axes(
            "right", size="4%", pad=0.04
            )
        right_buffer_svs.axis('off')

        # Initialize and configure the legend
        self.svs_legend = self.ax_svs.legend(
            handles=[self.svs_99_line, self.svs_95_line, self.svs_80_line],
            labels=['99%', '95%', '80%'],
            loc='upper left', bbox_to_anchor=(1.0, 1.0), ncol=1, fontsize=7,
            framealpha=1.0
            )
        self.svs_legend.set_zorder(100)

        # Initialize the interior text fields
        self.svs_val_80 = self.ax_svs.text(
            0.98, 0.79, "", color='#4A5568', fontsize=7, fontweight='bold',
            ha='right', va='top', transform=self.ax_svs.transAxes, zorder=26
            )
        self.svs_val_95 = self.ax_svs.text(
            0.98, 0.88, "", color='#D90429', fontsize=7, fontweight='bold',
            ha='right', va='top', transform=self.ax_svs.transAxes, zorder=26
            )
        self.svs_val_99 = self.ax_svs.text(
            0.98, 0.97, "", color='#2A9D8F', fontsize=7, fontweight='bold',
            ha='right', va='top', transform=self.ax_svs.transAxes, zorder=26
            )

        # Set title, axis labels, and grid
        self.ax_svs.set_title('Covariance Rank', fontsize=10, pad=6)
        self.ax_svs.set_xlabel('Generation')
        self.ax_svs.set_ylabel('Effective Rank')
        self.ax_svs.grid(True, linestyle=':', alpha=0.4)

        # =====================================================================
        # SUBTITLE AND INITIAL DRAWING
        # =====================================================================

        # Set the global title
        self.fig.suptitle('Visualizer', fontsize=16, fontweight='bold', y=0.96)

        # Initialize the dynamic subtitle
        self.suptitle_text = self.fig.text(
            0.5, 0.90, "", fontsize=11, fontweight='normal', ha='center',
            va='center'
            )

        # Open the figure and render the initial layout
        plt.show(block=False)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def update(
            self,
            iteration,
            population,
            mean,
            cov,
            svs,
            sigma,
            fitness,
            squared_bound_errors,
            optimal_value,
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

        svs : ndarray
            Singular values of the covariance matrix.

        sigma : float
            Step-size parameter.

        fitness : ndarray
            Fitness values of the current generation.

        squared_bound_errors : ndarray
            Squared errors (bound violations) of the current generation.

        optimal_value : float
            Global minimum fitness value up to the current generation.

        delay : float, default=0.001
            Delay parameter (in seconds) to control update speed.
        """

        # Start the update time recording
        start = time()

        # Check if the iteration number is small enough
        if iteration <= 6:

            # Generate dense ticks
            custom_x_ticks = list(range(1, iteration + 1))

        else:

            # Get the dense ticks
            raw_ticks = linspace(1, iteration, 6)

            # Determine the step size
            step = (
                5.0 if iteration <= 100 else 50.0 if iteration <=500 else 100.0
                )

            # Compute the middle ticks
            middle_ticks = clip(raw_ticks[1:-1] / step, 0, None)
            middle_ticks = (middle_ticks * step).round().astype(int).tolist()

            # Get the custom ticks by including explicit bound ticks
            custom_x_ticks = unique(
                [1] + middle_ticks + [iteration]).tolist()

        # Loop over fitness history and feasibility spectrum x-axes
        for ax in [self.ax_fit, self.ax_svs]:

            # Set the limit and ticks for the x-axis
            ax.set_xlim(1, iteration + 2)
            ax.set_xticks(custom_x_ticks)

        # =====================================================================
        # RECONSTRUCTED OBJECTIVE SPACE
        # =====================================================================

        # Get the indices to be updated
        indices = (arange(self.pop_size) + self._hist_count) % self.max_size

        # Add the new data to the histories
        self._hist_x[indices] = population[:, 0]
        self._hist_y[indices] = population[:, 1]
        self._hist_f[indices] = fitness

        # Increment the history counter
        self._hist_count += self.pop_size

        # Assign the population coordinates to discrete bins
        x_indices = digitize(population[:, 0], self.x_nodes) - 1
        y_indices = digitize(population[:, 1], self.y_nodes) - 1

        # Get a mask for the valid indices
        grid_shape = self.grid_memory.shape
        valid_mask = (
            (x_indices >= 0) & (x_indices < grid_shape[1]) &
            (y_indices >= 0) & (y_indices < grid_shape[0])
            )

        # Check if any indices are valid
        if any(valid_mask):

            # Get the current grid values (replacing nan with inf)
            current_grid_values = where(
                isnan(self.grid_memory), inf, self.grid_memory
                )

            # Overwrite grid values with the current minima
            minimum.at(
                current_grid_values,
                (y_indices[valid_mask], x_indices[valid_mask]),
                fitness[valid_mask]
                )

            # Write the grid values back to memory (replacing inf by nan)
            self.grid_memory = where(
                current_grid_values == inf, nan, current_grid_values
                )


        # Precompute a mask for the NaN values
        nan_mask = isnan(self.grid_memory)
        all_nan = nan_mask.all()

        # Get the current minimum and maximum of the grid values
        current_min = float(nanmin(self.grid_memory)) if not all_nan else 1e-5
        current_max = float(nanmax(self.grid_memory)) if not all_nan else 1e-5

        # Check if the range is very small
        if abs(current_max - current_min) < 1e-9:

            # Add a small epsilon to the maximum
            current_max += 1e-9

        # Create a local copy of the grid memory
        display_grid = self.grid_memory.copy()

        # Get the log scale indicator
        is_log = (current_max - current_min) > 10000 and current_min > 0

        # Check if the log scale should be used
        if is_log:

            # Logarithmize the display grid
            log_grid = log10(display_grid)

            # Get a mask of the valid grid points
            mask = (~isnan(log_grid)).astype(float)

            # Replace nan with zero
            log_grid[isnan(log_grid)] = 0

            # Smooth the grid and mask with a Gaussian filter
            smoothed_grid = gaussian_filter(log_grid, sigma=0.5)
            smoothed_mask = gaussian_filter(mask, sigma=0.5)

            # Initialize a NaN array
            grid_z_log = full_like(smoothed_grid, nan)

            # Normalize the logarithmic grid and revert the logarithm
            divide(
                smoothed_grid, smoothed_mask, out=grid_z_log,
                where=(smoothed_mask > 0.01)
                )
            grid_z = 10.0 ** grid_z_log

            # Define the logarithmic contour levels
            levels = logspace(log10(current_min), log10(current_max), 10)

        else:

            # Create a mask for non-NaN values
            mask = (~nan_mask).astype(float)

            # Clean the display grid
            display_grid[nan_mask] = 0

            # Smooth the grid and mask with a Gaussian filter
            smoothed_grid = gaussian_filter(display_grid, sigma=0.5)
            smoothed_mask = gaussian_filter(mask, sigma=0.5)

            # Initialize a NaN array
            grid_z = full_like(smoothed_grid, nan)

            # Normalize the smoothed linear grid
            divide(
                smoothed_grid, smoothed_mask, out=grid_z,
                where=(smoothed_mask > 0.01)
                )

            # Define the number of linear contour levels
            levels = 10

        # Check if the grid contains at least one valid value
        if not isnan(grid_z).all():

            # Set up the normalization scale (log or linear)
            norm_scale = (
                LogNorm(vmin=current_min, vmax=current_max) if is_log
                else Normalize(vmin=current_min, vmax=current_max)
                )

            # Check if contours exist
            if self.contours is not None:

                try:

                    # Loop over the contour collections
                    for collection in self.contours.collections:

                        # Remove the collection
                        collection.remove()

                except (ValueError, AttributeError):

                    pass

            # Generate the contours
            self.contours = self.ax_rec.contourf(
                self.grid_x, self.grid_y, grid_z, levels=levels,
                cmap='coolwarm', norm=norm_scale, alpha=0.7, zorder=1
                )

            # Rasterize the contours
            self.contours.set_rasterized(True)

            # Check if the log scale is used
            if is_log:

                # Calculate logarithmic bounds for the linear mapping
                log_min, log_max = log10(current_min), log10(current_max)
                linear_log_norm = Normalize(vmin=log_min, vmax=log_max)

                # Apply log normalization and set color limits
                self.rec_colorbar.mappable.set_norm(linear_log_norm)
                self.rec_colorbar.mappable.set_clim(vmin=log_min, vmax=log_max)

            else:

                # Apply linear normalization and set color limits
                self.rec_colorbar.mappable.set_norm(norm_scale)
                self.rec_colorbar.mappable.set_clim(
                    vmin=current_min, vmax=current_max
                    )

            # Refresh the colorbar
            self.rec_colorbar.update_normal(self.rec_colorbar.mappable)

            # Check if the log scale is used
            if is_log:

                # Determine the integer decade range for ticks
                start_dec = int(ceil(log_min))
                end_dec = int(floor(log_max))

                # Get the number of decades
                num_decades = end_dec - start_dec

                # Set the initial stride
                stride = 1

                # Check if the number of decades is greater than 5
                if num_decades > 5:

                    # Double the stride
                    stride = 2

                # Check if the number of decades is greater than 10
                if num_decades > 10:

                    # Triple the stride
                    stride = 3

                # Create an array of internal decades
                inner_decades = (
                    arange(start_dec, end_dec + 1) if start_dec <= end_dec
                    else array([])
                    )

                # Add the boundaries to the tick positions
                tick_positions = sorted(
                    unique(hstack(([log_min], inner_decades, [log_max])))
                    )

                # Initialize the tick labels
                tick_labels = []

                # Loop over the tick positions
                for pos in tick_positions:

                    # Check if the current position is at the minimum
                    if pos == log_min:

                        # Append the custom formatted minimum
                        tick_labels.append(f"{current_min:.1e}")

                    # Check if the current position is at the maximum
                    elif pos == log_max:

                        # Append the custom formatted maximum
                        tick_labels.append(f"{current_max:.1e}")

                    else:

                        # Check if the decade should get a label
                        if (int(pos) - start_dec) % stride == 0:

                            # Append the custom formatted inner label
                            tick_labels.append(f"$10^{{{int(pos)}}}$")

                        else:

                            # Append an empty label
                            tick_labels.append("")

                # Apply the calculated ticks to the colorbar
                self.rec_colorbar.ax.xaxis.set_major_locator(
                    FixedLocator(tick_positions)
                    )
                self.rec_colorbar.ax.xaxis.set_major_formatter(
                    FixedFormatter(tick_labels)
                    )
                self.rec_colorbar.ax.xaxis.set_minor_locator(NullLocator())

                # Force a canvas draw
                self.fig.canvas.draw()

                # Extract the generated colorbar labels
                labels = self.rec_colorbar.ax.get_xticklabels()

                # Check if more than 1 label has been generated
                if len(labels) >= 2:

                    # Get the base transformation matrix for the x-axis
                    base_transform = (
                        self.rec_colorbar.ax.get_xaxis_transform()
                        )

                    # Align the first label to the right (with offset)
                    labels[0].set_horizontalalignment('right')
                    labels[0].set_transform(
                        base_transform + ScaledTranslation(
                            -0.06, -0.02, self.fig.dpi_scale_trans)
                        )

                    # Align the last label to the left (with offset)
                    labels[-1].set_horizontalalignment('left')
                    labels[-1].set_transform(
                        base_transform + ScaledTranslation(
                            0.06, -0.02, self.fig.dpi_scale_trans)
                        )

                    # Get the current renderer
                    renderer = self.fig.canvas.get_renderer()

                    # Get the bounding box extrema
                    bbox_min = labels[0].get_window_extent(renderer)
                    bbox_max = labels[-1].get_window_extent(renderer)

                    # Define a default translation offset
                    inner_offset = ScaledTranslation(
                        0, -0.1, self.fig.dpi_scale_trans
                        )

                    # Loop over the range of the inner tick labels
                    for i in range(1, len(labels) - 1):

                        # Center-align the label and shift it downwards
                        labels[i].set_horizontalalignment('center')
                        labels[i].set_transform(base_transform + inner_offset)

                        # Get the bounding box of the label
                        bbox_inner = labels[i].get_window_extent(renderer)

                        # Check if the inner label collides with the outer
                        if (bbox_inner.x0 < bbox_min.x1
                                or bbox_inner.x1 > bbox_max.x0
                                ):

                            # Hide the label
                            labels[i].set_visible(False)

                        else:

                            # Show the label
                            labels[i].set_visible(True)

            else:

                # Define evenly spaced tick locations
                tick_locs = linspace(current_min, current_max, 5)

                # Set the ticks on the colorbar
                self.rec_colorbar.set_ticks(tick_locs)

                # Initialize a scientific formatter
                formatter = ScalarFormatter(useMathText=True)
                formatter.set_scientific(True)

                # Apply the formatter to the colorbar x-axis
                self.rec_colorbar.ax.xaxis.set_major_formatter(formatter)

        # Slice history safely based on filled buffer size
        history_length = min(self._hist_count, self.max_size)

        if history_length > 0:

            # Get the index of the current best fitness value
            best_idx = argmin(
                self._hist_f[:min(self._hist_count, self.max_size)]
                )

            # Update the current best solution
            self.best_x_star.set_offsets(
                [[self._hist_x[best_idx], self._hist_y[best_idx]]]
                )

        # Extract the 2D mean and covariance
        mu_2d = mean[:2]
        cov_2d = cov[:2, :2]

        # Compute the covariance ellipse
        evals, evecs = eigh(cov_2d)
        scaling = sigma * sqrt(maximum(evals, 1e-12))
        ellipse_2d = evecs @ (scaling[:, None] * self._ellipse_base)

        # Shift the ellipse coordinates by the mean
        ellipse_x = ellipse_2d[0, :] + mu_2d[0]
        ellipse_y = ellipse_2d[1, :] + mu_2d[1]

        # Update the dynamic scatter points and lines
        self.mean_dot.set_offsets([mu_2d])
        self.cov_line_outer.set_data(ellipse_x, ellipse_y)
        self.cov_line_inner.set_data(ellipse_x, ellipse_y)
        self.pop_dots.set_offsets(population[:, :2])

        # =====================================================================
        # COORDINATE SPACE
        # =====================================================================

        # Compute the lower/upper percentile and the dimension-wise mean
        lower_perc = percentile(population, 2.5, axis=0)
        upper_perc = percentile(population, 97.5, axis=0)
        coord_mean = nmean(population, axis=0)

        # Update the coordinate mean line
        self.coord_mean_line.set_data(self.indices_dim, coord_mean)

        # Check if the 95%-CI band exists
        if self.coord_fill_band is not None:

            try:

                # Remove the band
                self.coord_fill_band.remove()

            except (ValueError, AttributeError):

                pass

        # Generate the 95%-CI band
        self.coord_fill_band = self.ax_coord.fill_between(
            self.indices_dim, lower_perc, upper_perc, color='#4EA8DE',
            alpha=0.6, zorder=4
            )

        # Compute the 95%-CI range
        coord_max = max(upper_perc.max(), coord_mean.max())
        coord_min = min(lower_perc.min(), coord_mean.min())
        coord_range = coord_max - coord_min

        # Check if the range is very small
        if abs(coord_range) < 1e-9:

            # Add fixed margins to limits on the y-axis
            self.ax_coord.set_ylim(coord_min - 0.1, coord_max + 0.1)

        else:

            # Add range-dependent margins to the limits on the y-axis
            self.ax_coord.set_ylim(
                coord_min - 0.05 * coord_range, coord_max + 0.05 * coord_range
                )

        # =====================================================================
        # FITNESS HISTORY
        # =====================================================================

        # Add the new data to the histories
        self._hist_gens.append(iteration)
        self._hist_min.append(fitness.min())
        self._hist_q25.append(percentile(fitness, 25))
        self._hist_mean.append(nmean(fitness))
        self._hist_q75.append(percentile(fitness, 75))
        self._hist_max.append(fitness.max())

        # Check if the first iteration is running
        if iteration == 1:

            # Activate and configure the markers
            self.fit_min_line.set_marker('o')
            self.fit_min_line.set_markersize(4)
            self.fit_mean_line.set_marker('o')
            self.fit_mean_line.set_markersize(3)
            self.fit_max_line.set_marker('o')
            self.fit_max_line.set_markersize(3)

        else:

            # Remove the markers
            self.fit_min_line.set_marker('')
            self.fit_mean_line.set_marker('')
            self.fit_max_line.set_marker('')

        # Update the line plots
        self.fit_min_line.set_data(self._hist_gens, self._hist_min)
        self.fit_mean_line.set_data(self._hist_gens, self._hist_mean)
        self.fit_max_line.set_data(self._hist_gens, self._hist_max)

        try:

            # Check if the IQR band exists
            if self.fit_inner_band is not None:

                # Remove the band
                self.fit_inner_band.remove()

            # Check if the range band exists
            if self.fit_outer_band is not None:

                # Remove the band
                self.fit_outer_band.remove()

        except (ValueError, AttributeError):

            pass

        # Check if the first iteration has passed
        if iteration > 1:

            # Generate the IQR and range bands
            self.fit_inner_band = self.ax_fit.fill_between(
                self._hist_gens, self._hist_q25, self._hist_q75, color='red',
                alpha=0.35, zorder=5, label='IQR'
                )
            self.fit_outer_band = self.ax_fit.fill_between(
                self._hist_gens, self._hist_min, self._hist_max, color='red',
                alpha=0.22, zorder=4, label='Range'
                )

        # Get the minimum and maximum fitness from the history
        y_min, y_max = min(self._hist_min), max(self._hist_max)

        # Get the fitness range
        y_range = y_max - y_min

        #
        is_log = (y_max - y_min) > 10000 and y_min > 0

        #
        if is_log:

            #
            self.ax_fit.set_yscale('log')

            #
            log_buf_min = 10**(0.05 * log10(y_max / y_min))
            self.ax_fit.set_ylim(y_min / log_buf_min, y_max * log_buf_min)

        else:

            #
            self.ax_fit.set_yscale('linear')

            # Check if the fitness range is very small
            if abs(y_range) < 1e-9:

                # Set the range to a fixed value
                y_range = 1e-9

            #
            self.ax_fit.set_ylim(
                y_min - 0.05 * y_range, y_max + 0.05 * y_range
                )

        # =====================================================================
        # FEASIBILITY SPECTRUM
        # =====================================================================

        # Check if bound errors have been recorded
        if squared_bound_errors is not None:

            # Sort the squared boundary errors (ascending)
            err_sorted = argsort(squared_bound_errors)
            err_sorted = squared_bound_errors[err_sorted].astype(float)

            # Check if any errors occurred
            if nall(squared_bound_errors == 0.0):

                # Show the "Feasible" text
                self.err_status_text.set_text("✔ FEASIBLE")
                self.err_status_text.set_visible(True)

            else:

                # Set the "Feasible" text invisible
                self.err_status_text.set_visible(False)

            # Initialize an array for the logarithmized errors
            log_err_sorted = zeros_like(err_sorted)

            # Get the error mask
            err_mask = err_sorted > 0.0

            # Check if any errors occurred
            if any(err_mask):

                # Add the logarithmic errors to the array
                log_err_sorted[err_mask] = log10(err_sorted[err_mask])

            # Set the remaining values to the defined lower value
            log_err_sorted[~err_mask] = self._err_log_floor

            # Clip the logarithmized values
            log_err_sorted = clip(
                log_err_sorted, self._err_log_floor, self._err_log_ceil
                )

            # Set the lower-bound values to nan
            log_err_sorted[log_err_sorted <= self._err_log_floor] = nan

            # Append data as a new matrix column
            self._err_spec_matrix = hstack(
                [self._err_spec_matrix, log_err_sorted.reshape(-1, 1)]
                )

            # Get the number of generations from the error spectrum
            current_gens = self._err_spec_matrix.shape[1]

            # Check if the mesh handle is still None
            if self.err_density_mesh is None:

                # Initialize the mesh handle
                self.err_density_mesh = self.ax_err.imshow(
                    self._err_spec_matrix,
                    extent=[0.5, current_gens + 0.5, 0.5, self.pop_size + 0.5],
                    origin='lower', cmap=self._err_cmap, norm=self._err_norm,
                    alpha=0.9, aspect='auto', zorder=1, interpolation='nearest'
                    )

            else:

                # Update the mesh handle
                self.err_density_mesh.set_data(self._err_spec_matrix)
                self.err_density_mesh.set_extent(
                    [0.5, current_gens + 0.5, 0.5, self.pop_size + 0.5]
                    )

            # Update the colorbar
            self.err_density_cbar.update_normal(self.err_density_mesh)
            self.err_density_cbar.ax.set_yticks(self._cbar_err_ticks)
            self.err_density_cbar.ax.set_yticklabels(
                [f"$10^{{{int(round(t))}}}$" for t in self._cbar_err_ticks]
                )

            # Set the limit and ticks for the x-axis
            self.ax_err.set_xlim(1, iteration + 2)
            self.ax_err.set_xticks(custom_x_ticks)

        # =====================================================================
        # COVARIANCE RANK
        # =====================================================================

        # Get the sorted singular values (descending)
        svs_sorted = sort(svs)[::-1]

        # Compute the sum of the singular values
        total_sum = sum(svs_sorted)

        # Check if the sum is positive
        if total_sum > 0:

            # Compute the relative cumulative distribution
            rel_cum_distr = (cumsum(svs_sorted) / total_sum) * 100.0

            # Get the indices related to 80 %, 95 %, and 99 %
            idx_80_arr = where(rel_cum_distr >= 80.0)[0]
            idx_95_arr = where(rel_cum_distr >= 95.0)[0]
            idx_99_arr = where(rel_cum_distr >= 99.0)[0]

            # Compute the number of PCs for these levels
            pcs_80 = (
                int(idx_80_arr[0]) + 1 if idx_80_arr.size > 0
                else self.dimensions
                )
            pcs_95 = (
                int(idx_95_arr[0]) + 1 if idx_95_arr.size > 0
                else self.dimensions
                )
            pcs_99 = (
                int(idx_99_arr[0]) + 1 if idx_99_arr.size > 0
                else self.dimensions
                )

        else:

            # Set the number of PCs to the full dimensionality
            pcs_80, pcs_95, pcs_99 = (
                self.dimensions, self.dimensions, self.dimensions
                )

        # Add the new data to the histories
        self._svs_hist_gens.append(iteration)
        self._svs_hist_80.append(pcs_80)
        self._svs_hist_95.append(pcs_95)
        self._svs_hist_99.append(pcs_99)

        # Update the line plots
        self.svs_80_line.set_data(self._svs_hist_gens, self._svs_hist_80)
        self.svs_95_line.set_data(self._svs_hist_gens, self._svs_hist_95)
        self.svs_99_line.set_data(self._svs_hist_gens, self._svs_hist_99)

        try:

            # Check if the IQR band exists
            if self.svs_inner_band is not None:

                # Remove the band
                self.svs_inner_band.remove()

            # Check if the range band exists
            if self.svs_outer_band is not None:

                # Remove the band
                self.svs_outer_band.remove()

        except (ValueError, AttributeError):

            pass

        # Check if the first iteration has passed
        if iteration > 1:

            # Generate the IQR and range bands
            self.svs_inner_band = self.ax_svs.fill_between(
                self._svs_hist_gens, self._svs_hist_80, self._svs_hist_95,
                color='#2A9D8F', alpha=0.12, zorder=5
                )
            self.svs_outer_band = self.ax_svs.fill_between(
                self._svs_hist_gens, self._svs_hist_80, self._svs_hist_99,
                color='#2A9D8F', alpha=0.05, zorder=4
                )

        # Update the interior text fields
        self.svs_val_80.set_text(f"{pcs_80} PCs")
        self.svs_val_95.set_text(f"{pcs_95} PCs")
        self.svs_val_99.set_text(f"{pcs_99} PCs")

        # Get the global minimum rank
        global_min_rank = min(
            [min(self._svs_hist_80), min(self._svs_hist_95),
             min(self._svs_hist_99)]
            )

        # Derive the minimum on the y-axis
        y_lim_min = max(1, global_min_rank)

        # Set the limits on the y-axis
        self.ax_svs.set_ylim(y_lim_min, self.dimensions)

        # Get the ticks for the y-axis
        raw_y_ticks = linspace(y_lim_min, self.dimensions, 6)
        clean_y_ticks = unique(raw_y_ticks.round().astype(int)).tolist()

        # Set formatter and locator for the y-axis
        self.ax_svs.yaxis.set_major_formatter(ScalarFormatter())
        self.ax_svs.yaxis.set_minor_locator(plt.NullLocator())

        # Set the ticks and labels at the y-axis
        self.ax_svs.set_yticks(clean_y_ticks)

        try:

            # Get the graphic renderer
            renderer = self.fig.canvas.get_renderer()

            # Get the bounding box
            axes_bbox = self.ax_svs.get_window_extent(renderer)

            # Loop over the interior text fields
            for text_obj in [
                    self.svs_val_80, self.svs_val_95, self.svs_val_99]:

                # Set the position
                text_obj.set_horizontalalignment('left')
                text_obj.set_x(0.84)

                # Recalculate the bounding box after positioning
                text_bbox = text_obj.get_window_extent(renderer)

                # Check if the text "jumps" out of the plot
                if text_bbox.x1 > (axes_bbox.x1 - 10.0):

                    # Change the text position
                    text_obj.set_horizontalalignment('right')
                    text_obj.set_x(0.98)

        except (AttributeError, RuntimeError):

            pass

        # =====================================================================
        # SUBTITLE AND REFRESH
        # =====================================================================

        # Update the dynamic subtitle
        self.suptitle_text.set_text(
            f"Generation {iteration:d}  |  f = {optimal_value:.2e}  |  "
            fr"$\sigma$ = {sigma:.2e}  |  Updated in {time()-start:.2f}s"
        )

        # Refresh the figure
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        self.fig.canvas.start_event_loop(delay)

    def export(
        self,
        which,
        filename="sm_plot.pdf",
        aspect=None,
        dpi=300):
        """
        Export a single plot object into a separate file.

        Parameters
        ----------
        which : {'landscape', 'coordinates', 'fitness', 'feasibility', 'rank'}
            The specific subplot to export. Options are:

                - 'landscape': the reconstructed 2D objective space (top left)
                - 'coordinates': the coordinate space (bottom left)
                - 'fitness': the fitness history (top right)
                - 'feasibility': the feasibility spectrum (middle right)
                - 'rank': the covariance rank (bottom right)

        filename : str
            Output path ending in '.pdf' or '.png'.

        aspect : float or None, default=None
            The aspect ratio of the plot box (height / width).
            For example, 1.0 enforces an exact square. If None, the current
            layout aspect ratio from the grid is preserved.

        dpi : int
            Resolution for PNG exports (ignored for PDF).
        """

        # Map the plot types to the axes
        axes = {
            'landscape': self.ax_rec,
            'coordinates': self.ax_coord,
            'fitness': self.ax_fit,
            'feasibility': self.ax_err,
            'rank': self.ax_svs}

        # Get the target axis
        target_ax = axes[which]

        # Save the original aspect ratio state to restore it later
        orig_aspect = target_ax.get_box_aspect()

        # Apply the temporary custom aspect ratio if requested
        if aspect is not None:
            target_ax.set_box_aspect(aspect)

        try:
            # Force a canvas draw to compute actual layout geometries
            self.fig.canvas.draw()

            # Get the current renderer
            renderer = self.fig.canvas.get_renderer()

            # Compute the tight bounding box of the target axis
            bbox = target_ax.get_tightbbox(renderer)

            # Include the horizontal colorbar for the landscape plot
            if which == 'landscape' and hasattr(self, 'rec_colorbar'):

                # Get the colorbar bounding box
                cbar_bbox = self.rec_colorbar.ax.get_tightbbox(renderer)

                # Combine the plot area and colorbar area
                from matplotlib.transforms import Bbox
                bbox = Bbox.union([bbox, cbar_bbox])

            # Include the vertical colorbar for the feasibility spectrum
            elif which == 'feasibility' and hasattr(self, 'err_density_cbar'):

                # Get the colorbar bounding box
                cbar_bbox = self.err_density_cbar.ax.get_tightbbox(renderer)

                # Combine the plot area and colorbar area
                from matplotlib.transforms import Bbox
                bbox = Bbox.union([bbox, cbar_bbox])

            # Convert the bounding box from pixels to inches
            bbox_inches = bbox.transformed(self.fig.dpi_scale_trans.inverted())

            # Save the area inside the bounding box
            self.fig.savefig(
                filename, bbox_inches=bbox_inches, dpi=dpi,
                format=filename.split('.')[-1]
                )

        finally:
            # Safely restore the original layout aspect ratio for the live UI
            target_ax.set_box_aspect(orig_aspect)
            self.fig.canvas.draw()
