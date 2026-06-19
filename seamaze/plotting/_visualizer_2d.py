"""2D interactive visualizer."""

# Authors: Tim Ortkamp, Chinmay Patwardhan, Pia Stammer

# %% External package import

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.legend_handler import HandlerTuple
from matplotlib.ticker import ScalarFormatter
from numpy import (
    arange, argmin, argsort, clip, cos, cumsum, digitize, float64, full,
    isnan, linspace, log1p, log10, maximum, meshgrid, nan, nanmax, nanmin,
    percentile, pi, sin, sort, sqrt, stack, unique, where, zeros, zeros_like)
from numpy import all as nall
from numpy import mean as nmean
from scipy.ndimage import gaussian_filter
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
            "legend.fontsize": 7,
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
        if bounds is None:
            self.bounds = [(-1, -1), (1, 1)]
        else:
            self.bounds = bounds
        self.dimensions = dimensions
        self.pop_size = pop_size

        # Get the index sets (konsistent 1-basiert)
        self.indices_dim = arange(1, self.dimensions + 1, 1)
        self.indices_pop = arange(1, self.pop_size + 1, 1)

        #
        if self.dimensions <= 6:
            self.custom_dim_ticks = list(range(1, self.dimensions + 1))
        else:
            # Wir berechnen eine glatte Schrittweite (z. B. 200 bei 1000)
            approx_step = self.dimensions // 5
            if approx_step >= 100:
                step = (approx_step // 100) * 100  # Runden auf nächste 100
            elif approx_step >= 10:
                step = (approx_step // 10) * 10    # Runden auf nächste 10
            else:
                step = approx_step
            step = max(1, step)

            # Erzeuge glatte Zwischenschritte (z. B.)
            middle_ticks = list(range(step, self.dimensions, step))

            # Kollisionsschutz: Wenn der letzte Zwischenschritt zu nah am echten Ende klebt, weg damit
            if middle_ticks and (self.dimensions - middle_ticks[-1]) < (step * 0.3):
                middle_ticks.pop()

            # Baue die finale Liste: IMMER die 1, die glatten Schritte, IMMER das exakte Ende!
            self.custom_dim_ticks = [1] + middle_ticks + [self.dimensions]

        # Get the marker size
        marker_size_dim = (
            0.0 if self.dimensions > 150
            else float(clip(15.0 / log1p(self.dimensions), 1.0, 5.0))
            )

        # -------------------------------------------------
        # GENERAL FIGURE SETUP ----------------------------
        # -------------------------------------------------

        # Initialize the figure
        self.fig = plt.figure(figsize=(12, 7.5))

        # Adjust the subplots
        self.fig.subplots_adjust(
            top=0.78, bottom=0.08, left=0.07, right=0.92, wspace=0.5,
            hspace=0.7
            )

        # Define a 3x2 grid layout
        plot_grid = self.fig.add_gridspec(
            3, 2, width_ratios=[2.2, 1], height_ratios=[1.0, 1.0, 1.0]
            )

        # Assign the subplots
        self.ax_2d = self.fig.add_subplot(plot_grid[0:2, 0])
        self.ax_coord = self.fig.add_subplot(plot_grid[2, 0])
        self.ax_fit = self.fig.add_subplot(plot_grid[0, 1])
        self.ax_err = self.fig.add_subplot(plot_grid[1, 1])
        self.ax_svs = self.fig.add_subplot(plot_grid[2, 1])

        #
        self.ax_2d.set_aspect('auto')

        # Get the plotting bounds
        x_min, x_max = self.bounds[0][0], self.bounds[1][0]
        y_min, y_max = self.bounds[0][1], self.bounds[1][1]

        # -------------------------------------------------
        # 2D LANDSCAPE PLOT INITIALIZATION (MEMORY-NET) ---
        # -------------------------------------------------

        max_hist_size = 100000
        self._hist_x = zeros(max_hist_size, dtype=float64)
        self._hist_y = zeros(max_hist_size, dtype=float64)
        self._hist_f = zeros(max_hist_size, dtype=float64)
        self._hist_count = 0

        grid_points = 100
        self.grid_x, self.grid_y = meshgrid(
            linspace(x_min, x_max, grid_points),
            linspace(y_min, y_max, grid_points)
        )
        self._grid_nodes = stack([self.grid_x.ravel(), self.grid_y.ravel()], axis=-1)

        # Das mathematische Gedächtnis-Netz (100x100 Pixel)
        self.grid_memory = full((grid_points, grid_points), nan)

        # Für schnelles Mapping von Koordinaten auf Pixel-Indizes
        self.x_edges = linspace(x_min, x_max, grid_points + 1)
        self.y_edges = linspace(y_min, y_max, grid_points + 1)

        # WICHTIG: Kein Dummy-Contourf mehr! Wir setzen den Starter auf None.
        self.memory_contours = None

        # FIX: Starr auf Breitbild stellen, BEVOR irgendwas gezeichnet wird (verhindert das Hüpfen)
        self.ax_2d.set_aspect('auto')

        # FIX: Colorbar über ein flexibles ScalarMappable initialisieren
        from matplotlib.cm import ScalarMappable
        self.initial_mappable = ScalarMappable(norm=Normalize(vmin=0, vmax=1), cmap='coolwarm')

        self.memory_colorbar = self.fig.colorbar(
            self.initial_mappable, ax=self.ax_2d, location='bottom',
            orientation='horizontal', shrink=0.7, pad=0.16, aspect=40
        )
        formatter = ScalarFormatter(useMathText=False)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-3, 3))
        formatter.set_useOffset(False)
        self.memory_colorbar.ax.xaxis.set_major_formatter(formatter)
        self.memory_colorbar.set_label(r'$f$', rotation=0, labelpad=5, fontsize=7)
        self.memory_colorbar.ax.tick_params(labelsize=8)

        # Initialize the orange star for the current best solution
        self.best_x_star = self.ax_2d.scatter(
            [], [], color='#FFD700', marker='*', s=180, linewidths=0.5,
            zorder=13, edgecolors='#1A1A1A', label='Best'
            )

        # Set plot title, axis labels, and limits
        self.ax_2d.set_title(
            r'Reconstructed Objective Space ($x_1$, $x_2$)', fontsize=10,
            loc='center', pad=6
            )
        self.ax_2d.set_xlabel(r'$x_1$')
        self.ax_2d.set_ylabel(r'$x_2$')
        self.ax_2d.set_xlim(x_min, x_max)
        self.ax_2d.set_ylim(y_min, y_max)

        # Initialize the mean/population dots and the covariance line
        self.mean_dot = self.ax_2d.scatter(
            [], [], color='#FFFFFF', marker='*', s=160, linewidths=0.5,
            edgecolors='#1A1A1A', zorder=14, label='Mean'
            )
        self.cov_line_outer, = self.ax_2d.plot(
            [], [], color='#1A1A1A', lw=3.0, zorder=11, label='Covariance'
            )
        self.cov_line_inner, = self.ax_2d.plot(
            [], [], color='#FFFFFF', lw=1.2, zorder=12, label='_nolegend_'
            )
        self.pop_dots = self.ax_2d.scatter(
            [], [], color='#FF9F1C', s=30, linewidths=0.4, alpha=0.8,
            edgecolors='#1A1A1A', zorder=5, label='Population'
            )

        # Set the legend
        legend = self.ax_2d.legend(
            handles=[
                self.best_x_star,
                self.mean_dot,
                (self.cov_line_outer, self.cov_line_inner),
                self.pop_dots
            ],
            labels=[
                'Best',
                'Mean',
                'Covariance',
                'Population'
            ],
            handler_map={tuple: HandlerTuple(ndivide=None, pad=0.0)},
            loc='upper left',
            bbox_to_anchor=(1.01, 1.0),
            ncol=1,
            fontsize=7,
            framealpha=1.0
            )
        legend.set_zorder(100)

        # -------------------------------------------------
        # PARALLEL COORDINATES PLOT INITIALIZATION --------
        # -------------------------------------------------

        # Configure the parallel coordinates subplot
        self.ax_coord.set_title('Coordinate Space', fontsize=10, pad=6)
        self.ax_coord.set_xlabel('Dimension')
        self.ax_coord.set_ylabel('Value')
        self.ax_coord.set_xlim(
            1 - 0.02 * self.dimensions, 1.02 * self.dimensions
            )
        self.ax_coord.grid(True, linestyle=':', alpha=0.6)

        # Place evenly-spaced x-ticks
        self.ax_coord.set_xticks(self.custom_dim_ticks)

        # Initialize the mean line and the population line collection
        self.coord_mean_line, = self.ax_coord.plot(
            [], [], color='#03045E', lw=2.2, marker='o',
            markersize=marker_size_dim, zorder=10, label='Mean'
            )

        # OPTIMIERUNG: Wir initialisieren eine leere PolyCollection für das Band.
        # fill_between erzeugt beim ersten Zeichnen dieses Objekt.
        self.coord_fill_band = None

        # Legendensetup (Wir nutzen einen kleinen Trick, um das Band in der Legende als Fläche anzuzeigen)
        from matplotlib.patches import Patch
        legend = self.ax_coord.legend(
            handles=[self.coord_mean_line, Patch(color='#4EA8DE', alpha=0.6)],
            labels=['Mean', r'$\sigma$-band'],
            loc='upper left', bbox_to_anchor=(1.01, 1.0), ncol=1, fontsize=7, framealpha=1.0
        )
        legend.set_zorder(100)

        # -------------------------------------------------
        # 6. FITNESS PLOT INITIALIZATION (PERFECT ALIGNMENT VIA DIVIDER)
        # -------------------------------------------------
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        from matplotlib.patches import Patch

        self.ax_fit.set_title('Fitness History', fontsize=10, pad=6)
        self.ax_fit.set_xlabel('Generation')
        self.ax_fit.set_ylabel('Fitness')
        self.ax_fit.grid(True, linestyle=':', alpha=0.4)

        self._hist_gens = []
        self._hist_min = []
        self._hist_q25 = []
        self._hist_mean = []
        self._hist_q75 = []
        self._hist_max = []

        self.fit_trail_max_line, = self.ax_fit.plot([], [], color='#ADB5BD', linestyle='-', lw=0.6, alpha=0.5, zorder=3)
        self.fit_mean_line, = self.ax_fit.plot([], [], color='#495057', linestyle='-', lw=1.2, zorder=5)
        self.fit_min_line, = self.ax_fit.plot([], [], color='red', linestyle='-', lw=2.0, zorder=10)

        self.fit_inner_band = None
        self.fit_outer_band = None

        # Symmetrie-Fix: Wir schneiden rechts exakt 4% Platz ab, um das Alignment zum unteren Plot zu erzwingen!
        divider_fit = make_axes_locatable(self.ax_fit)
        leg_ax = divider_fit.append_axes("right", size="4%", pad=0.04)
        leg_ax.axis('off') # Diese Hilfsachse bleibt komplett unsichtbar

        dummy_outer = Patch(color='red', alpha=0.08)
        dummy_inner = Patch(color='red', alpha=0.20)

        # Die Legende sitzt starr in der leeren rechten Spalte
        self.fit_legend = self.ax_fit.legend(
            handles=[self.fit_min_line, self.fit_mean_line, dummy_inner, dummy_outer],
            labels=['Best', 'Mean', 'IQR', 'Range'],
            loc='upper left', bbox_to_anchor=(1.0, 1.0), ncol=1, fontsize=7, framealpha=1.0
        )
        self.fit_legend.set_zorder(100)

        # -------------------------------------------------
        # 7. CONSTRAINT FEASIBILITY INITIALIZATION (SPECTRUM)
        # -------------------------------------------------
        from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm

        # FIX: Titel auf "Constraint Feasibility", X-Label auf "Generation" geändert
        self.ax_err.set_title('Feasibility Spectrum', fontsize=10, pad=6)
        self.ax_err.set_xlabel('Generation')

        # FIX: Kein physikalisches Label auf der Y-Achse
        self.ax_err.set_ylabel('Individual')
        self.ax_err.grid(False)

        self._err_spec_max_gens = 100000
        self._err_spec_matrix = full((self.pop_size, self._err_spec_max_gens), nan)

        self._err_log_floor = -10.0
        self._err_log_ceil = 5.0

        err_boundaries = arange(self._err_log_floor, self._err_log_ceil + 1, 1)

        high_contrast_colors = ["#FFF0F5", "#FAD2E1", "#F77F00", "#D90429", "#3F000B"]
        base_traffic_cmap = LinearSegmentedColormap.from_list("infeasibility_fade", high_contrast_colors)

        self._err_cmap = base_traffic_cmap
        self._err_cmap.set_bad("#FFFFFF")
        self._err_norm = BoundaryNorm(boundaries=err_boundaries, ncolors=self._err_cmap.N, clip=True)

        self.err_density_mesh = self.ax_err.imshow(
            self._err_spec_matrix,
            extent=[0.5, self._err_spec_max_gens + 0.5, 1, self.pop_size],
            origin='lower', cmap=self._err_cmap, norm=self._err_norm, alpha=0.9, aspect='auto', zorder=1,
            interpolation='nearest'
        )

        self.ax_err.set_xlim(0.5, 50.5)
        self.ax_err.set_ylim(1, self.pop_size)

        # FIX: Nur noch Start und Ende als Ticks, keine Zwischenlabels
        self.ax_err.set_yticks([1, self.pop_size])

        self.err_status_text = self.ax_err.text(
            0.97, 0.93, "", color='#38B000', fontsize=8, fontweight='bold',
            ha='right', va='top', transform=self.ax_err.transAxes, zorder=100,
            animated=False,
            bbox=dict(facecolor='#FFFFFF', edgecolor='#38B000', alpha=0.95, boxstyle='round,pad=0.4', lw=1.0)
        )
        self.err_status_text.set_visible(False)

        # Schnitt rechts: Exakt 4% Platz abspalten mit pad=0.04
        divider_err = make_axes_locatable(self.ax_err)
        cbar_ax = divider_err.append_axes("right", size="4%", pad=0.04)

        # Colorbar anlegen
        # Wir speichern die cbar_ticks als Klassenvariable (self.), damit wir im Update darauf zugreifen können!
        self._cbar_err_ticks = arange(self._err_log_floor, self._err_log_ceil + 1, 3)
        self.err_density_cbar = self.fig.colorbar(
            self.err_density_mesh, cax=cbar_ax, orientation='vertical', ticks=self._cbar_err_ticks
        )

        self.err_density_cbar.set_label('Violation', fontsize=7, labelpad=5)
        self.err_density_cbar.ax.tick_params(labelsize=7)

        # -------------------------------------------------
        # SINGULAR VALUES PLOT INITIALIZATION (RANK EVOLUTION)
        # -------------------------------------------------
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        self.ax_svs.set_title('Covariance Rank', fontsize=10, pad=6)
        self.ax_svs.set_xlabel('Generation')
        self.ax_svs.set_ylabel('Effective Rank')
        self.ax_svs.grid(True, linestyle=':', alpha=0.4)

        self._svs_hist_gens = []
        self._svs_hist_80 = []
        self._svs_hist_95 = []
        self._svs_hist_99 = []

        # Die drei Linien (wie gehabt)
        self.svs_80_line, = self.ax_svs.plot([], [], color='#4A5568', linestyle='-', lw=2.0, zorder=8)
        self.svs_95_line, = self.ax_svs.plot([], [], color='#D90429', linestyle='-', lw=2.2, zorder=10)
        self.svs_99_line, = self.ax_svs.plot([], [], color='#2A9D8F', linestyle='-', lw=2.0, zorder=7)

        # -------------------------------------------------
        # COVARIANCE RANK TEXTBOX INITIALIZATION (OFFSETBOX)
        # -------------------------------------------------

        # 2. Die drei Werte-Felder, exakt rechts neben den Zeilentiteln positioniert.
        # Jedes Feld bekommt die identische Farbe deiner zugehörigen Linien!
        # X=0.18 schiebt die Zahlen perfekt hinter den Text.
        # Die Y-Werte (0.895, 0.835, 0.775) treffen das Zeilenraster dank linespacing haarscharf.
        self.svs_val_80 = self.ax_svs.text(0.84, 0.79, "", color='#4A5568', fontsize=7, fontweight='bold', ha='left', va='top', transform=self.ax_svs.transAxes, zorder=26)
        self.svs_val_95 = self.ax_svs.text(0.84, 0.88, "", color='#D90429', fontsize=7, fontweight='bold', ha='left', va='top', transform=self.ax_svs.transAxes, zorder=26)
        self.svs_val_99 = self.ax_svs.text(0.84, 0.97, "", color='#2A9D8F', fontsize=7, fontweight='bold', ha='left', va='top', transform=self.ax_svs.transAxes, zorder=26)

        self.svs_inner_band = None
        self.svs_outer_band = None

        divider_svs = make_axes_locatable(self.ax_svs)
        svs_leg_ax = divider_svs.append_axes("right", size="4%", pad=0.04)
        svs_leg_ax.axis('off')

        # Die Legende sitzt perfekt sortiert rechts außerhalb
        self.svs_legend = svs_leg_ax.legend(
            handles=[self.svs_99_line, self.svs_95_line, self.svs_80_line],
            labels=['99%', '95%', '80%'],
            loc='upper left', bbox_to_anchor=(-0.4, 1.0), ncol=1, fontsize=7, framealpha=1.0
        )
        self.svs_legend.set_zorder(100)

        # -------------------------------------------------
        # TITLE/SUBTITLE AND INITIAL DRAWING --------------
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
            svs,
            sigma,
            fitness,
            squared_bound_errors,
            best_fitness,
            delay=0.1):
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

        best_fitness : float
            Fitness value (best) of the current generation.

        delay : float, default=0.001
            Delay parameter (in seconds) to control update speed.
        """

        # -------------------------------------------------
        # 2D LANDSCAPE PLOT UPDATE (MEMORY-NET) -----------
        # -------------------------------------------------

        #
        pop_size = len(fitness)
        max_size = 100000  # Muss mit der Größe in __init__ übereinstimmen

        # Berechne zirkuläre Indizes für das globale Gedächtnis
        start_idx = self._hist_count % max_size
        end_idx = start_idx + pop_size

        # Schreibe sofort in den Ringpuffer, damit Daten für alle Plots bereitstehen
        if end_idx <= max_size:
            self._hist_x[start_idx:end_idx] = population[:, 0]
            self._hist_y[start_idx:end_idx] = population[:, 1]
            self._hist_f[start_idx:end_idx] = fitness
        else:
            overflow = end_idx - max_size
            self._hist_x[start_idx:max_size] = population[:-overflow, 0]
            self._hist_y[start_idx:max_size] = population[:-overflow, 1]
            self._hist_f[start_idx:max_size] = fitness[:-overflow]

            self._hist_x[0:overflow] = population[-overflow:, 0]
            self._hist_y[0:overflow] = population[-overflow:, 1]
            self._hist_f[0:overflow] = fitness[-overflow:]

        # Inkrementiere den globalen Zähler
        self._hist_count += pop_size
        total_pts = min(self._hist_count, max_size)

        #
        # Finde für jeden Punkt der Population den passenden Pixel im 100x100 Gitter
        x_indices = digitize(population[:, 0], self.x_edges) - 1
        y_indices = digitize(population[:, 1], self.y_edges) - 1

        # Punkte in das Gitternetz eintragen (nur wenn sie innerhalb der Bounds liegen)
        for idx in range(len(fitness)):
            xi = x_indices[idx]
            yi = y_indices[idx]
            if 0 <= xi < 100 and 0 <= yi < 100:
                # Wenn der Pixel leer ist oder der neue Wert besser ist, überschreiben
                if isnan(self.grid_memory[yi, xi]) or fitness[idx] < self.grid_memory[yi, xi]:
                    self.grid_memory[yi, xi] = fitness[idx]

        #
        display_grid = self.grid_memory.copy()
        mask = (~isnan(display_grid)).astype(float)
        display_grid[isnan(display_grid)] = 0

        # Bestimmt die Breite der Glättung um die gemessenen Punkte herum
        sigma_smooth = 1.5
        smoothed_grid = gaussian_filter(display_grid, sigma=sigma_smooth)
        smoothed_mask = gaussian_filter(mask, sigma=sigma_smooth)

        grid_z = where(smoothed_mask > 0.01, smoothed_grid / (smoothed_mask + 1e-10), nan)

        # Farb-Skalierung aus deinen echten im Netz gespeicherten Min/Max Werten
        if not nall(isnan(grid_z)):
            current_min = float(nanmin(self.grid_memory))
            current_max = float(nanmax(self.grid_memory))
            if abs(current_max - current_min) < 1e-9:
                current_max += 1e-9

            norm_scale = Normalize(vmin=current_min, vmax=current_max)

            # Alte Farbflächen sauber löschen
            if self.memory_contours is not None:
                try:
                    self.memory_contours.remove()
                except Exception:
                    pass

            # Neue gefüllte Farbzonen zeichnen
            self.memory_contours = self.ax_2d.contourf(
                self.grid_x, self.grid_y, grid_z, levels=10,
                cmap='coolwarm', norm=norm_scale, alpha=0.7, zorder=1
            )

            # Zwingt die Achse dazu, dauerhaft die volle Breite des Layouts zu nutzen
            self.ax_2d.set_aspect('auto')

            # Normierung und Limits direkt an das Mappable der Colorbar übergeben
            self.memory_colorbar.mappable.set_norm(norm_scale)
            self.memory_colorbar.mappable.set_clim(vmin=current_min, vmax=current_max)

            # Ticks berechnen
            tick_locs = linspace(current_min, current_max, 5)
            self.memory_colorbar.set_ticks(tick_locs)

            # Vollständiges Auffrischen der Skala triggern
            self.memory_colorbar.update_normal(self.memory_colorbar.mappable)

        #
        best_idx = argmin(self._hist_f[:total_pts])
        self.best_x_star.set_offsets([[self._hist_x[best_idx], self._hist_y[best_idx]]])

        mu_2d = mean[:2]
        cov_2d = cov[:2, :2]

        evals, evecs = eigh(cov_2d)
        angles = linspace(0, 2 * pi, 100)
        scaling = sigma * sqrt(maximum(evals, 1e-12))
        ellipse_2d = evecs @ (scaling[:, None] * stack([cos(angles), sin(angles)]))

        self.mean_dot.set_offsets([mu_2d])
        self.cov_line_outer.set_data(ellipse_2d[0, :] + mu_2d[0], ellipse_2d[1, :] + mu_2d[1])
        self.cov_line_inner.set_data(ellipse_2d[0, :] + mu_2d[0], ellipse_2d[1, :] + mu_2d[1])
        self.pop_dots.set_offsets(population[:, :2])

        # -------------------------------------------------
        # PARALLEL COORDINATES PLOT UPDATE (UNSICHERHEITSBAND)
        # -------------------------------------------------
        # Berechne das 5. und 95. Perzentil über die Population hinweg (Achse 0)
        # Das blendet extreme Ausreißer sanft aus und zeigt den wahren Kern der Verteilung.
        # Alternativ kannst du auch population.min(axis=0) und .max(axis=0) nutzen.
        band_lower = percentile(population, 2.5, axis=0)
        band_upper = percentile(population, 97.5, axis=0)
        band_mean = nmean(population, axis=0)

        # Die Mean-Linie wie gewohnt aktualisieren
        self.coord_mean_line.set_data(self.indices_dim, band_mean)

        # Altes Band aus der Achse entfernen, um Memory-Leaks zu verhindern
        if self.coord_fill_band is not None:
            try:
                self.coord_fill_band.remove()
            except Exception:
                pass

        # Neues, seidenweiches Band zeichnen
        self.coord_fill_band = self.ax_coord.fill_between(
            self.indices_dim, band_lower, band_upper,
            color='#4EA8DE', alpha=0.6, zorder=4
        )

        # Automatische Skalierung der Y-Achse
        coord_max = max(band_upper.max(), mean.max())
        coord_min = min(band_lower.min(), mean.min())
        coord_range = coord_max - coord_min
        if abs(coord_range) < 1e-9:
            self.ax_coord.set_ylim(coord_min - 0.1, coord_max + 0.1)
        else:
            self.ax_coord.set_ylim(coord_min - 0.05 * coord_range, coord_max + 0.05 * coord_range)

        # =================================================
        # 6. FITNESS PLOT UPDATE (MINIMUM-FOCUSED) ========
        # =================================================

        # 1. DYNAMISCHE X-ACHSEN TICKS BERECHNUNG
        if iteration <= 10:
            middle_ticks = list(range(2, iteration))
        elif iteration <= 50:
            middle_ticks = list(range(10, iteration, 10))
        elif iteration <= 200:
            middle_ticks = list(range(50, iteration, 50))
        else:
            approx_step = iteration // 5
            if approx_step > 100:
                step = (approx_step // 100) * 100
            else:
                step = (approx_step // 50) * 50
            step = max(50, step)
            middle_ticks = list(range(step, iteration, step))

        min_distance_to_end = max(2, iteration * 0.1)
        middle_ticks = [t for t in middle_ticks if (iteration - t) > min_distance_to_end]
        custom_x_ticks = [1] + middle_ticks + [iteration]

        # 2. STATISTIK-BERECHNUNG MIT ECHTEN WERTEN
        self._hist_gens.append(iteration)
        self._hist_min.append(fitness.min())
        self._hist_q25.append(percentile(fitness, 25))
        self._hist_mean.append(nmean(fitness))
        self._hist_q75.append(percentile(fitness, 75))
        self._hist_max.append(fitness.max())

        # FIX FÜR ITERATION 1: Wenn wir im ersten Schritt sind, aktivieren wir Marker,
        # damit man die Werte sofort als Punkte sieht. Ab Gen 2 schalten wir sie ab!
        if iteration == 1:
            self.fit_min_line.set_marker('o')
            self.fit_min_line.set_markersize(4)
            self.fit_mean_line.set_marker('o')
            self.fit_mean_line.set_markersize(3)
            self.fit_trail_max_line.set_marker('o')
            self.fit_trail_max_line.set_markersize(3)
        else:
            self.fit_min_line.set_marker('')
            self.fit_mean_line.set_marker('')
            self.fit_trail_max_line.set_marker('')

        # 3. LINIEN-UPDATES (Fokus auf Min in Rot)
        self.fit_min_line.set_data(self._hist_gens, self._hist_min)  # Rote Linie folgt dem Rekord!
        self.fit_mean_line.set_data(self._hist_gens, self._hist_mean)
        self.fit_trail_max_line.set_data(self._hist_gens, self._hist_max)

        # 4. Schattierungen refreshen (sie liegen als "Trichter" über den Linien)
        if self.fit_inner_band is not None: self.fit_inner_band.remove()
        if self.fit_outer_band is not None: self.fit_outer_band.remove()

        if iteration > 1:

            self.fit_outer_band = self.ax_fit.fill_between(
                self._hist_gens, self._hist_min, self._hist_max,
                color='red', alpha=0.22, zorder=4, label='Range'
            )
            self.fit_inner_band = self.ax_fit.fill_between(
                self._hist_gens, self._hist_q25, self._hist_q75,
                color='red', alpha=0.35, zorder=5, label='IQR'
            )

        # 5. ACHSENSKALIERUNG
        self.ax_fit.set_xlim(1, iteration + 2)
        self.ax_fit.set_xticks(custom_x_ticks)

        y_min, y_max = min(self._hist_min), max(self._hist_max)
        y_range = y_max - y_min
        if abs(y_range) < 1e-9: y_range = 1e-9
        self.ax_fit.set_ylim(y_min - 0.05 * y_range, y_max + 0.05 * y_range)

        # =================================================
        # 7. CONSTRAINT FEASIBILITY PLOT UPDATE ===========
        # =================================================
        # 1. DYNAMISCHE X-ACHSEN TICKS BERECHNUNG (Max 6 Ticks, perfekt gleichverteilt)
        if iteration <= 6:
            custom_err_x_ticks = list(range(1, iteration + 1))
        else:
            raw_ticks = linspace(1, iteration, 6)
            if iteration <= 100:
                middle_ticks = clip(raw_ticks[1:-1] / 5, 0, None)
                middle_ticks = (middle_ticks * 5).round()
            elif iteration <= 500:
                middle_ticks = clip(raw_ticks[1:-1] / 50, 0, None)
                middle_ticks = (middle_ticks * 50).round()
            else:
                middle_ticks = clip(raw_ticks[1:-1] / 100, 0, None)
                middle_ticks = (middle_ticks * 100).round()

            middle_ticks = middle_ticks.astype(int).tolist()
            custom_err_x_ticks = unique([1] + middle_ticks + [iteration]).tolist()

        # Synchronisiere die X-Achse des Fitness-Plots mit
        self.ax_fit.set_xticks(custom_err_x_ticks)
        self.ax_fit.set_xlim(1, iteration + 2)

        # 2. FEHLERWERTE SORTIEREN & MANUELLE LOG10-TRANSFORMATION
        err_sorted = argsort(squared_bound_errors)
        err_sorted = squared_bound_errors[err_sorted].astype(float)

        if all(squared_bound_errors == 0.0):
            self.err_status_text.set_text("✔ FEASIBLE")
            self.err_status_text.set_visible(True)
        else:
            self.err_status_text.set_visible(False)

        log_err_mapped = zeros_like(err_sorted)
        has_err_mask = err_sorted > 0.0
        if any(has_err_mask):
            log_err_mapped[has_err_mask] = log10(err_sorted[has_err_mask])

        log_err_mapped[~has_err_mask] = self._err_log_floor
        log_err_mapped = clip(log_err_mapped, self._err_log_floor, self._err_log_ceil)
        log_err_mapped[log_err_mapped <= self._err_log_floor] = nan

        # 3. DIREKTES EINTRAGEN IN DIE MATRIX
        col_idx = iteration - 1
        if col_idx < self._err_spec_matrix.shape[1]:
            self._err_spec_matrix[:, col_idx] = log_err_mapped

        # 4. IN-PLACE TEXTURE UPDATE
        self.err_density_mesh.set_data(self._err_spec_matrix)

        # Das überschreibt im Standardmodus leider die Beschriftung...
        self.err_density_cbar.update_normal(self.err_density_mesh)

        # FIX FÜR DIE BESCHRIFTUNG: Wir zwingen Matplotlib DIREKT nach dem Update,
        # die Achsenticks wieder als echte Zehnerpotenzen zu formatieren!
        self.err_density_cbar.ax.set_yticks(self._cbar_err_ticks)
        self.err_density_cbar.ax.set_yticklabels([f"$10^{{{int(round(t))}}}$" for t in self._cbar_err_ticks])

        # 5. ACHSENSKALIERUNG UND GEZEICHNETE TICKS ZUWEISEN
        self.ax_err.set_xlim(1, iteration + 2)
        self.ax_err.set_xticks(custom_err_x_ticks)

        # =================================================
        # 8. COVARIANCE RANK UPDATE (TEXTBOX & EQUIDISTANT)
        # =================================================
        # MATHEMATISCHE AUSWERTUNG DER DREI ENERGIE-SCHRANKEN
        svs_sorted = sort(svs)[::-1]
        total_sum = sum(svs_sorted)

        if total_sum > 0:
            cum_energy_pct = (cumsum(svs_sorted) / total_sum) * 100.0

            # Das [0] sprengt die Tupel-Hülle von where() fehlerfrei weg
            idx_80_arr = where(cum_energy_pct >= 80.0)[0]
            idx_95_arr = where(cum_energy_pct >= 95.0)[0]
            idx_99_arr = where(cum_energy_pct >= 99.0)[0]

            pcs_80 = int(idx_80_arr[0]) + 1 if idx_80_arr.size > 0 else self.dimensions
            pcs_95 = int(idx_95_arr[0]) + 1 if idx_95_arr.size > 0 else self.dimensions
            pcs_99 = int(idx_99_arr[0]) + 1 if idx_99_arr.size > 0 else self.dimensions
        else:
            pcs_80, pcs_95, pcs_99 = self.dimensions, self.dimensions, self.dimensions

        # Datenpunkte in die unbegrenzten Historien-Listen anhängen
        self._svs_hist_gens.append(iteration)
        self._svs_hist_80.append(pcs_80)
        self._svs_hist_95.append(pcs_95)
        self._svs_hist_99.append(pcs_99)

        # Linien- und Bänderupdates
        self.svs_80_line.set_data(self._svs_hist_gens, self._svs_hist_80)
        self.svs_95_line.set_data(self._svs_hist_gens, self._svs_hist_95)
        self.svs_99_line.set_data(self._svs_hist_gens, self._svs_hist_99)

        if self.svs_inner_band is not None: self.svs_inner_band.remove()
        if self.svs_outer_band is not None: self.svs_outer_band.remove()

        if iteration > 1:
            self.svs_outer_band = self.ax_svs.fill_between(
                self._svs_hist_gens, self._svs_hist_80, self._svs_hist_99, color='#2A9D8F', alpha=0.05, zorder=4
            )
            self.svs_inner_band = self.ax_svs.fill_between(
                self._svs_hist_gens, self._svs_hist_80, self._svs_hist_95, color='#2A9D8F', alpha=0.12, zorder=5
            )

        # FIX: Die drei Werte einzeln in ihre farbigen Textfelder schreiben.
        # Die Zahlen "reiten" optisch perfekt mitsamt der Hintergrundbox oben links!
        self.svs_val_80.set_text(f"{pcs_80} PCs")
        self.svs_val_95.set_text(f"{pcs_95} PCs")
        self.svs_val_99.set_text(f"{pcs_99} PCs")

        # Achsenskalierung (X-Achse synchron zu den oberen Plots)
        self.ax_svs.set_xlim(1, iteration + 2)
        self.ax_svs.set_xticks(custom_err_x_ticks)

        # -------------------------------------------------
        # FIX: STRIKT ÄQUIDISTANTE Y-ACHSE ----------------
        # -------------------------------------------------
        # 1. Das obere Achsenlimit (y_lim_min) wird starr vom historischen Minimum bestimmt
        global_min_rank = min([min(self._svs_hist_80), min(self._svs_hist_95), min(self._svs_hist_99)])
        y_lim_min = max(1, global_min_rank)

        # Aufsteigende Y-Achse fixieren (Kleine Ränge unten, max Dimension oben)
        self.ax_svs.set_ylim(y_lim_min, self.dimensions)

        # 2. FIX: Unterteile den Bereich von y_lim_min bis self.dimensions in EXAKT 6 äquidistante Ticks!
        # raw_y_ticks erzeugt die 6 mathematisch perfekt gleichmäßig verteilten Punkte.
        raw_y_ticks = linspace(y_lim_min, self.dimensions, 6)

        # Um krumme Zahlenwerte zu verhindern, runden wir das Gitter sauber in Ganzzahlen um.
        # unique() stellt sicher, dass am Anfang (falls der Zoom klein ist) keine doppelten Ticks gezeichnet werden.
        clean_y_ticks = unique(raw_y_ticks.round().astype(int)).tolist()

        # Matplotlibs Standard-Textformate zurücksetzen (Verhindert den alten AttributeError)
        from matplotlib.ticker import ScalarFormatter
        self.ax_svs.yaxis.set_major_formatter(ScalarFormatter())
        self.ax_svs.yaxis.set_minor_locator(plt.NullLocator())

        # Ticks und Labels unerschütterlich äquidistant zuweisen
        self.ax_svs.set_yticks(clean_y_ticks)
        self.ax_svs.set_yticklabels([str(t) for t in clean_y_ticks])

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
