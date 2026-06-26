"""Optimization results plotting."""

# Authors: Tim Ortkamp, Chinmay Patwardhan, Pia Stammer

# %% Internal package import

from datetime import datetime
from pathlib import Path
from numpy import array, nan, pad

# %% Internal package import

from seamaze.plotting import (
    plot_bound_violations, plot_fitness, plot_matrix_slices, plot_series)

# %% Class definition


class ResultPlotter:
    """
    This class implements a result plotter to generate and save different
    types of metric plots.

    Parameters
    ----------
    data : dict
        Dictionary with the tracked optimization parameters and histories.

    label : str
        Sublabel for the plot titles and generated filenames.

    save_folder : str or pathlib.Path, optional
        Target directory for exporting figure files. If None, plots are only
        generated in-memory without disk export.
    """

    def __init__(
            self,
            data,
            label,
            save_folder=None):

        # Get the arguments
        self.data = data
        self.label = label
        self.prefix = self._initialize_save_directory(save_folder)

        # Initialize the flags
        self.show_objective = True
        self.show_fitness = True
        self.show_bound_viol = True
        self.show_step_size = True
        self.show_mean_change_norm = True
        self.show_sigma_path_norm = True
        self.show_cov_path_norm = True
        self.show_cov_svs = True
        self.show_cov_norm = True
        self.show_cov_cn = True
        self.show_cov_spectr_norm = True
        self.show_rank = True

    def _initialize_save_directory(
            self,
            save_folder):
        """
        Configure the target directory.

        Parameters
        ----------
        save_folder : str or pathlib.Path, optional
            Target directory for exporting figure files. If None, plots are
            only generated in-memory without disk export.

        Returns
        -------
        pathlib.Path or None
            Target prefix for exporting figure files.
        """

        # Check if the
        if not save_folder:

            return None

        # Set the directory path and create a folder if not existent
        folder_path = Path(save_folder)
        folder_path.mkdir(parents=True, exist_ok=True)

        # Create a datetime string
        date_str = datetime.now().strftime("%Y-%m-%d")

        # Return the prefix
        return folder_path / f"{date_str}_{self.label}_"

    def _get_save_path(
            self,
            filename):
        """
        Resolve the full path for a file plot.

        Parameters
        ----------
        filename : str
            Name of the plot.

        Returns
        -------
        str or None
            The resolved full path.
        """

        # Check if no prefix is available
        if self.prefix is None:

            return None

        # Return the joint path and filename prefix
        return str(self.prefix.with_name(f"{self.prefix.name}{filename}"))

    def plot_objective(self):
        """Plot the objective values."""

        # Check if the objective values should be plotted
        if self.show_objective and 'optimal_value' in self.data:

            # Plot the series
            plot_series(
                series=self.data['optimal_value'],
                head=None,
                semilog=False,
                title=f'Objective value ({self.label})',
                xlabel='Generation',
                ylabel='Value',
                save_path=self._get_save_path('optimal_value.pdf')
                )

    def plot_fitness_statistics(self):
        """Plot the fitness statistics."""

        # Check if the fitness values should be plotted
        if self.show_fitness and all(key in self.data for key in (
                'best_fitness', 'mean_fitness', 'worst_fitness')):

            # Plot the fitness statistics
            plot_fitness(
                fitness=[
                    self.data['best_fitness'], self.data['mean_fitness'],
                    self.data['worst_fitness']],
                head=None,
                semilog=False,
                title=f'Fitness ({self.label})',
                xlabel='Generation',
                ylabel='Value',
                save_path=self._get_save_path('fitness.pdf')
                )

    def plot_boundary_violations(self):
        """Plot the boundary error statistics and penalty factors."""

        # Check if the bound violations should be plotted
        if self.show_bound_viol and all(key in self.data for key in (
                'max_bound_viol', 'mean_bound_viol', 'gamma')):

            # Plot the bound violation statistics
            plot_bound_violations(
                violation=[
                    self.data['max_bound_viol'], self.data['mean_bound_viol']
                    ],
                head=None,
                semilog=False,
                title=f'Sum of squared bound errors ({self.label})',
                xlabel='Generation',
                ylabel='Value',
                save_path=self._get_save_path('squared_bound_viols.pdf')
                )

            # Plot the gamma value (penalty factor)
            plot_series(
                series=self.data['gamma'],
                head=None,
                semilog=False,
                title=f'Penalty factor ({self.label})',
                xlabel='Generation',
                ylabel='Value',
                save_path=self._get_save_path('gamma.pdf')
                )

    def plot_step_size(self):
        """Plot the trajectory of the step size (sigma)."""

        # Check if the step size should be plotted
        if self.show_step_size and 'sigma' in self.data:

            # Plot the step size
            plot_series(
                series=self.data['sigma'],
                head=None,
                semilog=False,
                title=f'Step size ({self.label})',
                xlabel='Generation',
                ylabel='Value',
                save_path=self._get_save_path('sigma.pdf')
                )

    def plot_mean_change_norm(self):
        """Plot the norm of change in the distribution mean."""

        # Check if the mean change norm should be plotted
        if self.show_mean_change_norm and 'mean_change_norm' in self.data:

            # Plot the mean change norm
            plot_series(
                series=self.data['mean_change_norm'][1:],
                head=None,
                semilog=False,
                title=f'Mean change norm ({self.label})',
                xlabel='Generation',
                ylabel='Value',
                save_path=self._get_save_path('mean_change_norm.pdf')
                )

    def plot_sigma_path_norm(self):
        """Plot the norm of the sigma evolution path."""

        # Check if the sigma path norm should be plotted
        if self.show_sigma_path_norm and 'path_sigma_norm' in self.data:

            # Plot the sigma path norm
            plot_series(
                series=self.data['path_sigma_norm'],
                head=None,
                semilog=False,
                title=fr'$p_{{\sigma}}$-norm ({self.label})',
                xlabel='Generation',
                ylabel='Value',
                save_path=self._get_save_path('path_sigma_norm.pdf')
                )

    def plot_covariance_path_norm(self):
        """Plot the norm of the covariance matrix evolution path."""

        # Check if the covariance path norm should be plotted
        if self.show_cov_path_norm and 'path_cov_norm' in self.data:

            # Plot the covariance path norm
            plot_series(
                series=self.data['path_cov_norm'],
                head=None,
                semilog=False,
                title=fr'$p_{{cov}}$-norm ({self.label})',
                xlabel='Generation',
                ylabel='Value',
                save_path=self._get_save_path('path_cov_norm.pdf')
                )

    def plot_covariance_singular_values(self):
        """Plot the covariance singular values at checkpoints and as paths."""

        # Check if the covariance singular values should be plotted
        if not self.show_cov_svs or 'cov_svs' not in self.data:

            return

        # Get the singular values
        svs = self.data['cov_svs']

        # Get the number of generations
        number_of_generations = len(svs)

        # Set the checkpoints
        checkpoints = {
            'start': (0, 1),
            'mid': (
                number_of_generations // 2, number_of_generations // 2 + 1
                ),
            'end': (-1, number_of_generations)
            }

        # Loop over the checkpoints
        for phase, (index, iteration) in checkpoints.items():

            # Plot the singular values
            plot_series(
                series=sorted(svs[index], reverse=True),
                head=None,
                semilog=True,
                title=(
                    f'Singular values at iteration {iteration} ({self.label})'
                    ),
                xlabel='Index',
                ylabel='Singular value',
                save_path=self._get_save_path(f'cov_svs_{phase}.pdf')
                )

        # Pad the covariance paths if necessary
        max_length = max(len(arr) for arr in svs)
        padded_list = [
            pad(arr, (0, max_length - len(arr)), constant_values=nan)
            for arr in svs
            ]

        # Plot the covariance singular value paths
        plot_matrix_slices(
            matrix=array(padded_list),
            axis=0,
            step=1,
            semilog=True,
            title=f'Covariance singular values ({self.label})',
            xlabel='Generation',
            ylabel='Value',
            save_path=self._get_save_path('cov_svs.pdf')
            )

    def plot_covariance_norm(self):
        """Plot the norm of the covariance matrix."""

        # Check if the covariance norm should be plotted
        if self.show_cov_norm and 'cov_norm' in self.data:

            # Plot the covariance norm
            plot_series(
                series=self.data['cov_norm'],
                head=None,
                semilog=False,
                title=f'Covariance norm ({self.label})',
                xlabel='Generation',
                ylabel='Value',
                save_path=self._get_save_path('cov_norm.pdf')
                )

    def plot_covariance_condition_number(self):
        """Plot the condition number of the covariance matrix."""

        # Check if the covariance condition number should be plotted
        if self.show_cov_cn and 'cov_cn' in self.data:

            # Plot the covariance condition number
            plot_series(
                series=self.data['cov_cn'],
                head=None,
                semilog=False,
                title=f'Covariance condition number ({self.label})',
                xlabel='Generation',
                ylabel='Value',
                save_path=self._get_save_path('cov_cn.pdf')
                )

    def plot_covariance_spectral_norm(self):
        """Plot the spectral norm of the covariance matrix."""

        # Check if the covariance spectral norm should be plotted
        if self.show_cov_spectr_norm and 'cov_spectr_norm' in self.data:

            # Plot the covariance spectral norm
            plot_series(
                series=self.data['cov_spectr_norm'],
                head=None,
                semilog=False,
                title=f'Covariance spectral norm ({self.label})',
                xlabel='Generation',
                ylabel='Value',
                save_path=self._get_save_path('cov_spectr_norm.pdf')
                )

    def plot_rank(self):
        """Plot the rank of the DLR integrator."""

        # Check if the rank of the integrator should be plotted
        if self.show_rank and 'rank' in self.data:

            # Plot the integrator rank evolution
            plot_series(
                series=self.data['rank'],
                head=None,
                semilog=False,
                title=f'Covariance rank ({self.label})',
                xlabel='Generation',
                ylabel='Value',
                save_path=self._get_save_path('rank.pdf')
                )

    def plot_all(self):
        """Run all plotting methods."""

        self.plot_objective()
        self.plot_fitness_statistics()
        self.plot_boundary_violations()
        self.plot_step_size()
        self.plot_mean_change_norm()
        self.plot_sigma_path_norm()
        self.plot_covariance_path_norm()
        self.plot_covariance_singular_values()
        self.plot_covariance_norm()
        self.plot_covariance_condition_number()
        self.plot_covariance_spectral_norm()
        self.plot_rank()
