"""Result plotting."""

# Authors: Tim Ortkamp, Chinmay Patwardhan, Pia Stammer

# %% Internal package import

from numpy import array

# %% Internal package import

from seamaze.plotting import plot_fitness, plot_matrix_slices, plot_series

# %% Plotting function


def plot_results(
        data, label, show_objective=True, show_fitness=True,
        show_step_size=True, show_mean_change_norm=True,
        show_sigma_path_norm=True, show_cov_path_norm=True, show_cov_svs=True,
        show_cov_norm=True, show_cov_cn=True, show_cov_spectr_norm=True,
        save_path=None):
    """
    Plot the optimization results.

    Parameters
    ----------
    monitor_data : dict
        Dictionary with the tracked parameters.

    label : str
        Sublabel for the plot titles.

    show_objective : bool, default=True
        The indicator for plotting the objective values.

    show_fitness : bool, default=True
        The indicator for plotting the fitness values.

    show_step_size : bool, default=True
        The indicator for plotting the step size.

    show_mean_change_norm : bool, default=True
        The indicator for plotting the mean change norm.

    show_sigma_path_norm : bool, default=True
        The indicator for plotting the sigma path norm.

    show_cov_path_norm : bool, default=True
        The indicator for plotting the covariance path norm.

    show_cov_svs : bool, default=True
        The indicator for plotting the covariance singular values.

    show_cov_norm : bool, default=True
        The indicator for plotting the covariance norm.

    show_cov_cn : bool, default=True
        The indicator for plotting the covariance condition number.

    show_cov_spectr_norm : bool, default=True
        The indicator for plotting the covariance spectral norm.

    save_path : None or str, default=None
        The file path where the figure should be saved.
    """

    # Check if the objective values should be plotted
    if show_objective and 'optimal_value' in data:

        # Plot the objective values
        plot_series(
            series=data['optimal_value'],
            head=None,
            semilog=False,
            title=f'Objective value ({label})',
            xlabel='Generation',
            ylabel='Value',
            save_path=save_path)

    # Check if the fitness values should be plotted
    if show_fitness and all(key in data for key in (
            'best_fitness', 'mean_fitness', 'worst_fitness')):

        # Plot the fitness statistics
        plot_fitness(
            fitness=[
                data['best_fitness'], data['mean_fitness'],
                data['worst_fitness']],
            head=None,
            semilog=False,
            title=f'Fitness ({label})',
            xlabel='Generation',
            ylabel='Value',
            save_path=save_path)

    # Check if the step size should be plotted
    if show_step_size and 'sigma' in data:

        # Plot the step size
        plot_series(
            series=data['sigma'],
            head=None,
            semilog=False,
            title=f'Step size ({label})',
            xlabel='Generation',
            ylabel='Value',
            save_path=save_path)

    # Check if the mean change norm should be plotted
    if show_mean_change_norm and 'mean_change_norm' in data:

        # Plot the mean change norm
        plot_series(
            series=data['mean_change_norm'],
            head=None,
            semilog=False,
            title=f'Mean change norm ({label})',
            xlabel='Generation',
            ylabel='Value',
            save_path=save_path)

    # Check if the sigma path norm should be plotted
    if show_sigma_path_norm and 'path_sigma_norm' in data:

        # Plot the sigma path norm
        plot_series(
            series=data['path_sigma_norm'],
            head=None,
            semilog=False,
            title=fr'$p_{{\sigma}}$-norm ({label})',
            xlabel='Generation',
            ylabel='Value',
            save_path=save_path)

    # Check if the covariance path norm should be plotted
    if show_cov_path_norm and 'path_cov_norm' in data:

        # Plot the covariance path norm
        plot_series(
            series=data['path_cov_norm'],
            head=None,
            semilog=False,
            title=fr'$p_{{cov}}$-norm ({label})',
            xlabel='Generation',
            ylabel='Value',
            save_path=save_path)

    # Check if the covariance singular value paths should be plotted
    if show_cov_svs and 'cov_svs' in data:

        # Plot the covariance singular value paths
        plot_matrix_slices(
            matrix=array(data['cov_svs']),
            axis=0,
            step=1,
            semilog=True,
            title=f'Covariance singular values ({label})',
            xlabel='Generation',
            ylabel='Value (log scale)',
            save_path=save_path)

    # Check if the covariance norm should be plotted
    if show_cov_norm and 'cov_norm' in data:

        # Plot the covariance norm
        plot_series(
            series=data['cov_norm'],
            head=None,
            semilog=False,
            title=f'Covariance norm ({label})',
            xlabel='Generation',
            ylabel='Value',
            save_path=save_path)

    # Check if the covariance condition number should be plotted
    if show_cov_cn and 'cov_cn' in data:

        # Plot the covariance condition number
        plot_series(
            series=data['cov_cn'],
            head=None,
            semilog=False,
            title=f'Covariance condition number ({label})',
            xlabel='Generation',
            ylabel='Value',
            save_path=save_path)

    # Check if the covariance spectral norm should be plotted
    if show_cov_spectr_norm and 'cov_spectr_norm' in data:

        # Plot the covariance spectral norm
        plot_series(
            series=data['cov_spectr_norm'],
            head=None,
            semilog=False,
            title=f'Covariance spectral norm ({label})',
            xlabel='Generation',
            ylabel='Value',
            save_path=save_path)
