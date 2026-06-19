"""Fitness plotting."""

# Authors: Tim Ortkamp, Chinmay Patwardhan, Pia Stammer

# %% External package import

import matplotlib.pyplot as plt
from numpy import asarray, clip, log1p

# %% Plotting function


def plot_fitness(
        fitness, head=None, semilog=False, title='', xlabel='',
        ylabel='', save_path=None):
    """
    Plot the fitness evolution (best, mean and worst) over the iterations.

    Parameters
    ----------
    fitness : list
        List containing three 1D arrays or sequences corresponding to
        [best, mean, worst] fitness values.

    head : None or int, default=None
        The number of initial data points to plot in each fitness series.

    semilog : bool, default=False
        The indicator for logarithmic scaling (base 10) on the y-axis.

    title : str, default=''
        The figure title to be displayed.

    xlabel : str, default=''
        The x-axis label to be displayed.

    ylabel : str, default=''
        The y-axis label to be displayed.

    save_path : None or str, default=None
        The file path where the figure should be saved.
    """

    # Truncate the fitness series
    fitness = [asarray(series[:head]) for series in fitness]

    # Check if the fitness series contains no data
    if len(fitness) == 0 or len(fitness[0]) == 0:

        return

    # Run the plotting steps with contextual rcParams
    with plt.rc_context({'pdf.fonttype': 42, 'ps.fonttype': 42}):

        # Create the figure
        fig, ax = plt.subplots(figsize=(8, 6))

        # Define the labels and colors
        labels = ['Best', 'Mean', 'Worst']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

        # Define the marker size
        series_length = len(fitness[0])
        marker_size = (
            0.0 if series_length > 150
            else float(clip(15.0 / log1p(series_length), 1.0, 5.0))
            )

        # Get the plot function
        plot_func = ax.semilogy if semilog else ax.plot

        # Loop over the fitness series and styles
        for series, label, color in zip(fitness, labels, colors):

            # Plot the fitness line
            plot_func(
                range(1, len(series) + 1), series, marker='o',
                markersize=marker_size, linestyle='-', linewidth=1,
                color=color, label=label, alpha=0.9
                )

        # Check if a title has been provided
        if title:

            # Set the title
            ax.set_title(title, fontweight='bold', fontsize=12)

        # Set the axis labels
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(
            f"{ylabel} (log scale)" if semilog else ylabel, fontsize=10
            )

        # Set the tick params
        ax.tick_params(axis='both', labelsize=9)

        # Add a legend
        ax.legend(
            loc='best', frameon=True, fancybox=False, edgecolor='black',
            fontsize=9
            )

        # Add a grid
        ax.grid(True, which="both", ls=":", alpha=0.5, color='0.6')

        # Apply a tight layout
        plt.tight_layout(pad=0.5)

        # Check if the figure should be saved
        if save_path:

            # Save the figure
            plt.savefig(
                save_path, bbox_inches='tight', dpi=300, transparent=True
                )

            # Close the figure
            plt.close(fig)

        else:

            # Show the figure
            plt.show(block=True)

            # Close the figure
            plt.close(fig)
