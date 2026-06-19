"""Series plotting."""

# Authors: Tim Ortkamp, Chinmay Patwardhan, Pia Stammer

# %% External package import

import matplotlib.pyplot as plt
from numpy import asarray, clip, log1p

# %% Plotting function


def plot_series(
        series, head=None, semilog=False, title='', xlabel='', ylabel='',
        save_path=None):
    """
    Plot a 1-D series.

    Parameters
    ----------
    series : ndarray
        The 1D array or sequence.

    head : None or int, default=None
        The number of initial data points to plot.

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

    # Slice the data
    series = asarray(series)[:head]

    # Check if the series contains no data
    if len(series) == 0:

        return

    # Run the plotting steps with contextual rcParams
    with plt.rc_context({'pdf.fonttype': 42, 'ps.fonttype': 42}):

        # Create the figure
        fig, ax = plt.subplots(figsize=(8, 6))

        # Define the marker size
        series_length = len(series)
        marker_size = (
            0.0 if series_length > 150
            else float(clip(15.0 / log1p(series_length), 1.0, 5.0))
            )

        # Get the plot function
        plot_func = ax.semilogy if semilog else ax.plot

        # Plot the series
        plot_func(
            range(1, len(series) + 1), series, marker='o',
            markersize=marker_size, linestyle='-', linewidth=1, alpha=0.8,
            color='b'
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
