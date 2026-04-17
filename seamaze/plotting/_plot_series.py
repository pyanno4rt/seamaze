"""Series plotting."""

# Authors: Tim Ortkamp, Chinmay Patwardhan, Pia Stammer

# %% External package import

import matplotlib.pyplot as plt

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

    # Set the style parameters
    plt.rcParams.update({
        "text.usetex": False,
        "font.family": 'sans-serif',
        "font.serif": ['Helvetica', 'Arial', 'DejaVu Sans', 'Liberation Sans'],
        "axes.labelsize": 10,
        "font.size": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.labelweight": 'normal',
        "pdf.fonttype": 42,
        "ps.fonttype": 42
        })

    # Create the figure
    _, ax = plt.subplots(figsize=(8, 6))

    # Truncate the series
    series = series[:head] if head is not None else series

    # Define the marker style and line width
    marker_style = '.' if len(series) < 50 else None
    line_width = 0.75

    # Get the plot function
    plot_func = ax.semilogy if semilog else ax.plot

    # Plot the series
    plot_func(
        series, marker=marker_style, markersize=3,
        linestyle='-', linewidth=line_width, alpha=0.8, color='b'
        )

    # Check if a title has been provided
    if title:

        # Set the title
        ax.set_title(title, fontweight='bold')

    # Set the axis labels
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Add a grid
    ax.grid(True, which="both", ls=":", alpha=0.5, color='0.6')

    # Apply a tight layout
    plt.tight_layout(pad=0.5)

    # Check if the figure should be saved
    if save_path:

        # Save the figure
        plt.savefig(save_path, bbox_inches='tight', dpi=300, transparent=True)

    # Show the figure
    plt.show(block=True)

    # Close the figure
    plt.close()
