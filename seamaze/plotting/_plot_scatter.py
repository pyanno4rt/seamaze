"""Scatter plotting."""

# Authors: Tim Ortkamp, Chinmay Patwardhan, Pia Stammer

# %% External package import

import matplotlib.pyplot as plt

# %% Plotting function


def plot_scatter(
        x, y, head=None, semilog=False, title='', xlabel='', ylabel='',
        save_path=None):
    """
    Plot scatter points.

    Parameters
    ----------
    x : ndarray
        The 1D array with x-coordinates.

    y : ndarray
        The 1D array with y-coordinates.

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

    # Get the coordinates
    coordinates = list(zip(x, y))

    # Truncate the series
    coordinates = coordinates[:head] if head is not None else coordinates

    # Define the marker style and line width
    marker_style = '.' if len(coordinates) < 50 else None

    # Unpack the coordinates
    x, y = zip(*coordinates)

    # Plot the series
    plt.scatter(
        x=x, y=y, s=3, c='b', marker=marker_style, linestyle='-', alpha=0.8
        )

    # Check if a semilog plot should be generated
    if semilog:

        # Logarithmize the y-axis
        plt.yscale('log')

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

        # Close the figure
        plt.close()

    else:

        # Show the figure
        plt.show(block=True)

        # Close the figure
        plt.close()
