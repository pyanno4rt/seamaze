"""Fitness plotting."""

# Authors: Tim Ortkamp, Chinmay Patwardhan, Pia Stammer

# %% External package import

import matplotlib.pyplot as plt

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

    # Truncate the fitness series
    fitness = [
        series[:head] if head is not None else series for series in fitness]

    # Define the labels and colors
    labels = ['Best', 'Mean', 'Worst']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    # Define the marker style and line width
    series_length = len(fitness[0])
    marker_style = '.' if series_length < 50 else None
    line_width = 0.75

    # Get the plot function
    plot_func = ax.semilogy if semilog else ax.plot

    # Loop over the fitness series and styles
    for series, label, color in zip(fitness, labels, colors):


        # Plot the fitness line
        plot_func(
            series, marker=marker_style, markersize=3, linestyle='-',
            linewidth=line_width, color=color, label=label, alpha=0.9)

    # Check if a title has been provided
    if title:

        # Set the title
        ax.set_title(title, fontweight='bold')

    # Set the axis labels
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Add a legend
    ax.legend(loc='best', frameon=True, fancybox=False, edgecolor='black')

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
