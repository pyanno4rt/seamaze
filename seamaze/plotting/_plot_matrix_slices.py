"""Matrix slices plotting."""

# Authors: Tim Ortkamp, Chinmay Patwardhan, Pia Stammer

# %% External package import

import matplotlib.pyplot as plt
from numpy import asarray, linspace

# %% Plotting function


def plot_matrix_slices(
        matrix, axis=0, step=1, semilog=False, title='', xlabel='', ylabel='',
        save_path=None):
    """
    Plot 1-D matrix slices.

    Parameters
    ----------
    matrix : ndarray
        The 2D input matrix.

    axis : {0, 1}, default=0
        The axis along which the matrix is sliced. If 0, the function \
        iterates over the columns, else over the rows.

    step : int, default=1
        The sampling interval for data points within the slice.

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

    # Ensure the input is an array
    matrix = asarray(matrix)

    # Check if the matrix is empty or less than 2-dimensional
    if matrix.size == 0 or matrix.ndim < 2:

        return

    # Run the plotting steps with contextual rcParams
    with plt.rc_context({'pdf.fonttype': 42, 'ps.fonttype': 42}):

        # Create the figure
        fig, ax = plt.subplots(figsize=(8, 6))

        # Transpose the matrix if column-wise slicing should be applied
        matrix = matrix.T if axis == 0 else matrix

        # Generate the color set
        colors = plt.cm.viridis(linspace(0, 1, matrix.shape[0]))

        # Loop over the matrix slices
        for index, mslice in enumerate(matrix):

            # Select values from the slice
            series = mslice[::step]

            # Get the index set
            indices = range(1, len(mslice) + 1, step)

            # Define the marker style and size
            marker_style = (
                'o' if len(series) < 30 else
                ('.' if len(series) < 100 else None)
                )

            # Get the plot function
            plot_func = ax.semilogy if semilog else ax.plot

            # Plot the line
            plot_func(
                indices, series, marker=marker_style, markersize=3,
                linestyle='-', linewidth=0.75, alpha=0.8, color=colors[index]
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
