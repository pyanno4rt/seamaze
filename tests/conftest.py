"""Shared pytest configuration."""

# Authors: Tim Ortkamp, Chinmay Patwardhan, Pia Stammer

# %% External package import

import matplotlib
import matplotlib.pyplot as plt

# %% Internal package import

# Importing seamaze switches matplotlib to an interactive backend and enables
# interactive mode (see seamaze/plotting/_visualizer.py), so import it first
import seamaze  # noqa: F401

# %% Configuration

# Then use the non-interactive Agg backend for all tests, so that plotting
# works on machines without a display (e.g. CI runners) and never opens a
# window
matplotlib.use('Agg')
plt.ioff()
