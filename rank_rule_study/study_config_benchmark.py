"""Settings of the benchmark script itself: n = 10 and 100 with the provided
start points, default population size and λ = 100, without gradient. All
other settings as in study_config.py (tolerance 1e-12, decay of Hansen 2016,
eq. 47). Selected with STUDY_CONFIG=study_config_benchmark (start_benchmark.sh).
"""
from study_config import *    # noqa: F401,F403

DIMS = [10, 100]
POP_SIZES = [None, 100]
USE_GRADIENT = [False]

RESULTS_DIR = 'results_benchmark_settings'
