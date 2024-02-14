import os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# remove everything after the string 'bayes_gsl', if anything exists
PROJECT_ROOT = PROJECT_ROOT[:PROJECT_ROOT.find('gsl-bnn') + len('gsl-bnn')]

RAW_DATA_PATH = PROJECT_ROOT + "/data/raw/"
SYNTHETIC_DATA_PATH = PROJECT_ROOT + "/data/synthetic/"

RESULTS_PATH = PROJECT_ROOT + "/results/"


"""
    Plotting and Metrics
"""

FIGURES_PATH = PROJECT_ROOT + "/figures/"

# Calibration
NUM_BINS = 10

# plotting info
TMLR_TEXTWIDTH = 6.50127
TMLR_PLOT_DEPTH = 1.75

#FIGURES_PATH = PROJECT_ROOT + "/reports/figures/"

#FIGURE_DATA_DIR = PROJECT_ROOT + "/reports/figures_data/"
