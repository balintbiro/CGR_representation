from pathlib import Path
from matplotlib import pyplot as plt, rcParams

HERE=Path(__file__).resolve().parent
PROJECT_ROOT=HERE.parent.parent
CODES=PROJECT_ROOT / "codes"
DATA=PROJECT_ROOT / "data"
RESULTS=PROJECT_ROOT / "results"
FIGURES=PROJECT_ROOT / "results/figures"

FIG_WIDTH = 7  # width of figure in inches

TINY_SIZE = 6
SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=TINY_SIZE, titleweight='bold')  # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=TINY_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=TINY_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=TINY_SIZE)  # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

rcParams['axes.spines.top'] = False
rcParams['axes.spines.right'] = False