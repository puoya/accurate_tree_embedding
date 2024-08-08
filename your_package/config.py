# config.py

# Default parameters
MAX_DIAM = 10
DEFAULT_MODEL = 'loid'

# Default parameters
CURV_RATIO = 0.5
NO_WEIGHT_RATIO = 0.5


LEARNING_RATE_INIT = 0.01
TOTAL_EPOCHS  = 2000
DIMENSION = 2
WINDOW_RATIO = 0.025
INCREASE_FACTOR = 1.001
DECREASE_FACTOR = 0.98
INCREASE_COUNT_RATIO = 0.1
LOG_DIR = 'log'
RESULTS_DIR = "your_package/tmp/Results"
IMAGE_DIR = "your_package/tmp/Images"
MOVIE_DIR = "your_package/tmp/Movies"
SUBSPACE_DIR = "your_package/tmp/Subapce"
EPS = 10**(-12)

MAX_HEATPLOT = 2
MIN_HEATPLOT = -10

MOVIE_LENGTH = 60

# Add other default parameters as needed

EMBEDDING_MODE = 'naive'




############################### embedding ###############################

poincare_domain = (0,1)
loid_domain = (-1-10**(-3), -1+10**(-3))
frechet_lr = 0.001
frechet_max_iter = 1000
frechet_tol = 1e-8
norm_projection = 0.999999
atol = 1e-6