###############################################################################
# Configuration
###############################################################################


import yapecs

from .config import defaults

yapecs.configure("timbre", defaults)

###############################################################################
# Module
###############################################################################

# from .train import ...
from . import data, model
from .config.defaults import *
