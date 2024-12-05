###############################################################################
# Configuration
###############################################################################


from .config import defaults

import yapecs

yapecs.configure("timbre", defaults)

from .config.defaults import *

###############################################################################
# Module
###############################################################################

# from .train import ...
from . import data
from . import model
