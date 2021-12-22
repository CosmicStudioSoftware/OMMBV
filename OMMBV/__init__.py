# -*- coding: utf-8 -*-
"""init routine for OMMBV."""

import os
import warnings

__version__ = '0.5.5'

try:
    from OMMBV import igrf
    from OMMBV import fortran_coords
    from OMMBV import sources
except ImportError:
    warnings.warn("Fortran module could not be imported.", ImportWarning)
    igrf, sources, fortran_coords = None, None, None

from OMMBV import satellite
from OMMBV import trace
from OMMBV import trans
from OMMBV import utils
from OMMBV import vector

from OMMBV import _core
from OMMBV._core import *

from OMMBV import heritage

__all__ = ['heritage', 'satellite', 'trace', 'trans', 'utils', 'vector']
