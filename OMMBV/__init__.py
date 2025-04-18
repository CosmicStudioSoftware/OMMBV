# -*- coding: utf-8 -*-
"""init routine for OMMBV."""

__version__ = '1.1.0'

try:
    from OMMBV import igrf
    from OMMBV import fortran_coords
    from OMMBV import sources
except ImportError:
    # Warning about lack of import handled in trans
    igrf, sources, fortran_coords = None, None, None

from OMMBV import satellite
from OMMBV import trace
from OMMBV import trans
from OMMBV import utils
from OMMBV import vector

from OMMBV import _core
from OMMBV._core import *

from OMMBV import heritage

__all__ = ['igrf', 'fortran_coords', 'sources', 'satellite',
           'trace', 'trans', 'utils', 'vector', 'heritage']

print("OMMBV brought to you by Cosmic Studio.")