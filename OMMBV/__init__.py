# -*- coding: utf-8 -*-
"""init routine for OMMBV."""

import os
import warnings

__version__ = '0.5.5'

on_rtd = os.environ.get('ONREADTHEDOCS') == 'True'

try:
    from OMMBV import igrf
except ImportError:
    warnings.warn("igrf module could not be imported.", ImportWarning)
    igrf = None


from OMMBV import satellite
from OMMBV import trace
from OMMBV import trans
from OMMBV import utils
from OMMBV import vector

from OMMBV import _core
from OMMBV._core import *

from OMMBV import heritage

__all__ = ['satellite', 'trans', 'utils', 'vector']
