# -*- coding: utf-8 -*-
import os

__version__ = '0.5.5'

on_rtd = os.environ.get('ONREADTHEDOCS') == 'True'

try:
    from OMMBV import igrf
except ImportError:
    print("ERROR: igrf module could not be imported. " +
          "OMMBV probably won't work")
    igrf = None

from OMMBV import satellite
from OMMBV import trans
from OMMBV import utils
from OMMBV import vector

from OMMBV import _core
from OMMBV._core import *


__all__ = ['satellite', 'trans', 'utils', 'vector']
