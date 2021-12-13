# -*- coding: utf-8 -*-
import os

__version__ = '0.5.5'

on_rtd = os.environ.get('ONREADTHEDOCS') == 'True'

if not on_rtd:
    from OMMBV import igrf
else:
    igrf = None

from OMMBV import _core
from OMMBV._core import *
from OMMBV import satellite
from OMMBV import utils

__all__ = []
