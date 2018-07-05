# -*- coding: utf-8 -*-
import os
here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, 'version.txt')) as version_file:
    __version__ = version_file.read().strip()
del here

on_rtd = os.environ.get('ONREADTHEDOCS') == 'True'

if not on_rtd:
    from . import igrf
else:
    igrf = None
    
from . import _core
from ._core import *

__all__ = []
