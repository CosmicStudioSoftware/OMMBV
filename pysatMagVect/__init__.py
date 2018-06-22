# -*- coding: utf-8 -*-
import os
here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, 'version.txt')) as version_file:
    __version__ = version_file.read().strip()
del here


from . import _core
from ._core import *
from . import igrf

__all__ = []