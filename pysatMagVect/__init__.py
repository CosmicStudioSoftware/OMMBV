# -*- coding: utf-8 -*-
import os
here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, 'version.txt')) as version_file:
    __version__ = version_file.read().strip()
del here

from . import igrf
from . import _core
from ._core import *

__all__ = []
