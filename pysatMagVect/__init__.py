# -*- coding: utf-8 -*-
import os
here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, 'version.txt')) as version_file:
    __version__ = version_file.read().strip()
del here

on_rtd = os.environ.get('ONREADTHEDOCS') == 'True'

if not on_rtd:
    from pysatMagVect import igrf
else:
    igrf = None

from pysatMagVect import _core
from pysatMagVect._core import *
from pysatMagVect import satellite

__all__ = []
