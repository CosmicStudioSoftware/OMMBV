#!/usr/bin/env python
# -*- encoding: utf-8 -*-
from __future__ import absolute_import

import os
from setuptools import setup, find_packages

here = os.path.abspath(os.path.dirname(__file__))
version_filename = os.path.join('OMMBV', 'version.txt')
with open(os.path.join(here, version_filename)) as version_file:
    version = version_file.read().strip()

# Include extensions only when not on readthedocs.org
if os.environ.get('READTHEDOCS', None) == 'True':
    extensions = []
else:
    from numpy.distutils.core import setup, Extension

    extensions = [Extension(name='OMMBV.igrf',
                            sources=[os.path.join('OMMBV', 'igrf13.f')],
                            extra_f77_compile_args=['--std=legacy',
                                                    '-Wno-line-truncation',
                                                    '-Wno-conversion',
                                                    '-Wno-unused-variable',
                                                    '-Wno-maybe-uninitialized',
                                                    '-Wno-unused-dummy-argument']),
                  Extension(name='OMMBV.fortran_coords',
                            sources=[os.path.join('OMMBV', '_coords.f')])]

setup(name='OMMBV',
      version=version,
      packages=['OMMBV', 'OMMBV.tests'],
      description=' '.join(('Orthogonal geomagnetic vector basis and',
                            'field-line mapping for multipole magnetic',
                            'fields.')),
      url='http://github.com/rstoneback/OMMBV',

      # Author details
      author='Russell Stoneback',
      author_email='rstoneba@utdallas.edu',
      data_files=[('OMMBV', ['OMMBV/version.txt'])],
      include_package_data=True,

      # Required modules
      install_requires=['numpy', 'scipy'],
      ext_modules=extensions)
