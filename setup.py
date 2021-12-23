#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""Setup routine for OMMBV.

Note
----
package metadata stored in setup.cfg

"""

import os
from setuptools import setup

# Include extensions only when not on readthedocs.org
if os.environ.get('READTHEDOCS', None) == 'True':
    extensions = []
else:
    from numpy.distutils.core import Extension
    from numpy.distutils.core import setup  # noqa: F811

    extra_args = ['--std=legacy', '-Wno-line-truncation', '-Wno-conversion',
                  '-Wno-unused-variable', '-Wno-maybe-uninitialized',
                  '-Wno-unused-dummy-argument']
    extensions = [Extension(name='OMMBV.igrf',
                            sources=[os.path.join('OMMBV', 'igrf13.f')],
                            extra_f77_compile_args=extra_args),
                  Extension(name='OMMBV.sources',
                            sources=[os.path.join('OMMBV', 'sources.f'),
                                     os.path.join('OMMBV', 'igrf13.f')],
                            extra_f77_compile_args=extra_args),
                  Extension(name='OMMBV.fortran_coords',
                            sources=[os.path.join('OMMBV', '_coords.f')])]

setup_kwargs = {'ext_modules': extensions}
setup(**setup_kwargs)
