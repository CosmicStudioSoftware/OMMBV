#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""Setup routine for OMMBV.

Note
----
package metadata stored in setup.cfg

"""

import os
from setuptools import setup

version = '0.5.5'

# Include extensions only when not on readthedocs.org
if os.environ.get('READTHEDOCS', None) == 'True':

    extensions = []
else:
    from numpy.distutils.core import Extension
    from numpy.distutils.core import setup

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

setup(name='OMMBV',
      version=version,
      license='BSD-3-Clause',
      packages=['OMMBV'],
      description=' '.join(('Orthogonal geomagnetic vector basis and',
                            'field-line mapping for multipole magnetic',
                            'fields.')),
      url='https://github.com/rstoneback/OMMBV',

      # Author details
      author='Russell Stoneback',
      author_email='russell@stoneris.com',

      # data_files=[('OMMBV', )],
      # include_package_data=True,
      classifiers=[
          # complete classifier list:
          # http://pypi.python.org/pypi?%3Aaction=list_classifiers
          'Development Status :: 5 - Production/Stable',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: BSD-3-Clause License',
          'Operating System :: Unix',
          'Operating System :: POSIX',
          'Operating System :: Microsoft :: Windows',
          'Operating System :: MacOS :: MacOS X',
          'Programming Language :: Python',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: Implementation :: CPython',
          'Topic :: Scientific/Engineering :: Physics',
          'Topic :: Utilities',
      ],

      # Required modules
      install_requires=['numpy', 'scipy'],
      ext_modules=extensions)
