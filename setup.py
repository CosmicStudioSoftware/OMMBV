# Copyrighted by The University of Texas at Dallas subject to the United States 
# government unlimited rights (DFAR 252.227-7013-5).

import setuptools
import numpy.distutils.core
import os
import sys
import numpy as np

from numpy.distutils.misc_util import Configuration

config = Configuration('pysatMagVect')

# create extension for calling IGRF
config.add_extension(name='igrf',
    sources = [os.path.join('pysatMagVect', 'igrf12.f')]
    )

here = os.path.abspath(os.path.dirname(__file__))
#with open(path.join(here, 'description.txt'), encoding='utf-8') as f:
#    long_description = f.read()
version_filename = os.path.join('pysatMagVect', 'version.txt')
with open(os.path.join(here, version_filename)) as version_file:
    version = version_file.read().strip()

# call setup
#--------------------------------------------------------------------------
numpy.distutils.core.setup( 

    name = 'pysatMagVect',
    version = version,        
    packages = ['pysatMagVect','pysatMagVect.tests'],
    description= ''.join(('Calculates geomagnetic unit vectors (field aligned, zonal, and meridional) '
                 'and includes supporting routines for characterizing the motion of ionospheric plasma.')),
    # cmdclass={'sdist': sdist},
   
    # Author details
    author='Russell Stoneback',
    author_email='rstoneba@utdallas.edu',
    package_data={'pysatMagVect': ['pysatMagVect/version.txt'],},
    include_package_data=True,
    
    ext_modules = config.todict()['ext_modules'],

    install_requires = ['numpy', 'scipy'],

)  



