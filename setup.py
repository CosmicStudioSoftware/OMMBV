import setuptools
import numpy.distutils.core
import os

from numpy.distutils.core import Extension

# create extension for calling IGRF
extensions = [Extension(name='OMMBV.igrf',
                        sources=[os.path.join('OMMBV', 'igrf13.f')]),
              Extension(name='OMMBV.fortran_coords',
                        sources=[os.path.join('OMMBV', '_coords.f')])]

here = os.path.abspath(os.path.dirname(__file__))
version_filename = os.path.join('OMMBV', 'version.txt')
with open(os.path.join(here, version_filename)) as version_file:
    version = version_file.read().strip()

# call setup
numpy.distutils.core.setup(
    name='OMMBV',
    version=version,
    packages=['OMMBV', 'OMMBV.tests'],
    description='Orthogonal geomagnetic vector basis and field-line mapping for multipole magnetic fields.',
    url='http://github.com/rstoneback/OMMBV',

    # Author details
    author='Russell Stoneback',
    author_email='rstoneba@utdallas.edu',
    data_files=[('OMMBV', ['OMMBV/version.txt'])],
    include_package_data=True,

    # required modules
    install_requires=['numpy', 'scipy'],
    ext_modules=extensions,
)
