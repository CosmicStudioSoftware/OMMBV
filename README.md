# pysatMagVect
[![Build Status](https://travis-ci.org/rstoneback/pysatMagVect.svg?branch=master)](https://travis-ci.org/rstoneback/pysatMagVect)
[![Coverage Status](https://coveralls.io/repos/github/rstoneback/pysatMagVect/badge.svg?branch=master)](https://coveralls.io/github/rstoneback/pysatMagVect?branch=master)
[![Documentation Status](https://readthedocs.org/projects/pysatmagvect/badge/?version=latest)](https://pysatmagvect.readthedocs.io/en/latest/?badge=latest)
[![DOI](https://zenodo.org/badge/138220240.svg)](https://zenodo.org/badge/latestdoi/138220240)

Calculates geomagnetic unit vectors (field aligned, zonal, and meridional) and includes supporting routines for characterizing the motion of ionospheric plasma

# Geomagnetic Unit Vectors
Plasma in the ionosphere is constrained by the geomagnetic field. Motion along magnetic field lines is easy while motion across field lines is comparatively hard. To understand the motion of ions it is generally best to do so along these directions.

 - Field Aligned: Along the geomagnetic field, pointing generally from south to north at the equator.

 - Zonal: Prependicular to the field aligned vector and the meridional plane of the field line (plane formed by the large scale distribution of the field line)

 - Meridional: Perpendicular to the zonal and field aligned directions. This vector is positive upward and is vertical at the geomagnetic equator. To remain perpendicular to the field, the meridional vector has a poleward component when away from the magnetic equator. Note that meridional may sometimes be used in other contexts to be north/south. Here, the vector is generally up/down.
 
 # Coordinate Transformations
 Supports the conversion of geographic and geodetic (WGS84) into eachother and into Earth Centered Earth Fixed (ECEF). ECEF coordinates are fixed with respect to the Earth, x points from the center towards 0 degrees longitude at the geographic equator, y similarly points to 90 degrees east, while z points along the Earth's rotation axis.
