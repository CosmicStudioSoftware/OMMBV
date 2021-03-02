# Orthogonal Multi-pole Magnetic Basis Vectors (OMMBV)
[![Build Status](https://travis-ci.com/rstoneback/OMMBV.svg?branch=master)](https://travis-ci.com/rstoneback/OMMBV)
[![Coverage Status](https://coveralls.io/repos/github/rstoneback/OMMBV/badge.svg?branch=master)](https://coveralls.io/github/rstoneback/OMMBV?branch=master)
[![Documentation Status](https://readthedocs.org/projects/ommbv/badge/?version=latest)](https://ommbv.readthedocs.io/en/latest/?badge=latest)
[![DOI](https://zenodo.org/badge/138220240.svg)](https://zenodo.org/badge/latestdoi/138220240)

The motion of plasma in the ionosphere is the result of forcing from neutral winds, electric fields,  
as well as the orientation of the background magnetic field. Plasma moves easily along the magnetic field line and less 
 so across it. OMMBV (Orthogonal Multipole Magnetic Basis Vectors) calculates directions (unit vectors) 
 based upon the geomagnetic field that are optimized for understanding the movement of plasma and the coupling 
 with the neutral atmosphere. This system is the first to remain orthogonal for multipole magnetic fields as well as
 when including a geodetic reference surface (Earth).
 
 OMMBV also includes methods for scaling ion drifts at one location to 
 either the magnetic footpoint or to the magnetic equator. Scaling to the footpoint is critical for understanding how 
 neutral atmosphere winds at low altitudes (footpoint heights) will be expressed either at the satellite location or at 
 the magnetic equator. Scaling to the magnetic equator can be particularly effective when creating a common basis for 
 integrating measurements from multiple platforms.

OMMBV is used by the upcoming NASA Ionospheric Connections (ICON) Explorer Mission to understand how remote 
measurements of neutral motions at 120 km impacts the motion of plasma measured in situ (at the satellite location). 
This package is also being used by the upcoming NOAA/NSPO COSMIC-2 constellation to express plasma measurements made 
at the satellite locations in a more geophysically useful basis. OMMBV is currently being incorporated into analysis 
routines suitable for integrating physics-based models (TIEGCM) and measurements from the Communications/Navigation 
Outage Forecasting System (C/NOFS) satellite.

The development of the multipole software has been supported, in part, by multiple agencies under the following grants:
Naval Research Laboratory N00173-19-1-G016 and NASA 80NSSC18K1203.

Previous versions of this software that provided an 'average' basis were funded by: 
National Aeronautics and Space Agency (NASA NNG12FA45C), National Oceanic and Atmospheric 
Administration(NOAA NSF AGS-1033112), and the National Science Foundation (NSF 1651393).

# Field-Line Tracing
The International Geomagnetic Reference Field (IGRF) is coupled into SciPy's odeint to produce an accurate field
line tracing algorithm that terminates at a supplied reference height, or after a fixed number of steps. The SciPy integrator is an adaptive method that internally determines an appropriate step size thus the performance of the technique is both robust and accurate. The sensitivity of the field line tracing and other quantities in this package have been established via direct comparison (when possible) as well as sensitivity and consistency tests.

# Geomagnetic Unit Vectors
Plasma in the ionosphere is constrained by the geomagnetic field. Motion along magnetic field lines is easy while motion across field lines is comparatively hard. To understand the motion of ions it is generally best to do so along these directions.

 - Field Aligned: Along the geomagnetic field, pointing generally from south to north at the equator.

 - Zonal: Perpendicular to the field aligned vector and the meridional plane of the field line (plane formed by the large scale distribution of the field line)

 - Meridional: Perpendicular to the zonal and field aligned directions. This vector is positive upward and is vertical at the geomagnetic equator. To remain perpendicular to the field, the meridional vector has a poleward component when away from the magnetic equator. Note that meridional may sometimes be used in other contexts to be north/south. Here, the vector is generally up/down.

 # Ion Drift Mapping
 Calculates scalars for mapping ion motions expressed in geomagnetic unit vectors to either the magnetic footpoint or to the magnetic equator. These scalars are determined assuming that magnetic field lines are equipotential, thus the electric field associated with ion motion will vary as the distance between two geomagnetic field lines changes. The ExB/B^2 motion accounted for here also varies with magnetic field strength. Either the mappings for just the electric field, or the ion drift, are available. Scaling from the equator or footpoint to a different location is achieved by taking the inverse of the supplied parameters.

 # Coordinate Transformations
 Supports the conversion of geographic and geodetic (WGS84) into each other and into Earth Centered Earth Fixed (ECEF). ECEF coordinates are fixed with respect to the Earth, x points from the center towards 0 degrees longitude at the geographic equator, y similarly points to 90 degrees east, while z points along the Earth's rotation axis.

 # Vector Transformations
 Supports expressing a vector known in one basis into the same vector expressed in another basis. This supports translating measurements made in a spacecraft frame into frames more relevant for scientific analysis.
