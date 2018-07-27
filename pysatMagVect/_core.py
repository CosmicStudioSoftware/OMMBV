
"""
Supporting routines for coordinate conversions as well as vector operations and
transformations used in Space Science.

Note these routines are not formatted by direct use by pysat.Instrument custom
function features. Given the transformations will generally be part of a larger 
calculation, the functions are formatted more traditionally.

"""

import scipy
import scipy.integrate
import numpy as np
import datetime

# import reference IGRF fortran code within the package
from . import igrf

# parameters used to define Earth ellipsoid
# WGS84 parameters below
earth_a = 6378.1370
earth_b = 6356.75231424518

# standard geoncentric Earth radius
# average radius of Earth
earth_geo_radius = 6371.

def geocentric_to_ecef(latitude, longitude, altitude):
    """Convert geocentric coordinates into ECEF
    
    Parameters
    ----------
    latitude : float or array_like
        Geocentric latitude (degrees)
    longitude : float or array_like
        Geocentric longitude (degrees)
    altitude : float or array_like
        Height (km) above presumed spherical Earth with radius 6371 km.
        
    Returns
    -------
    x, y, z
        numpy arrays of x, y, z locations in km
        
    """

    r = earth_geo_radius + altitude
    x = r * np.cos(np.deg2rad(latitude)) * np.cos(np.deg2rad(longitude))
    y = r * np.cos(np.deg2rad(latitude)) * np.sin(np.deg2rad(longitude))
    z = r * np.sin(np.deg2rad(latitude))

    return x, y, z


def ecef_to_geocentric(x, y, z, ref_height=None):
    """Convert ECEF into geocentric coordinates
    
    Parameters
    ----------
    x : float or array_like
        ECEF-X in km
    y : float or array_like
        ECEF-Y in km
    z : float or array_like
        ECEF-Z in km
    ref_height : float or array_like
        Reference radius used for calculating height.
        Defaults to average radius of 6371 km
    Returns
    -------
    latitude, longitude, altitude
        numpy arrays of locations in degrees, degrees, and km
        
    """
    if ref_height is None:
        ref_height = earth_geo_radius

    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    colatitude = np.rad2deg(np.arccos(z / r))
    longitude = np.rad2deg(np.arctan2(y, x))
    latitude = 90. - colatitude

    return latitude, longitude, r - ref_height


def geodetic_to_ecef(latitude, longitude, altitude):
    """Convert WGS84 geodetic coordinates into ECEF
    
    Parameters
    ----------
    latitude : float or array_like
        Geodetic latitude (degrees)
    longitude : float or array_like
        Geodetic longitude (degrees)
    altitude : float or array_like
        Geodetic Height (km) above WGS84 reference ellipsoid.
        
    Returns
    -------
    x, y, z
        numpy arrays of x, y, z locations in km
        
    """


    ellip = np.sqrt(1. - earth_b ** 2 / earth_a ** 2)
    r_n = earth_a / np.sqrt(1. - ellip ** 2 * np.sin(np.deg2rad(latitude)) ** 2)

    # colatitude = 90. - latitude
    x = (r_n + altitude) * np.cos(np.deg2rad(latitude)) * np.cos(np.deg2rad(longitude))
    y = (r_n + altitude) * np.cos(np.deg2rad(latitude)) * np.sin(np.deg2rad(longitude))
    z = (r_n * (1. - ellip ** 2) + altitude) * np.sin(np.deg2rad(latitude))

    return x, y, z


def ecef_to_geodetic(x, y, z, method=None):
    """Convert ECEF into Geodetic WGS84 coordinates
    
    Parameters
    ----------
    x : float or array_like
        ECEF-X in km
    y : float or array_like
        ECEF-Y in km
    z : float or array_like
        ECEF-Z in km
    method : 'iterative' or 'closed' ('closed' is deafult)
        String selects method of conversion. Closed for mathematical
        solution (http://www.epsg.org/Portals/0/373-07-2.pdf , page 96 section 2.2.1)
        or iterative (http://www.oc.nps.edu/oc2902w/coord/coordcvt.pdf).
        
    Returns
    -------
    latitude, longitude, altitude
        numpy arrays of locations in degrees, degrees, and km
        
    """

    # quick notes on ECEF to Geodetic transformations 
    # http://danceswithcode.net/engineeringnotes/geodetic_to_ecef/geodetic_to_ecef.html
    
    method = method or 'closed'

    # ellipticity of Earth    
    ellip = np.sqrt(1. - earth_b ** 2 / earth_a ** 2)
    # first eccentricity squared
    e2 = ellip ** 2  # 6.6943799901377997E-3

    longitude = np.arctan2(y, x)
    # cylindrical radius
    p = np.sqrt(x ** 2 + y ** 2)
    
    # closed form solution
    # a source, http://www.epsg.org/Portals/0/373-07-2.pdf , page 96 section 2.2.1
    if method == 'closed':
        e_prime = np.sqrt((earth_a**2 - earth_b**2) / earth_b**2)
        theta = np.arctan2(z*earth_a, p*earth_b)
        latitude = np.arctan2(z + e_prime**2*earth_b*np.sin(theta)**3, p - e2*earth_a*np.cos(theta)**3)
        r_n = earth_a / np.sqrt(1. - e2 * np.sin(latitude) ** 2)
        h = p / np.cos(latitude) - r_n

    # another possibility
    # http://ir.lib.ncku.edu.tw/bitstream/987654321/39750/1/3011200501001.pdf

    ## iterative method
    # http://www.oc.nps.edu/oc2902w/coord/coordcvt.pdf
    if method == 'iterative':
        latitude = np.arctan2(p, z)
        r_n = earth_a / np.sqrt(1. - e2*np.sin(latitude)**2)
        for i in np.arange(6):
            # print latitude
            r_n = earth_a / np.sqrt(1. - e2*np.sin(latitude)**2)
            h = p / np.cos(latitude) - r_n
            latitude = np.arctan(z / p / (1. - e2 * (r_n / (r_n + h))))
            # print h
        # final ellipsoidal height update
        h = p / np.cos(latitude) - r_n

    return np.rad2deg(latitude), np.rad2deg(longitude), h

def enu_to_ecef_vector(east, north, up, glat, glong):
    """Converts vector from East, North, Up components to ECEF
    
    Position of vector in geospace may be specified in either
    geocentric or geodetic coordinates, with corresponding expression
    of the vector using radial or ellipsoidal unit vectors.
    
    Parameters
    ----------
    east : float or array-like
        Eastward component of vector
    north : float or array-like
        Northward component of vector
    up : float or array-like
        Upward component of vector
    latitude : float or array_like
        Geodetic or geocentric latitude (degrees)
    longitude : float or array_like
        Geodetic or geocentric longitude (degrees)
    
    Returns
    -------
    x, y, z
        Vector components along ECEF x, y, and z directions
    
    """
    
    # convert lat and lon in degrees to radians
    rlat = np.radians(glat)
    rlon = np.radians(glong)
    
    x = -east*np.sin(rlon) - north*np.cos(rlon)*np.sin(rlat) + up*np.cos(rlon)*np.cos(rlat)
    y = east*np.cos(rlon) - north*np.sin(rlon)*np.sin(rlat) + up*np.sin(rlon)*np.cos(rlat)
    z = north*np.cos(rlat) + up*np.sin(rlat)

    return x, y, z

def ecef_to_enu_vector(x, y, z, glat, glong):
    """Converts vector from ECEF X,Y,Z components to East, North, Up
    
    Position of vector in geospace may be specified in either
    geocentric or geodetic coordinates, with corresponding expression
    of the vector using radial or ellipsoidal unit vectors.
    
    Parameters
    ----------
    x : float or array-like
        ECEF-X component of vector
    y : float or array-like
        ECEF-Y component of vector
    z : float or array-like
        ECEF-Z component of vector
    latitude : float or array_like
        Geodetic or geocentric latitude (degrees)
    longitude : float or array_like
        Geodetic or geocentric longitude (degrees)
    
    Returns
    -------
    east, north, up
        Vector components along east, north, and up directions
    
    """

    # convert lat and lon in degrees to radians
    rlat = np.radians(glat)
    rlon = np.radians(glong)
    
    east = -x*np.sin(rlon) + y*np.cos(rlon)
    north = -x*np.cos(rlon)*np.sin(rlat) - y*np.sin(rlon)*np.sin(rlat) + z*np.cos(rlat)
    up = x*np.cos(rlon)*np.cos(rlat) + y*np.sin(rlon)*np.cos(rlat)+ z*np.sin(rlat)

    return east, north, up


def project_ecef_vector_onto_basis(x, y, z, xx, xy, xz, yx, yy, yz, zx, zy, zz):
    """Projects vector in ecef onto different basis, with components also expressed in ECEF
    
    Parameters
    ----------
    x : float or array-like
        ECEF-X component of vector
    y : float or array-like
        ECEF-Y component of vector
    z : float or array-like
        ECEF-Z component of vector
    xx : float or array-like
        ECEF-X component of the x unit vector of new basis 
    xy : float or array-like
        ECEF-Y component of the x unit vector of new basis 
    xz : float or array-like
        ECEF-Z component of the x unit vector of new basis 
    
    """
    
    out_x = x*xx + y*xy + z*xz
    out_y = x*yx + y*yy + z*yz
    out_z = x*zx + y*zy + z*zz
    
    return out_x, out_y, out_z
    
def normalize_vector(x, y, z):
    """
    Normalizes vector to produce a unit vector.
    
    Parameters
    ----------
    x : float or array-like
        X component of vector
    y : float or array-like
        Y component of vector
    z : float or array-like
        Z component of vector
        
    Returns
    -------
    x, y, z
        Unit vector x,y,z components
        
    """
    
    mag = np.sqrt(x**2 + y**2 + z**2)
    x = x/mag
    y = y/mag
    z = z/mag
    return x, y, z
    
def cross_product(x1, y1, z1, x2, y2, z2):
    """
    Cross product of two vectors, v1 x v2
    
    Parameters
    ----------
    x1 : float or array-like
        X component of vector 1
    y1 : float or array-like
        Y component of vector 1
    z1 : float or array-like
        Z component of vector 1
    x2 : float or array-like
        X component of vector 2
    y2 : float or array-like
        Y component of vector 2
    z2 : float or array-like
        Z component of vector 2
        
    Returns
    -------
    x, y, z
        Unit vector x,y,z components
        
    """
    x = y1*z2 - y2*z1
    y = z1*x2 - x1*z2
    z = x1*y2 - y1*x2
    return x, y, z 


def field_line_trace(init, date, direction, height, steps=None,
                     max_steps=1E5, step_size=5.):
    """Perform field line tracing using IGRF and scipy.integrate.odeint.
    
    Parameters
    ----------
    init : array-like of floats
        Position to begin field line tracing from in ECEF (x,y,z) km
    date : datetime or float
        Date to perform tracing on (year + day/365 + hours/24. + etc.)
        Accounts for leap year if datetime provided.
    direction : int
         1 : field aligned, generally south to north. 
        -1 : anti-field aligned, generally north to south.
    height : float
        Altitude to terminate trace, geodetic WGS84 (km)
    steps : array-like of ints or floats
        Number of steps along field line when field line trace positions should 
        be reported. By default, each step is reported; steps=np.arange(max_steps).
    max_steps : float
        Maximum number of steps along field line that should be taken
    step_size : float
        Distance in km for each large integration step. Multiple substeps
        are taken as determined by scipy.integrate.odeint
        
    Returns
    -------
    numpy array
        2D array. [0,:] has the x,y,z location for initial point
        [:,0] is the x positions over the integration.
        Positions are reported in ECEF (km).
        
    
    """
    
    if steps is None:
        steps = np.arange(max_steps)
    if not isinstance(date, float):
        # recast from datetime to float, as required by IGRF12 code
        doy = (date - datetime.datetime(date.year,1,1)).days
        # number of days in year, works for leap years
        num_doy_year = (datetime.datetime(date.year+1,1,1) - datetime.datetime(date.year,1,1)).days
        date = date.year + float(doy)/float(num_doy_year) + (date.hour + date.minute/60. + date.second/3600.)/24.  
      
    trace_north = scipy.integrate.odeint(igrf.igrf_step, init.copy(),
                                         steps,
                                         args=(date, step_size, direction, height),
                                         full_output=False,
                                         printmessg=False,
                                         ixpr=False,
                                         mxstep=500)
    # ,
    # rtol = 1.E-10,
    # atol = 1.E-10)

    return trace_north

def calculate_mag_drift_unit_vectors_ecef(latitude, longitude, altitude, datetimes,
                                          max_steps=40000, step_size=0.5,
                                          method='auto', ref_height=120.):
    """Calculates unit vectors expressing the ion drift coordinate system
    organized by the geomagnetic field. Unit vectors are expressed
    in ECEF coordinates.
    
    Parameters
    ----------
    latitude : array-like of floats (degrees)
        Latitude of location, degrees, WGS84
    longitude : array-like of floats (degrees)
        Longitude of location, degrees, WGS84
    altitude : array-like of floats (km)
        Altitude of location, height above surface, WGS84
    datetimes : array-like of datetimes
        Time to calculate vectors
    max_steps : int
        Maximum number of steps allowed for field line tracing
    step_size : float
        Maximum step size (km) allowed when field line tracing
    method : str ('auto' 'foot_field', 'cross_foot')
        Delineates between different methods of determining the zonal vector
        'cross_foot' uses the cross product of vectors pointing from location 
        to the footpoints in both hemispheres. 'field_foot' uses the cross
        product of vectors pointing along the field and one pointing to
        the field line footpoint. 'auto' does both calculations but uses 
        whichever method produces the larger unit cross product.
    ref_height : float
        Altitude used as cutoff for labeling a field line location a footpoint
        
    Returns
    -------
    zon_x, zon_y, zon_z, fa_x, fa_y, fa_z, mer_x, mer_y, mer_z
            
    """

    steps = np.arange(max_steps)
    # calculate satellite position in ECEF coordinates
    ecef_x, ecef_y, ecef_z = geodetic_to_ecef(latitude, longitude, altitude)
    # also get position in geocentric coordinates
    geo_lat, geo_long, geo_alt = ecef_to_geocentric(ecef_x, ecef_y, ecef_z, ref_height=0.)
    idx, = np.where(geo_long < 0)
    geo_long += 360.

    north_x = [];
    north_y = [];
    north_z = []
    south_x = [];
    south_y = [];
    south_z = []
    bn = [];
    be = [];
    bd = []

    out = []
    for x, y, z, alt, colat, elong, time in zip(ecef_x, ecef_y, ecef_z, geo_alt, np.deg2rad(90. - geo_lat),
                                          np.deg2rad(geo_long), datetimes):
        init = np.array([x, y, z])
        # date = inst.yr + inst.doy / 366.
        trace_north = field_line_trace(init, time, 1., ref_height, steps=steps,
                                       step_size=step_size, max_steps=max_steps)
        trace_south = field_line_trace(init, time, -1., ref_height, steps=steps,
                                       step_size=step_size, max_steps=max_steps)
        # store final location
        trace_north = trace_north[-1, :]
        trace_south = trace_south[-1, :]
        # magnetic field at spacecraft location, using geocentric inputs
        # to get magnetic field in geocentric output
        # recast from datetime to float, as required by IGRF12 code
        doy = (time - datetime.datetime(time.year,1,1)).days
        # number of days in year, works for leap years
        num_doy_year = (datetime.datetime(time.year+1,1,1) - datetime.datetime(time.year,1,1)).days
        date = time.year + float(doy)/float(num_doy_year) + (time.hour + time.minute/60. + time.second/3600.)/24.
        # get IGRF field components
        tbn, tbe, tbd, tbmag = igrf.igrf12syn(0, date, 1, alt, colat, elong)

        south_x.append(trace_south[0])
        south_y.append(trace_south[1])
        south_z.append(trace_south[2])
        north_x.append(trace_north[0])
        north_y.append(trace_north[1])
        north_z.append(trace_north[2])

        bn.append(tbn);
        be.append(tbe);
        bd.append(tbd)

    north_x = np.array(north_x)
    north_y = np.array(north_y)
    north_z = np.array(north_z)
    south_x = np.array(south_x)
    south_y = np.array(south_y)
    south_z = np.array(south_z)
    bn = np.array(bn)
    be = np.array(be)
    bd = np.array(bd)

    # calculate vector from satellite to northern/southern footpoints
    north_x = north_x - ecef_x
    north_y = north_y - ecef_y
    north_z = north_z - ecef_z
    north_x, north_y, north_z = normalize_vector(north_x, north_y, north_z)
    south_x = south_x - ecef_x
    south_y = south_y - ecef_y
    south_z = south_z - ecef_z
    south_x, south_y, south_z = normalize_vector(south_x, south_y, south_z)
    # calculate magnetic unit vector
    bx, by, bz = enu_to_ecef_vector(be, bn, -bd, geo_lat, geo_long)
    bx, by, bz = normalize_vector(bx, by, bz)
    
    # take cross product of southward and northward vectors to get the zonal vector
    zvx_foot, zvy_foot, zvz_foot = cross_product(south_x, south_y, south_z,
                                                 north_x, north_y, north_z)  
    # getting zonal vector utilizing magnetic field vector instead
    zvx_north, zvy_north, zvz_north = cross_product(north_x, north_y, north_z,
                                                    bx, by, bz)
    # getting zonal vector utilizing magnetic field vector instead and southern point
    zvx_south, zvy_south, zvz_south = cross_product(south_x, south_y, south_z,
                                                    bx, by, bz)
    # normalize the vectors
    norm_foot = np.sqrt(zvx_foot ** 2 + zvy_foot ** 2 + zvz_foot ** 2)
    norm_north = np.sqrt(zvx_north ** 2 + zvy_north ** 2 + zvz_north ** 2)
    norm_south = np.sqrt(zvx_south ** 2 + zvy_south ** 2 + zvz_south ** 2)

    # pick the method with the largest cross product vector
    # should have largest numerical accuracy
    if method == 'foot_field':
        # use the magnetic field explicitly 
        zvx, zvy, zvz = normalize_vector(zvx_north, zvy_north, zvz_north)
        # south
        idx, = np.where(norm_south > norm_north)
        zvx[idx] = zvx_south[idx] / norm_south[idx]
        zvy[idx] = zvy_south[idx] / norm_south[idx]
        zvz[idx] = zvz_south[idx] / norm_south[idx]
    elif method == 'cross_foot':
        zvx = zvx_foot / norm_foot
        zvy = zvy_foot / norm_foot
        zvz = zvz_foot / norm_foot
        # remove any field aligned component to the zonal vector
        dot_fa = zvx * bx + zvy * by + zvz * bz
        zvx -= dot_fa * bx
        zvy -= dot_fa * by
        zvz -= dot_fa * bz
        zvx, zny, zvz = normalize_vector(zvx, zvy, zvz)
    elif method == 'auto':
        # use the magnetic field explicitly 
        zvx = zvx_north / norm_north
        zvy = zvy_north / norm_north
        zvz = zvz_north / norm_north
        full_norm = norm_north
        # south
        idx, = np.where(norm_south > full_norm)
        zvx[idx] = zvx_south[idx] / norm_south[idx]
        zvy[idx] = zvy_south[idx] / norm_south[idx]
        zvz[idx] = zvz_south[idx] / norm_south[idx]
        full_norm[idx] = norm_south[idx]
        # foot cross
        idx, = np.where(norm_foot > full_norm)
        zvx[idx] = zvx_foot[idx] / norm_foot[idx]
        zvy[idx] = zvy_foot[idx] / norm_foot[idx]
        zvz[idx] = zvz_foot[idx] / norm_foot[idx]
        # remove any field aligned component to the zonal vector
        dot_fa = zvx * bx + zvy * by + zvz * bz
        zvx -= dot_fa * bx
        zvy -= dot_fa * by
        zvz -= dot_fa * bz
        zvx, zvy, zvz = normalize_vector(zvx, zvy, zvz)

    # compute meridional vector
    # cross product of zonal and magnetic unit vector
    mx, my, mz = cross_product(zvx, zvy, zvz,
                               bx, by, bz)

    # add unit vectors for magnetic drifts in ecef coordinates
    return zvx, zvy, zvz, bx, by, bz, mx, my, mz


def add_mag_drift_unit_vectors_ecef(inst, max_steps=40000, step_size=0.5,
                                    method='auto', ref_height=120.):
    """Adds unit vectors expressing the ion drift coordinate system
    organized by the geomagnetic field. Unit vectors are expressed
    in ECEF coordinates.
    
    Parameters
    ----------
    inst : pysat.Instrument
        Instrument object that will get unit vectors
    max_steps : int
        Maximum number of steps allowed for field line tracing
    step_size : float
        Maximum step size (km) allowed when field line tracing
    method : str ('auto' 'foot_field', 'cross_foot')
        Delineates between different methods of determining the zonal vector
        'cross_foot' uses the cross product of vectors pointing from location 
        to the footpoints in both hemispheres. 'field_foot' uses the cross
        product of vectors pointing along the field and one pointing to
        the field line footpoint. 'auto' does both calculations but uses 
        whichever method produces the larger unit cross product.
    ref_height : float
        Altitude used as cutoff for labeling a field line location a footpoint
        
    Returns
    -------
    None
        unit vectors are added to the passed Instrument object with a naming 
        scheme:
            'unit_zon_ecef_*' : unit zonal vector, component along ECEF-(X,Y,or Z)
            'unit_fa_ecef_*' : unit field-aligned vector, component along ECEF-(X,Y,or Z)
            'unit_mer_ecef_*' : unit meridional vector, component along ECEF-(X,Y,or Z)
            
    """

    # add unit vectors for magnetic drifts in ecef coordinates
    zvx, zvy, zvz, bx, by, bz, mx, my, mz = calculate_mag_drift_unit_vectors_ecef(inst['latitude'], inst['longitude'], inst['altitude'], inst.data.index,
                                                                                  max_steps=max_steps, step_size=step_size, method=method, ref_height=ref_height)
    inst['unit_zon_ecef_x'] = zvx
    inst['unit_zon_ecef_y'] = zvy
    inst['unit_zon_ecef_z'] = zvz

    inst['unit_fa_ecef_x'] = bx
    inst['unit_fa_ecef_y'] = by
    inst['unit_fa_ecef_z'] = bz

    inst['unit_mer_ecef_x'] = mx
    inst['unit_mer_ecef_y'] = my
    inst['unit_mer_ecef_z'] = mz

    inst.meta['unit_zon_ecef_x'] = {'long_name': 'Zonal unit vector along ECEF-x',
                                    'desc': 'Zonal unit vector along ECEF-x',
                                    'label': 'Zonal unit vector along ECEF-x',
                                    'notes': ('Unit vector expressed using Earth Centered Earth Fixed (ECEF) frame. '
                                              'Vector system is calcluated by field-line tracing along IGRF values '
                                              'down to reference altitudes of 120 km in both the Northern and Southern '
                                              'hemispheres. These two points, along with the satellite position, are '
                                              'used to define the magnetic meridian. Vector math from here generates '
                                              'the orthogonal system.'),
                                    'axis': 'Zonal unit vector along ECEF-x',
                                    'value_min': -1.,
                                    'value_max': 1.,
                                    }
    inst.meta['unit_zon_ecef_y'] = {'long_name': 'Zonal unit vector along ECEF-y',
                                    'desc': 'Zonal unit vector along ECEF-y',
                                    'label': 'Zonal unit vector along ECEF-y',
                                    'notes': ('Unit vector expressed using Earth Centered Earth Fixed (ECEF) frame. '
                                              'Vector system is calcluated by field-line tracing along IGRF values '
                                              'down to reference altitudes of 120 km in both the Northern and Southern '
                                              'hemispheres. These two points, along with the satellite position, are '
                                              'used to define the magnetic meridian. Vector math from here generates '
                                              'the orthogonal system.'),
                                    'axis': 'Zonal unit vector along ECEF-y',
                                    'value_min': -1.,
                                    'value_max': 1.,
                                    }
    inst.meta['unit_zon_ecef_z'] = {'long_name': 'Zonal unit vector along ECEF-z',
                                    'desc': 'Zonal unit vector along ECEF-z',
                                    'label': 'Zonal unit vector along ECEF-z',
                                    'notes': ('Unit vector expressed using Earth Centered Earth Fixed (ECEF) frame. '
                                              'Vector system is calcluated by field-line tracing along IGRF values '
                                              'down to reference altitudes of 120 km in both the Northern and Southern '
                                              'hemispheres. These two points, along with the satellite position, are '
                                              'used to define the magnetic meridian. Vector math from here generates '
                                              'the orthogonal system.'),
                                    'axis': 'Zonal unit vector along ECEF-z',
                                    'value_min': -1.,
                                    'value_max': 1.,
                                    }

    inst.meta['unit_fa_ecef_x'] = {'long_name': 'Field-aligned unit vector along ECEF-x',
                                    'desc': 'Field-aligned unit vector along ECEF-x',
                                    'label': 'Field-aligned unit vector along ECEF-x',
                                    'notes': ('Unit vector expressed using Earth Centered Earth Fixed (ECEF) frame. '
                                              'Vector system is calcluated by field-line tracing along IGRF values '
                                              'down to reference altitudes of 120 km in both the Northern and Southern '
                                              'hemispheres. These two points, along with the satellite position, are '
                                              'used to define the magnetic meridian. Vector math from here generates '
                                              'the orthogonal system.'),
                                    'axis': 'Field-aligned unit vector along ECEF-x',
                                    'value_min': -1.,
                                    'value_max': 1.,
                                    }
    inst.meta['unit_fa_ecef_y'] = {'long_name': 'Field-aligned unit vector along ECEF-y',
                                    'desc': 'Field-aligned unit vector along ECEF-y',
                                    'label': 'Field-aligned unit vector along ECEF-y',
                                    'notes': ('Unit vector expressed using Earth Centered Earth Fixed (ECEF) frame. '
                                              'Vector system is calcluated by field-line tracing along IGRF values '
                                              'down to reference altitudes of 120 km in both the Northern and Southern '
                                              'hemispheres. These two points, along with the satellite position, are '
                                              'used to define the magnetic meridian. Vector math from here generates '
                                              'the orthogonal system.'),
                                    'axis': 'Field-aligned unit vector along ECEF-y',
                                    'value_min': -1.,
                                    'value_max': 1.,
                                    }
    inst.meta['unit_fa_ecef_z'] = {'long_name': 'Field-aligned unit vector along ECEF-z',
                                    'desc': 'Field-aligned unit vector along ECEF-z',
                                    'label': 'Field-aligned unit vector along ECEF-z',
                                    'notes': ('Unit vector expressed using Earth Centered Earth Fixed (ECEF) frame. '
                                              'Vector system is calcluated by field-line tracing along IGRF values '
                                              'down to reference altitudes of 120 km in both the Northern and Southern '
                                              'hemispheres. These two points, along with the satellite position, are '
                                              'used to define the magnetic meridian. Vector math from here generates '
                                              'the orthogonal system.'),
                                    'axis': 'Field-aligned unit vector along ECEF-z',
                                    'value_min': -1.,
                                    'value_max': 1.,
                                    }

    inst.meta['unit_mer_ecef_x'] = {'long_name': 'Meridional unit vector along ECEF-x',
                                    'desc': 'Meridional unit vector along ECEF-x',
                                    'label': 'Meridional unit vector along ECEF-x',
                                    'notes': ('Unit vector expressed using Earth Centered Earth Fixed (ECEF) frame. '
                                              'Vector system is calcluated by field-line tracing along IGRF values '
                                              'down to reference altitudes of 120 km in both the Northern and Southern '
                                              'hemispheres. These two points, along with the satellite position, are '
                                              'used to define the magnetic meridian. Vector math from here generates '
                                              'the orthogonal system.'),
                                    'axis': 'Meridional unit vector along ECEF-x',
                                    'value_min': -1.,
                                    'value_max': 1.,
                                    }
    inst.meta['unit_mer_ecef_y'] = {'long_name': 'Meridional unit vector along ECEF-y',
                                    'desc': 'Meridional unit vector along ECEF-y',
                                    'label': 'Meridional unit vector along ECEF-y',
                                    'notes': ('Unit vector expressed using Earth Centered Earth Fixed (ECEF) frame. '
                                              'Vector system is calcluated by field-line tracing along IGRF values '
                                              'down to reference altitudes of 120 km in both the Northern and Southern '
                                              'hemispheres. These two points, along with the satellite position, are '
                                              'used to define the magnetic meridian. Vector math from here generates '
                                              'the orthogonal system.'),
                                    'axis': 'Meridional unit vector along ECEF-y',
                                    'value_min': -1.,
                                    'value_max': 1.,
                                    }
    inst.meta['unit_mer_ecef_z'] = {'long_name': 'Meridional unit vector along ECEF-z',
                                    'desc': 'Meridional unit vector along ECEF-z',
                                    'label': 'Meridional unit vector along ECEF-z',
                                    'notes': ('Unit vector expressed using Earth Centered Earth Fixed (ECEF) frame. '
                                              'Vector system is calcluated by field-line tracing along IGRF values '
                                              'down to reference altitudes of 120 km in both the Northern and Southern '
                                              'hemispheres. These two points, along with the satellite position, are '
                                              'used to define the magnetic meridian. Vector math from here generates '
                                              'the orthogonal system.'),
                                    'axis': 'Meridional unit vector along ECEF-z',
                                    'value_min': -1.,
                                    'value_max': 1.,
                                    }

    return


def add_mag_drift_unit_vectors(inst, max_steps=40000, step_size=0.5,
                               method='auto'):
    """Add unit vectors expressing the ion drift coordinate system
    organized by the geomagnetic field. Unit vectors are expressed
    in S/C coordinates.
    
    Interally, routine calls add_mag_drift_unit_vectors_ecef. 
    See function for input parameter description.
    Requires the orientation of the S/C basis vectors in ECEF using naming,
    'sc_xhat_x' where *hat (*=x,y,z) is the S/C basis vector and _* (*=x,y,z)
    is the ECEF direction. 
    
    Parameters
    ----------
    inst : pysat.Instrument object
        Instrument object to be modified
    max_steps : int
        Maximum number of steps taken for field line integration
    step_size : float
        Maximum step size (km) allowed for field line tracer
    method : see add_mag_drift_unit_vectors_ecef
    
    Returns
    -------
    None
        Modifies instrument object in place. Adds 'unit_zon_*' where * = x,y,z
        'unit_fa_*' and 'unit_mer_*' for zonal, field aligned, and meridional
        directions. Note that vector components are expressed in the S/C basis.
        
    """

    # vectors are returned in geo/ecef coordinate system
    add_mag_drift_unit_vectors_ecef(inst, max_steps=max_steps, step_size=step_size,
                                    method=method)
    # convert them to S/C using transformation supplied by OA
    inst['unit_zon_x'], inst['unit_zon_y'], inst['unit_zon_z'] = project_ecef_vector_onto_basis(inst['unit_zon_ecef_x'], inst['unit_zon_ecef_y'], inst['unit_zon_ecef_z'],
                                                                                                inst['sc_xhat_x'], inst['sc_xhat_y'], inst['sc_xhat_z'],
                                                                                                inst['sc_yhat_x'], inst['sc_yhat_y'], inst['sc_yhat_z'],
                                                                                                inst['sc_zhat_x'], inst['sc_zhat_y'], inst['sc_zhat_z'])
    inst['unit_fa_x'], inst['unit_fa_y'], inst['unit_fa_z'] = project_ecef_vector_onto_basis(inst['unit_fa_ecef_x'], inst['unit_fa_ecef_y'], inst['unit_fa_ecef_z'],
                                                                                                inst['sc_xhat_x'], inst['sc_xhat_y'], inst['sc_xhat_z'],
                                                                                                inst['sc_yhat_x'], inst['sc_yhat_y'], inst['sc_yhat_z'],
                                                                                                inst['sc_zhat_x'], inst['sc_zhat_y'], inst['sc_zhat_z'])
    inst['unit_mer_x'], inst['unit_mer_y'], inst['unit_mer_z'] = project_ecef_vector_onto_basis(inst['unit_mer_ecef_x'], inst['unit_mer_ecef_y'], inst['unit_mer_ecef_z'],
                                                                                                inst['sc_xhat_x'], inst['sc_xhat_y'], inst['sc_xhat_z'],
                                                                                                inst['sc_yhat_x'], inst['sc_yhat_y'], inst['sc_yhat_z'],
                                                                                                inst['sc_zhat_x'], inst['sc_zhat_y'], inst['sc_zhat_z'])

    inst.meta['unit_zon_x'] = { 'long_name':'Zonal direction along IVM-x',
                                'desc': 'Unit vector for the zonal geomagnetic direction.',
                                'label': 'Zonal Unit Vector: IVM-X component',
                                'axis': 'Zonal Unit Vector: IVM-X component',
                                'notes': ('Positive towards the east. Zonal vector is normal to magnetic meridian plane. '
                                          'The unit vector is expressed in the IVM coordinate system, x - along RAM, '
                                          'z - towards nadir, y - completes the system, generally southward. '
                                          'Calculated using the corresponding unit vector in ECEF and the orientation '
                                          'of the IVM also expressed in ECEF (sc_*hat_*).'),
                                'scale': 'linear',
                                'units': '',
                               'value_min':-1., 
                               'value_max':1}
    inst.meta['unit_zon_y'] = {'long_name':'Zonal direction along IVM-y',
                                'desc': 'Unit vector for the zonal geomagnetic direction.',
                                'label': 'Zonal Unit Vector: IVM-Y component',
                                'axis': 'Zonal Unit Vector: IVM-Y component',
                                'notes': ('Positive towards the east. Zonal vector is normal to magnetic meridian plane. '
                                          'The unit vector is expressed in the IVM coordinate system, x - along RAM, '
                                          'z - towards nadir, y - completes the system, generally southward. '
                                          'Calculated using the corresponding unit vector in ECEF and the orientation '
                                          'of the IVM also expressed in ECEF (sc_*hat_*).'),
                                'scale': 'linear',
                                'units': '',
                               'value_min':-1., 
                               'value_max':1}
    inst.meta['unit_zon_z'] = {'long_name':'Zonal direction along IVM-z',
                                'desc': 'Unit vector for the zonal geomagnetic direction.',
                                'label': 'Zonal Unit Vector: IVM-Z component',
                                'axis': 'Zonal Unit Vector: IVM-Z component',
                                'notes': ('Positive towards the east. Zonal vector is normal to magnetic meridian plane. '
                                          'The unit vector is expressed in the IVM coordinate system, x - along RAM, '
                                          'z - towards nadir, y - completes the system, generally southward. '
                                          'Calculated using the corresponding unit vector in ECEF and the orientation '
                                          'of the IVM also expressed in ECEF (sc_*hat_*).'),
                                'scale': 'linear',
                                'units': '',
                               'value_min':-1., 
                               'value_max':1}

    inst.meta['unit_fa_x'] = {'long_name':'Field-aligned direction along IVM-x',
                                'desc': 'Unit vector for the geomagnetic field line direction.',
                                'label': 'Field Aligned Unit Vector: IVM-X component',
                                'axis': 'Field Aligned Unit Vector: IVM-X component',
                                'notes': ('Positive along the field, generally northward. Unit vector is along the geomagnetic field. '
                                          'The unit vector is expressed in the IVM coordinate system, x - along RAM, '
                                          'z - towards nadir, y - completes the system, generally southward. '
                                          'Calculated using the corresponding unit vector in ECEF and the orientation '
                                          'of the IVM also expressed in ECEF (sc_*hat_*).'),
                                'scale': 'linear',
                                'units': '',
                               'value_min':-1., 
                               'value_max':1}
    inst.meta['unit_fa_y'] = {'long_name':'Field-aligned direction along IVM-y',
                                'desc': 'Unit vector for the geomagnetic field line direction.',
                                'label': 'Field Aligned Unit Vector: IVM-Y component',
                                'axis': 'Field Aligned Unit Vector: IVM-Y component',
                                'notes': ('Positive along the field, generally northward. Unit vector is along the geomagnetic field. '
                                          'The unit vector is expressed in the IVM coordinate system, x - along RAM, '
                                          'z - towards nadir, y - completes the system, generally southward. '
                                          'Calculated using the corresponding unit vector in ECEF and the orientation '
                                          'of the IVM also expressed in ECEF (sc_*hat_*).'),
                                'scale': 'linear',
                                'units': '',
                               'value_min':-1., 
                               'value_max':1}
    inst.meta['unit_fa_z'] = {'long_name':'Field-aligned direction along IVM-z',
                                'desc': 'Unit vector for the geomagnetic field line direction.',
                                'label': 'Field Aligned Unit Vector: IVM-Z component',
                                'axis': 'Field Aligned Unit Vector: IVM-Z component',
                                'notes': ('Positive along the field, generally northward. Unit vector is along the geomagnetic field. '
                                          'The unit vector is expressed in the IVM coordinate system, x - along RAM, '
                                          'z - towards nadir, y - completes the system, generally southward. '
                                          'Calculated using the corresponding unit vector in ECEF and the orientation '
                                          'of the IVM also expressed in ECEF (sc_*hat_*).'),
                                'scale': 'linear',
                                'units': '',
                               'value_min':-1., 
                               'value_max':1}

    inst.meta['unit_mer_x'] = {'long_name':'Meridional direction along IVM-x',
                                'desc': 'Unit vector for the geomagnetic meridional direction.',
                                'label': 'Meridional Unit Vector: IVM-X component',
                                'axis': 'Meridional Unit Vector: IVM-X component',
                                'notes': ('Positive is aligned with vertical at '
                                          'geomagnetic equator. Unit vector is perpendicular to the geomagnetic field '
                                          'and in the plane of the meridian.'
                                          'The unit vector is expressed in the IVM coordinate system, x - along RAM, '
                                          'z - towards nadir, y - completes the system, generally southward. '
                                          'Calculated using the corresponding unit vector in ECEF and the orientation '
                                          'of the IVM also expressed in ECEF (sc_*hat_*).'),
                                'scale': 'linear',
                                'units': '',
                               'value_min':-1., 
                               'value_max':1}
    inst.meta['unit_mer_y'] = {'long_name':'Meridional direction along IVM-y',
                                'desc': 'Unit vector for the geomagnetic meridional direction.',
                                'label': 'Meridional Unit Vector: IVM-Y component',
                                'axis': 'Meridional Unit Vector: IVM-Y component',
                                'notes': ('Positive is aligned with vertical at '
                                          'geomagnetic equator. Unit vector is perpendicular to the geomagnetic field '
                                          'and in the plane of the meridian.'
                                          'The unit vector is expressed in the IVM coordinate system, x - along RAM, '
                                          'z - towards nadir, y - completes the system, generally southward. '
                                          'Calculated using the corresponding unit vector in ECEF and the orientation '
                                          'of the IVM also expressed in ECEF (sc_*hat_*).'),
                                'scale': 'linear',
                                'units': '',
                               'value_min':-1., 
                               'value_max':1}
    inst.meta['unit_mer_z'] = {'long_name':'Meridional direction along IVM-z',
                                'desc': 'Unit vector for the geomagnetic meridional direction.',
                                'label': 'Meridional Unit Vector: IVM-Z component',
                                'axis': 'Meridional Unit Vector: IVM-Z component',
                                'notes': ('Positive is aligned with vertical at '
                                          'geomagnetic equator. Unit vector is perpendicular to the geomagnetic field '
                                          'and in the plane of the meridian.'
                                          'The unit vector is expressed in the IVM coordinate system, x - along RAM, '
                                          'z - towards nadir, y - completes the system, generally southward. '
                                          'Calculated using the corresponding unit vector in ECEF and the orientation '
                                          'of the IVM also expressed in ECEF (sc_*hat_*).'),
                                'scale': 'linear',
                                'units': '',
                               'value_min':-1., 
                               'value_max':1}

    return


def add_mag_drifts(inst):
    """Adds ion drifts in magnetic coordinates using ion drifts in S/C coordinates
    along with pre-calculated unit vectors for magnetic coordinates.
    
    Note
    ----
        Requires ion drifts under labels 'iv_*' where * = (x,y,z) along with
        unit vectors labels 'unit_zonal_*', 'unit_fa_*', and 'unit_mer_*',
        where the unit vectors are expressed in S/C coordinates. These
        vectors are calculated by add_mag_drift_unit_vectors.
    
    Parameters
    ----------
    inst : pysat.Instrument
        Instrument object will be modified to include new ion drift magnitudes
        
    Returns
    -------
    None
        Instrument object modified in place
    
    """
    
    inst['iv_zon'] = {'data':inst['unit_zon_x'] * inst['iv_x'] + inst['unit_zon_y']*inst['iv_y'] + inst['unit_zon_z']*inst['iv_z'],
                      'units':'m/s',
                      'long_name':'Zonal ion velocity',
                      'notes':('Ion velocity relative to co-rotation along zonal '
                               'direction, normal to meridional plane. Positive east. '
                               'Velocity obtained using ion velocities relative '
                               'to co-rotation in the instrument frame along '
                               'with the corresponding unit vectors expressed in '
                               'the instrument frame. '),
                      'label': 'Zonal Ion Velocity',
                      'axis': 'Zonal Ion Velocity',
                      'desc': 'Zonal ion velocity',
                      'scale': 'Linear',
                      'value_min':-500., 
                      'value_max':500.}
                      
    inst['iv_fa'] = {'data':inst['unit_fa_x'] * inst['iv_x'] + inst['unit_fa_y'] * inst['iv_y'] + inst['unit_fa_z'] * inst['iv_z'],
                      'units':'m/s',
                      'long_name':'Field-Aligned ion velocity',
                      'notes':('Ion velocity relative to co-rotation along magnetic field line. Positive along the field. ',
                               'Velocity obtained using ion velocities relative '
                               'to co-rotation in the instrument frame along '
                               'with the corresponding unit vectors expressed in '
                               'the instrument frame. '),
                      'label':'Field-Aligned Ion Velocity',
                      'axis':'Field-Aligned Ion Velocity',
                      'desc':'Field-Aligned Ion Velocity',
                      'scale':'Linear',
                      'value_min':-500., 
                      'value_max':500.}

    inst['iv_mer'] = {'data':inst['unit_mer_x'] * inst['iv_x'] + inst['unit_mer_y']*inst['iv_y'] + inst['unit_mer_z']*inst['iv_z'],
                      'units':'m/s',
                      'long_name':'Meridional ion velocity',
                      'notes':('Velocity along meridional direction, perpendicular '
                               'to field and within meridional plane. Positive is up at magnetic equator. ',
                               'Velocity obtained using ion velocities relative '
                               'to co-rotation in the instrument frame along '
                               'with the corresponding unit vectors expressed in '
                               'the instrument frame. '),
                      'label':'Meridional Ion Velocity',
                      'axis':'Meridional Ion Velocity',
                      'desc':'Meridional Ion Velocity',
                      'scale':'Linear',
                      'value_min':-500., 
                      'value_max':500.}
    
    return


def add_footpoint_and_equatorial_drifts(inst, equ_mer_scalar='equ_mer_drifts_scalar',
                                              equ_zonal_scalar='equ_zon_drifts_scalar',
                                              north_mer_scalar='north_footpoint_mer_drifts_scalar',
                                              north_zon_scalar='north_footpoint_zon_drifts_scalar',
                                              south_mer_scalar='south_footpoint_mer_drifts_scalar',
                                              south_zon_scalar='south_footpoint_zon_drifts_scalar',
                                              mer_drift='iv_mer',
                                              zon_drift='iv_zon'):
    """Translates geomagnetic ion velocities to those at footpoints and magnetic equator.

    Note
    ----
        Presumes scalar values for mapping ion velocities are already in the inst, labeled
        by north_footpoint_zon_drifts_scalar, north_footpoint_mer_drifts_scalar,
        equ_mer_drifts_scalar, equ_zon_drifts_scalar.
    
        Also presumes that ion motions in the geomagnetic system are present and labeled
        as 'iv_mer' and 'iv_zon' for meridional and zonal ion motions.
        
        This naming scheme is used by the other pysat oriented routines
        in this package.
    
    Parameters
    ----------
    inst : pysat.Instrument
    equ_mer_scalar : string
        Label used to identify equatorial scalar for meridional ion drift
    equ_zon_scalar : string
        Label used to identify equatorial scalar for zonal ion drift
    north_mer_scalar : string
        Label used to identify northern footpoint scalar for meridional ion drift
    north_zon_scalar : string
        Label used to identify northern footpoint scalar for zonal ion drift
    south_mer_scalar : string
        Label used to identify northern footpoint scalar for meridional ion drift
    south_zon_scalar : string
        Label used to identify southern footpoint scalar for zonal ion drift
    mer_drift : string
        Label used to identify meridional ion drifts within inst
    zon_drift : string
        Label used to identify zonal ion drifts within inst
        
    Returns
    -------
    None
        Modifies pysat.Instrument object in place. Drifts mapped to the magnetic equator
        are labeled 'equ_mer_drift' and 'equ_zon_drift'. Mappings to the northern
        and southern footpoints are labeled 'south_footpoint_mer_drift' and
        'south_footpoint_zon_drift'. Similarly for the northern hemisphere.

    """

    inst['equ_mer_drift'] = {'data' : inst[equ_mer_scalar]*inst[mer_drift],
                            'units':'m/s',
                            'long_name':'Equatorial meridional ion velocity',
                            'notes':('Velocity along meridional direction, perpendicular '
                                    'to field and within meridional plane, scaled to '
                                    'magnetic equator. Positive is up at magnetic equator. '
                                    'Velocity obtained using ion velocities relative '
                                    'to co-rotation in the instrument frame along '
                                    'with the corresponding unit vectors expressed in '
                                    'the instrument frame. Field-line mapping and '
                                    'the assumption of equi-potential field lines '
                                    'is used to translate the locally measured ion '
                                    'motion to the magnetic equator. The mapping '
                                    'is used to determine the change in magnetic '
                                    'field line distance, which, under assumption of '
                                    'equipotential field lines, in turn alters '
                                    'the electric field at that location (E=V/d). '),
                            'label':'Equatorial Meridional Ion Velocity',
                            'axis':'Equatorial Meridional Ion Velocity',
                            'desc':'Equatorial Meridional Ion Velocity',
                            'scale':'Linear',
                            'value_min':-500., 
                            'value_max':500.}

    inst['equ_zon_drift'] = {'data' : inst[equ_zonal_scalar]*inst[zon_drift],
                            'units':'m/s',
                            'long_name':'Equatorial zonal ion velocity',
                            'notes':('Velocity along zonal direction, perpendicular '
                                    'to field and the meridional plane, scaled to '
                                    'magnetic equator. Positive is generally eastward. '
                                    'Velocity obtained using ion velocities relative '
                                    'to co-rotation in the instrument frame along '
                                    'with the corresponding unit vectors expressed in '
                                    'the instrument frame. Field-line mapping and '
                                    'the assumption of equi-potential field lines '
                                    'is used to translate the locally measured ion '
                                    'motion to the magnetic equator. The mapping '
                                    'is used to determine the change in magnetic '
                                    'field line distance, which, under assumption of '
                                    'equipotential field lines, in turn alters '
                                    'the electric field at that location (E=V/d). '),
                            'label':'Equatorial Zonal Ion Velocity',
                            'axis':'Equatorial Zonal Ion Velocity',
                            'desc':'Equatorial Zonal Ion Velocity',
                            'scale':'Linear',
                            'value_min':-500., 
                            'value_max':500.}

    inst['south_footpoint_mer_drift'] = {'data' : inst[south_mer_scalar]*inst[mer_drift],
                            'units':'m/s',
                            'long_name':'Southern meridional ion velocity',
                            'notes':('Velocity along meridional direction, perpendicular '
                                    'to field and within meridional plane, scaled to '
                                    'southern footpoint. Positive is up at magnetic equator. '
                                    'Velocity obtained using ion velocities relative '
                                    'to co-rotation in the instrument frame along '
                                    'with the corresponding unit vectors expressed in '
                                    'the instrument frame. Field-line mapping and '
                                    'the assumption of equi-potential field lines '
                                    'is used to translate the locally measured ion '
                                    'motion to the magnetic footpoint. The mapping '
                                    'is used to determine the change in magnetic '
                                    'field line distance, which, under assumption of '
                                    'equipotential field lines, in turn alters '
                                    'the electric field at that location (E=V/d). '),
                            'label':'Southern Meridional Ion Velocity',
                            'axis':'Southern Meridional Ion Velocity',
                            'desc':'Southern Meridional Ion Velocity',
                            'scale':'Linear',
                            'value_min':-500., 
                            'value_max':500.}

    inst['south_footpoint_zon_drift'] = {'data':inst[south_zon_scalar]*inst[zon_drift],
                            'units':'m/s',
                            'long_name':'Southern zonal ion velocity',
                            'notes':('Velocity along zonal direction, perpendicular '
                                    'to field and the meridional plane, scaled to '
                                    'southern footpoint. Positive is generally eastward. '
                                    'Velocity obtained using ion velocities relative '
                                    'to co-rotation in the instrument frame along '
                                    'with the corresponding unit vectors expressed in '
                                    'the instrument frame. Field-line mapping and '
                                    'the assumption of equi-potential field lines '
                                    'is used to translate the locally measured ion '
                                    'motion to the southern footpoint. The mapping '
                                    'is used to determine the change in magnetic '
                                    'field line distance, which, under assumption of '
                                    'equipotential field lines, in turn alters '
                                    'the electric field at that location (E=V/d). '),
                            'label':'Southern Zonal Ion Velocity',
                            'axis':'Southern Zonal Ion Velocity',
                            'desc':'Southern Zonal Ion Velocity',
                            'scale':'Linear',
                            'value_min':-500., 
                            'value_max':500.}

    inst['north_footpoint_mer_drift'] = {'data':inst[north_mer_scalar]*inst[mer_drift],
                            'units':'m/s',
                            'long_name':'Northern meridional ion velocity',
                            'notes':('Velocity along meridional direction, perpendicular '
                                    'to field and within meridional plane, scaled to '
                                    'northern footpoint. Positive is up at magnetic equator. '
                                    'Velocity obtained using ion velocities relative '
                                    'to co-rotation in the instrument frame along '
                                    'with the corresponding unit vectors expressed in '
                                    'the instrument frame. Field-line mapping and '
                                    'the assumption of equi-potential field lines '
                                    'is used to translate the locally measured ion '
                                    'motion to the magnetic footpoint. The mapping '
                                    'is used to determine the change in magnetic '
                                    'field line distance, which, under assumption of '
                                    'equipotential field lines, in turn alters '
                                    'the electric field at that location (E=V/d). '),
                            'label':'Northern Meridional Ion Velocity',
                            'axis':'Northern Meridional Ion Velocity',
                            'desc':'Northern Meridional Ion Velocity',
                            'scale':'Linear',
                            'value_min':-500., 
                            'value_max':500.}

    inst['north_footpoint_zon_drift'] = {'data':inst[north_zon_scalar]*inst[zon_drift],
                            'units':'m/s',
                            'long_name':'Northern zonal ion velocity',
                            'notes':('Velocity along zonal direction, perpendicular '
                                    'to field and the meridional plane, scaled to '
                                    'northern footpoint. Positive is generally eastward. '
                                    'Velocity obtained using ion velocities relative '
                                    'to co-rotation in the instrument frame along '
                                    'with the corresponding unit vectors expressed in '
                                    'the instrument frame. Field-line mapping and '
                                    'the assumption of equi-potential field lines '
                                    'is used to translate the locally measured ion '
                                    'motion to the northern footpoint. The mapping '
                                    'is used to determine the change in magnetic '
                                    'field line distance, which, under assumption of '
                                    'equipotential field lines, in turn alters '
                                    'the electric field at that location (E=V/d). '),
                            'label':'Northern Zonal Ion Velocity',
                            'axis':'Northern Zonal Ion Velocity',
                            'desc':'Northern Zonal Ion Velocity',
                            'scale':'Linear',
                            'value_min':-500., 
                            'value_max':500.}

def scalars_for_mapping_ion_drifts(glats, glons, alts, dates):
    """
    Calculates scalars for translating ion motions at position
    glat, glon, and alt, for date, to the footpoints of the field line
    as well as at the magnetic equator.
    
    All inputs are assumed to be 1D arrays.
    
    Note
    ----
        Directions refer to the ion motion direction e.g. the zonal
        scalar applies to zonal ion motions (meridional E field assuming ExB ion motion)
    
    Parameters
    ----------
    glats : list-like of floats (degrees)
        Geodetic (WGS84) latitude
    glons : list-like of floats (degrees)
        Geodetic (WGS84) longitude 
    alts : list-like of floats (km)
        Geodetic (WGS84) altitude, height above surface
    dates : list-like of datetimes
        Date and time for determination of scalars
        
    Returns
    -------
    dict
        array-like of scalars for translating electric field. Keys are,
        'north_zonal_drifts_scalar', 'north_mer_drifts_scalar', and similarly
        for southern locations. 'equator_mer_drifts_scalar' and 
        'equator_zonal_drifts_scalar' cover the mappings to the equator.
    
    """
    
    import pysat
    import pandas

    step_size = 0.5
    max_steps = 40000
    steps = np.arange(max_steps)

    ivm = pysat.Instrument('pysat', 'testing')

    # use spacecraft location to get ECEF
    ecef_xs, ecef_ys, ecef_zs = geodetic_to_ecef(glats, glons, alts)

    # prepare output
    north_ftpnt_zon_drifts_scalar = []
    south_ftpnt_zon_drifts_scalar = []
    north_ftpnt_mer_drifts_scalar = []
    south_ftpnt_mer_drifts_scalar = []
    eq_zon_drifts_scalar = []
    eq_mer_drifts_scalar = []

    for ecef_x, ecef_y, ecef_z, glat, glon, alt, date in zip(ecef_xs, ecef_ys, ecef_zs, 
                                                             glats, glons, alts, 
                                                             dates):
        ivm.date = date
        ivm.yr, ivm.doy = pysat.utils.getyrdoy(date)
        double_date = ivm.yr + ivm.doy / 366.

        # trace to northern footpoint
        sc_root = np.array([ecef_x, ecef_y, ecef_z])
        trace_north = field_line_trace(sc_root, double_date, 1., 120., steps=steps,
                                       step_size=step_size, max_steps=max_steps)
        # southern tracing
        trace_south = field_line_trace(sc_root, double_date, -1., 120., steps=steps,
                                       step_size=step_size, max_steps=max_steps)
        # footpoint location
        north_ftpnt = trace_north[-1, :]
        south_ftpnt = trace_south[-1, :]
        # convert to geodetic coordinates
        north_ftpnt = ecef_to_geodetic(*north_ftpnt)
        south_ftpnt = ecef_to_geodetic(*south_ftpnt)

        # add magnetic unit vectors
        frame_input = {}
        frame_input['latitude'] = [north_ftpnt[0], south_ftpnt[0], glat]
        frame_input['longitude'] = [north_ftpnt[1], south_ftpnt[1], glon]
        frame_input['altitude'] = [north_ftpnt[2], south_ftpnt[2], alt]
        input_frame = pandas.DataFrame.from_records(frame_input)
        ivm.data = input_frame
        add_mag_drift_unit_vectors_ecef(ivm, ref_height=0.)

        # trace_north back in ECEF
        north_ftpnt = geodetic_to_ecef(*north_ftpnt)
        south_ftpnt = geodetic_to_ecef(*south_ftpnt)

        # take half step along + meridional direction
        unit_x = np.array([1, 0, 0])
        unit_y = np.array([0, 1, 0])
        unit_z = np.array([0, 0, 1])
        # take step
        north_plus_mer = north_ftpnt + 25. * unit_x * ivm[0, 'unit_mer_ecef_x'] + 25. * unit_y * ivm[0, 'unit_mer_ecef_y'] + 25. * unit_z * ivm[0, 'unit_mer_ecef_z']
        # trace this back to southern footpoint
        trace_south_plus_mer = field_line_trace(north_plus_mer, double_date, -1., 0., steps=steps,
                                                step_size=step_size, max_steps=max_steps)

        # take half step from northern along - meridional direction
        # take step
        north_minus_mer = north_ftpnt - 25. * unit_x * ivm[0, 'unit_mer_ecef_x'] - 25. * unit_y * ivm[0, 'unit_mer_ecef_y'] - 25. * unit_z * ivm[0, 'unit_mer_ecef_z']
        # trace this back to southern footpoint
        trace_south_minus_mer = field_line_trace(north_minus_mer, double_date, -1., 0., steps=steps,
                                                 step_size=step_size, max_steps=max_steps)

        # take half step from S/C along + meridional direction
        sc_plus_mer = sc_root + 25. * unit_x * ivm[2, 'unit_mer_ecef_x'] + 25. * unit_y * ivm[2, 'unit_mer_ecef_y'] + 25. * unit_z * ivm[2, 'unit_mer_ecef_z']
        # find closest point along + field line trace
        diff_plus_mer = trace_south_plus_mer - sc_plus_mer
        diff_plus_mer_mag = np.sqrt((diff_plus_mer ** 2).sum(axis=1))
        min_plus_idx = np.argmin(diff_plus_mer_mag)
        # calculate distance of that point from S/C location
        pos_mer_step_size = (trace_south_plus_mer[min_plus_idx, :] - sc_root) ** 2
        pos_mer_step_size = np.sqrt(pos_mer_step_size.sum())

        # take half step from S/C along - meridional direction (may not reach field line trace)
        sc_minus_mer = sc_root - 25. * unit_x * ivm[2, 'unit_mer_ecef_x'] - 25. * unit_y * ivm[2, 'unit_mer_ecef_y'] - 25. * unit_z * ivm[2, 'unit_mer_ecef_z']
        # find closest point along - field line trace
        diff_minus_mer = trace_south_minus_mer - sc_minus_mer
        diff_minus_mer_mag = np.sqrt((diff_minus_mer ** 2).sum(axis=1))
        min_minus_idx = np.argmin(diff_minus_mer_mag)
        # calculate distance of that point from S/C location
        minus_mer_step_size = (trace_south_minus_mer[min_minus_idx, :] - sc_root) ** 2
        minus_mer_step_size = np.sqrt(minus_mer_step_size.sum())

        # scalar for the northern footpoint
        full_mer_sc_step = pos_mer_step_size + minus_mer_step_size
        north_ftpnt_zon_drifts_scalar.append((full_mer_sc_step) / 50.)

        # calculate scalar for the equator
        # get furthest point for both pos and minus traces, get distance
        max_plus_idx = np.argmax(np.sqrt((trace_south_plus_mer ** 2).sum(axis=1)))
        max_minus_idx = np.argmax(np.sqrt((trace_south_minus_mer ** 2).sum(axis=1)))
        step_zon_apex = np.sqrt(
            ((trace_south_plus_mer[max_plus_idx, :] - trace_south_minus_mer[max_minus_idx, :]) ** 2).sum())
        eq_zon_drifts_scalar.append(full_mer_sc_step / step_zon_apex)

        # Now it is time to do the same calculation for the southern footpoint
        # scalar
        # take step
        south_plus_mer = south_ftpnt + 25. * unit_x * ivm[1, 'unit_mer_ecef_x'] + 25. * unit_y * ivm[1, 'unit_mer_ecef_y'] + 25. * unit_z * ivm[1, 'unit_mer_ecef_z']
        # trace this back to northern footpoint
        trace_north_plus_mer = field_line_trace(south_plus_mer, double_date, 1., 0., steps=steps,
                                                step_size=step_size, max_steps=max_steps)

        # take half step from southern along - meridional direction
        # take step
        south_minus_mer = south_ftpnt - 25. * unit_x * ivm[1, 'unit_mer_ecef_x'] - 25. * unit_y * ivm[1, 'unit_mer_ecef_y'] - 25. * unit_z * ivm[1, 'unit_mer_ecef_z']
        # trace this back to northern footpoint
        trace_north_minus_mer = field_line_trace(south_minus_mer, double_date, 1., 0., steps=steps,
                                                 step_size=step_size, max_steps=max_steps)

        # take half step from S/C along + meridional direction
        sc_plus_mer = sc_root + 25. * unit_x * ivm[2, 'unit_mer_ecef_x'] + 25. * unit_y * ivm[2, 'unit_mer_ecef_y'] + 25. * unit_z * ivm[2, 'unit_mer_ecef_z']
        # find closest point along + field line trace
        diff_plus_mer = trace_north_plus_mer - sc_plus_mer
        diff_plus_mer_mag = np.sqrt((diff_plus_mer ** 2).sum(axis=1))
        min_plus_idx = np.argmin(diff_plus_mer_mag)
        # calculate distance of that point from S/C location
        pos_mer_step_size = (trace_north_plus_mer[min_plus_idx, :] - sc_root) ** 2
        pos_mer_step_size = np.sqrt(pos_mer_step_size.sum())

        # take half step from S/C along - meridional direction (may not reach field line trace)
        sc_minus_mer = sc_root - 25. * unit_x * ivm[2, 'unit_mer_ecef_x'] - 25. * unit_y * ivm[2, 'unit_mer_ecef_y'] - 25. * unit_z * ivm[2, 'unit_mer_ecef_z']
        # find closest point along - field line trace
        diff_minus_mer = trace_north_minus_mer - sc_minus_mer
        diff_minus_mer_mag = np.sqrt((diff_minus_mer ** 2).sum(axis=1))
        min_minus_idx = np.argmin(diff_minus_mer_mag)
        # calculate distance of that point from S/C location
        minus_mer_step_size = (trace_north_minus_mer[min_minus_idx, :] - sc_root) ** 2
        minus_mer_step_size = np.sqrt(minus_mer_step_size.sum())

        # scalar for the southern footpoint
        south_ftpnt_zon_drifts_scalar.append((pos_mer_step_size + minus_mer_step_size) / 50.)

        #############
        # Time for scaling in the meridional drifts, the zonal electric field
        #############

        # take half step along + zonal direction
        north_plus_zon = north_ftpnt + 25. * unit_x * ivm[0, 'unit_zon_ecef_x'] + 25. * unit_y * ivm[0, 'unit_zon_ecef_y'] + 25. * unit_z * ivm[0, 'unit_zon_ecef_z']
        # trace this back to southern footpoint
        trace_south_plus_zon = field_line_trace(north_plus_zon, double_date, -1., 0., steps=steps,
                                                step_size=step_size, max_steps=max_steps)

        # take half step from northern along - zonal direction
        # take step
        north_minus_zon = north_ftpnt - 25. * unit_x * ivm[0, 'unit_zon_ecef_x'] - 25. * unit_y * ivm[0, 'unit_zon_ecef_y'] - 25. * unit_z * ivm[0, 'unit_zon_ecef_z']
        # trace this back to southern footpoint
        trace_south_minus_zon = field_line_trace(north_minus_zon, double_date, -1., 0., steps=steps,
                                                 step_size=step_size, max_steps=max_steps)

        # take half step from S/C along + zonal direction
        sc_plus_zon = sc_root + 25. * unit_x * ivm[2, 'unit_zon_ecef_x'] + 25. * unit_y * ivm[2, 'unit_zon_ecef_y'] + 25. * unit_z * ivm[2, 'unit_zon_ecef_z']
        # find closest point along + field line trace
        diff_plus_zon = trace_south_plus_zon - sc_plus_zon
        diff_plus_zon_mag = np.sqrt((diff_plus_zon ** 2).sum(axis=1))
        min_plus_idx = np.argmin(diff_plus_zon_mag)
        # calculate distance of that point from S/C location
        pos_zon_step_size = (trace_south_plus_zon[min_plus_idx, :] - sc_root) ** 2
        pos_zon_step_size = np.sqrt(pos_zon_step_size.sum())

        # take half step from S/C along - zonal direction (may not reach field line trace)
        sc_minus_zon = sc_root - 25. * unit_x * ivm[2, 'unit_zon_ecef_x'] - 25. * unit_y * ivm[2, 'unit_zon_ecef_y'] - 25. * unit_z * ivm[2, 'unit_zon_ecef_z']
        # find closest point along - field line trace
        diff_minus_zon = trace_south_minus_zon - sc_minus_zon
        diff_minus_zon_mag = np.sqrt((diff_minus_zon ** 2).sum(axis=1))
        min_minus_idx = np.argmin(diff_minus_zon_mag)
        # calculate distance of that point from S/C location
        minus_zon_step_size = (trace_south_minus_zon[min_minus_idx, :] - sc_root) ** 2
        minus_zon_step_size = np.sqrt(minus_zon_step_size.sum())

        # scalar for the northern footpoint
        full_zonal_sc_step = pos_zon_step_size + minus_zon_step_size
        north_ftpnt_mer_drifts_scalar.append((full_zonal_sc_step) / 50.)

        # calculate scalar for the equator
        # get furthest point for both pos and minus traces, get distance
        max_plus_idx = np.argmax(np.sqrt((trace_south_plus_zon ** 2).sum(axis=1)))
        max_minus_idx = np.argmax(np.sqrt((trace_south_minus_zon ** 2).sum(axis=1)))
        step_zon_apex = np.sqrt(
            ((trace_south_plus_zon[max_plus_idx, :] - trace_south_minus_zon[max_minus_idx, :]) ** 2).sum())
        eq_mer_drifts_scalar.append(full_zonal_sc_step / step_zon_apex)

        # Now it is time to do the same calculation for the southern footpoint
        # scalar
        # take step
        south_plus_zon = south_ftpnt + 25. * unit_x * ivm[1, 'unit_zon_ecef_x'] + 25. * unit_y * ivm[1, 'unit_zon_ecef_y'] + 25. * unit_z * ivm[1, 'unit_zon_ecef_z']
        # trace this back to northern footpoint
        trace_north_plus_zon = field_line_trace(south_plus_zon, double_date, 1., 0., steps=steps,
                                                step_size=step_size, max_steps=max_steps)

        # take half step from southern along - zonal direction
        # take step
        south_minus_zon = south_ftpnt - 25. * unit_x * ivm[1, 'unit_zon_ecef_x'] - 25. * unit_y * ivm[1, 'unit_zon_ecef_y'] - 25. * unit_z * ivm[1, 'unit_zon_ecef_z']
        # trace this back to northern footpoint
        trace_north_minus_zon = field_line_trace(south_minus_zon, double_date, 1., 0., steps=steps,
                                                 step_size=step_size, max_steps=max_steps)

        # take half step from S/C along + zonal direction
        sc_plus_zon = sc_root + 25. * unit_x * ivm[2, 'unit_zon_ecef_x'] + 25. * unit_y * ivm[2, 'unit_zon_ecef_y'] + 25. * unit_z * ivm[2, 'unit_zon_ecef_z']
        # find closest point along + field line trace
        diff_plus_zon = trace_north_plus_zon - sc_plus_zon
        diff_plus_zon_mag = np.sqrt((diff_plus_zon ** 2).sum(axis=1))
        min_plus_idx = np.argmin(diff_plus_zon_mag)
        # calculate distance of that point from S/C location
        pos_zon_step_size = (trace_north_plus_zon[min_plus_idx, :] - sc_root) ** 2
        pos_zon_step_size = np.sqrt(pos_zon_step_size.sum())

        # take half step from S/C along - zonal direction (may not reach field line trace)
        sc_minus_zon = sc_root - 25. * unit_x * ivm[2, 'unit_zon_ecef_x'] - 25. * unit_y * ivm[2, 'unit_zon_ecef_y'] - 25. * unit_z * ivm[2, 'unit_zon_ecef_z']
        # find closest point along - field line trace
        diff_minus_zon = trace_north_minus_zon - sc_minus_zon
        diff_minus_zon_mag = np.sqrt((diff_minus_zon ** 2).sum(axis=1))
        min_minus_idx = np.argmin(diff_minus_zon_mag)
        # calculate distance of that point from S/C location
        minus_zon_step_size = (trace_north_minus_zon[min_minus_idx, :] - sc_root) ** 2
        minus_zon_step_size = np.sqrt(minus_zon_step_size.sum())

        # scalar for the southern footpoint
        south_ftpnt_mer_drifts_scalar.append((pos_zon_step_size + minus_zon_step_size) / 50.)


        # print north_ftpnt_zon_drifts_scalar, south_ftpnt_zon_drifts_scalar, north_ftpnt_mer_drifts_scalar, south_ftpnt_mer_drifts_scalar, eq_zon_drifts_scalar, eq_mer_drifts_scalar

    out = {}
    out['north_zonal_drifts_scalar'] = north_ftpnt_zon_drifts_scalar
    out['south_zonal_drifts_scalar'] = south_ftpnt_zon_drifts_scalar
    out['north_mer_drifts_scalar'] = north_ftpnt_mer_drifts_scalar
    out['south_mer_drifts_scalar'] = south_ftpnt_mer_drifts_scalar
    out['equator_zonal_drifts_scalar'] = eq_zon_drifts_scalar
    out['equator_mer_drifts_scalar'] = eq_mer_drifts_scalar

    return out
