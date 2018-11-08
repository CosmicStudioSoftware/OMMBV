
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
import pysat
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
                     max_steps=1E5, step_size=10., recursive_loop_count=None, recurse=True):
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
    
    if recursive_loop_count is None:  
        recursive_loop_count = 0
    #     
    if steps is None:
        steps = np.arange(max_steps)
    if not isinstance(date, float):
        # recast from datetime to float, as required by IGRF12 code
        doy = (date - datetime.datetime(date.year,1,1)).days
        # number of days in year, works for leap years
        num_doy_year = (datetime.datetime(date.year+1,1,1) - datetime.datetime(date.year,1,1)).days
        date = float(date.year) + float(doy)/float(num_doy_year) + float(date.hour + date.minute/60. + date.second/3600.)/24.
          
    trace_north = scipy.integrate.odeint(igrf.igrf_step, init.copy(),
                                         steps,
                                         args=(date, step_size, direction, height),
                                         full_output=False,
                                         printmessg=False,
                                         ixpr=False) #,
                                         # mxstep=500)
    
    # check that we reached final altitude
    check = trace_north[-1, :]
    x, y, z = ecef_to_geodetic(*check)        
    if height == 0:
        check_height = 1.
    else:
        check_height = height
    # fortran integration gets close to target height        
    if recurse & (z > check_height*1.000001):
        if (recursive_loop_count < 1000):
            # When we have not reached the reference height, call field_line_trace 
            # again by taking check value as init - recursive call
            recursive_loop_count = recursive_loop_count + 1
            trace_north1 = field_line_trace(check, date, direction, height,
                                                step_size=step_size, max_steps=max_steps,
                                                recursive_loop_count=recursive_loop_count,
                                                steps=steps)
        else:
            raise RuntimeError("After 100 iterations couldn't reach target altitude")
        return np.vstack((trace_north, trace_north1))
    else:
        # return results if we make it to the target altitude
        # filter points to terminate at point closest to target height
        # x, y, z = ecef_to_geodetic(trace_north[:,0], trace_north[:,1], trace_north[:,2]) 
        # idx = np.argmin(np.abs(check_height - z)) 
        return trace_north #[:idx+1,:]
    
def calculate_mag_drift_unit_vectors_ecef(latitude, longitude, altitude, datetimes,
                                          steps=None, max_steps=10000, step_size=10.,
                                          ref_height=120.):
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
    ref_height : float
        Altitude used as cutoff for labeling a field line location a footpoint
        
    Returns
    -------
    zon_x, zon_y, zon_z, fa_x, fa_y, fa_z, mer_x, mer_y, mer_z
            
    """

    if steps is None:
        steps = np.arange(max_steps)
    # calculate satellite position in ECEF coordinates
    ecef_x, ecef_y, ecef_z = geodetic_to_ecef(latitude, longitude, altitude)
    # also get position in geocentric coordinates
    geo_lat, geo_long, geo_alt = ecef_to_geocentric(ecef_x, ecef_y, ecef_z, ref_height=0.)
    idx, = np.where(geo_long < 0)
    geo_long[idx] = geo_long[idx] + 360.
    
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
        # tbn, tbe, tbd, tbmag are in nT
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
    # norm_north = np.sqrt(zvx_north ** 2 + zvy_north ** 2 + zvz_north ** 2)
    # norm_south = np.sqrt(zvx_south ** 2 + zvy_south ** 2 + zvz_south ** 2)

    # calculate zonal vector
    zvx = zvx_foot / norm_foot
    zvy = zvy_foot / norm_foot
    zvz = zvz_foot / norm_foot
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


def intersection_field_line_and_unit_vector_projection(pos, field_line, sign, time, direction=None,
                                                       step_size_goal=1.):   
    """Starting at pos, method steps along magnetic unit vector direction towards the supplied 
    field line trace. Determines the distance of closest approach to field line.
    
    Routine is used when calculting the mapping of electric fields along magnetic field 
    lines. Voltage remains constant along the field but the distance between field lines does not.
    This routine may be used to form the last leg when trying to trace out a closed field line loop.
    
    Routine will create a high resolution field line trace (.01 km step size) near the location
    of closest approach to better determine where the intersection occurs. This centered segment is 
    40 km long, thus the input field_line trace should have a maximum step size of 20 km.
    
    Parameters
    ----------
    pos : array-like
        X, Y, and Z ECEF locations to start from
    field_line : array-like (:,3)
        X, Y, and Z ECEF locations of field line trace, produced by the
        field_line_trace method.
    sign : int
        if 1, move along positive unit vector. Negwtive direction for -1.
    time : datetime or float
        Date to perform tracing on (year + day/365 + hours/24. + etc.)
        Accounts for leap year if datetime provided.
    direction : string ('meridional', 'zonal', or 'aligned')
        Which unit vector direction to move slong when trying to intersect
        with supplied field line trace. See step_along_mag_unit_vector method
        for more.
    step_size_goal : float
        step size goal that method will try to match when stepping towards field line. 
    
    Returns
    -------
    (float, array, float)
        Total distance taken along vector direction; the position after taking the step [x, y, z] in ECEF;
        distance of closest approach from input pos towards the input field line trace.
         
    """ 
                                                         
    # simple things first
    # take distance for all points from test
    # then keep search down, moving along projection line

    # work on a copy, probably not needed
    field_copy = field_line.copy()
    # set a high last minimum distance to ensure first loop does better than this
    last_min_dist = 2500000.
    # scalar is the distance along unit vector line that we are taking
    scalar = 0.
    # repeat boolean
    repeat=True
    # first run boolean
    first=True
    # factor is a divisor applied to the remaining distance between point and field line
    # I slowly take steps towards the field line and I don't want to overshoot
    # each time my minimum distance increases, I step back, increase factor, reducing
    # my next step size, then I try again
    factor = 1
    while repeat:
        # take a total step along magnetic unit vector
        # try to take steps near user provided step_size_goal
        unit_steps = scalar//step_size_goal
        if unit_steps == 0:
            unit_steps = 1
        pos_step = step_along_mag_unit_vector(pos[0], pos[1], pos[2], time, direction=direction,
                                              num_steps=unit_steps, step_size=scalar/unit_steps) 
        # find closest point along + field line trace
        diff = field_copy - pos_step
        diff_mag = np.sqrt((diff ** 2).sum(axis=1))
        min_idx = np.argmin(diff_mag)
        if first:
            # first time in while loop, create some information
            # make a high resolution field line trace around closest distance
            init = field_copy[min_idx,:]
            high_res_trace = field_line_trace(init, time, 1., 0.,
                                                    step_size=0.01, max_steps=2000,
                                                    recurse=False)
            high_res_trace2 = field_line_trace(init, time, -1., 0.,
                                                    step_size=0.01, max_steps=2000,
                                                    recurse=False)
            # combine together for a complete trace
            field_copy = np.vstack((high_res_trace[::-1], high_res_trace2))
            # difference with position
            diff = field_copy - pos_step
            diff_mag = np.sqrt((diff ** 2).sum(axis=1))
            # find closest once
            min_idx = np.argmin(diff_mag)
            first = False
        # pull out distance of closest point 
        min_dist = diff_mag[min_idx]
        # check how the solution is doing
        # if well, add more distance to the total step and recheck if we get closer
        # if worse, step back and try a smaller step
        if min_dist > last_min_dist:
            # last step we took made the solution worse
            if factor > 4:
                # we've tried enough, stop looping
                repeat = False
                # undo increment to last total distance
                scalar = scalar - sign*last_min_dist/(2*factor)
                # calculate latest position
                pos_step = step_along_mag_unit_vector(pos[0], pos[1], pos[2], time, direction=direction,
                                        num_steps=unit_steps, step_size=scalar/unit_steps) 
            else:
                # undo increment to last total distance
                scalar = scalar - sign*last_min_dist/(2*factor)
                # increase the divisor used to reduce the distance actually stepped per increment
                factor = factor + 1.
                # try a new increment to total distance
                scalar = scalar + sign*last_min_dist/(2*factor)
        else:
            # increment scalar, but only by a fraction
            scalar = scalar + sign*min_dist/(2*factor)
            # we have a new standard to judge against, set it
            last_min_dist = min_dist.copy()

    # return magnitude of step
    return scalar/sign, pos_step, min_dist


def step_along_mag_unit_vector(x, y, z, date, direction=None, num_steps=1., step_size=25., scalar=1):
    """
    Move along 'lines' formed by following the magnetic unit vector directions.

    Moving along the field is effectively the same as a field line trace though
    extended movement along a field should use the specific field_line_trace method.
        
    
    Parameters
    ----------
    x : ECEF-x (km)
        Location to step from in ECEF (km). Scalar input.
    y : ECEF-y (km)
        Location to step from in ECEF (km). Scalar input.
    z : ECEF-z (km)
        Location to step from in ECEF (km). Scalar input.
    date : list-like of datetimes
        Date and time for magnetic field
    direction : string
        String identifier for which unit vector directino to move along.
        Supported inputs, 'meridional', 'zonal', 'aligned'
    num_steps : int
        Number of steps to take along unit vector direction
    step_size = float
        Distance taken for each step (km)
    scalar : int
        Scalar modifier for step size distance. Input a -1 to move along negative
        unit vector direction.
        
    Returns
    -------
    np.array
        [x, y, z] of ECEF location after taking num_steps along direction, each step_size long.
    
    """
    
    
    # set parameters for the field line tracing routines
    field_step_size = 100.
    field_max_steps = 1000
    field_steps = np.arange(field_max_steps)
    
    for i in np.arange(num_steps):
        # x, y, z in ECEF
        # convert to geodetic
        lat, lon, alt = ecef_to_geodetic(x, y, z)
        # get unit vector directions
        zvx, zvy, zvz, bx, by, bz, mx, my, mz = calculate_mag_drift_unit_vectors_ecef([lat], [lon], [alt], [date],
                                                        steps=field_steps, max_steps=field_max_steps, 
                                                        step_size=field_step_size, 
                                                        ref_height=0.)
        # pull out the direction we need
        if direction == 'meridional':
            ux, uy, uz = mx, my, mz
        elif direction == 'zonal':
            ux, uy, uz = zvx, zvy, zvz
        elif direction == 'aligned':
            ux, uy, uz = bx, by, bz
            
        # take steps along direction
        x = x + step_size*ux[0]*scalar
        y = y + step_size*uy[0]*scalar
        z = z + step_size*uz[0]*scalar
            
    return np.array([x, y, z])


def apex_location_info(glats, glons, alts, dates):
    """Determine the apex location for the field line passing through input point.
    
    Employs a two stage method. A broad step (10 km) field line trace spanning Northern/Southern
    footpoints is used to find the location with the largest geodetic (WGS84) height.
    A higher resolution trace (.1 km) is then used to get a better fix on this location.
    Greatest geodetic height is once again selected.
    
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
    (float, float, float, float, float, float)
        ECEF X, ECEF Y, ECEF Z, Geodetic Latitude (degrees), Geodetic Longitude (degrees), 
        Geodetic Altitude (km)
        
    """

    # use input location and convert to ECEF
    ecef_xs, ecef_ys, ecef_zs = geodetic_to_ecef(glats, glons, alts)

    step_size = 10.
    max_steps = 10000
    steps = np.arange(max_steps)

    # prepare output
    out_x = []
    out_y = []
    out_z = []

    for ecef_x, ecef_y, ecef_z, glat, glon, alt, date in zip(ecef_xs, ecef_ys, ecef_zs, 
                                                             glats, glons, alts, 
                                                             dates):
        # to get the apex location we need to do a field line trace
        # then find the highest point
        # trace north, then south, and combine
        trace_south = field_line_trace(np.array([ecef_x, ecef_y, ecef_z]), date, -1., 0., steps=steps,
                                                step_size=step_size, max_steps=max_steps)
        trace_north = field_line_trace(np.array([ecef_x, ecef_y, ecef_z]), date, 1., 0., steps=steps,
                                                step_size=step_size, max_steps=max_steps)
        trace = np.vstack((trace_north[::-1], trace_south))
        # convert all locations to geodetic coordinates
        tlat, tlon, talt = ecef_to_geodetic(trace[:,0], trace[:,1], trace[:,2])        
        # determine location that is highest with respect to the geodetic Earth
        max_idx = np.argmax(talt)

        # repeat using a high resolution trace
        trace_south = field_line_trace(np.array([trace[max_idx,0], trace[max_idx,1], trace[max_idx,2]]), date, -1., 0., 
                                                steps=np.arange(100),
                                                step_size=step_size/100., max_steps=100, recurse=False)
        trace_north = field_line_trace(np.array([trace[max_idx,0], trace[max_idx,1], trace[max_idx,2]]), date, 1., 0., 
                                                steps=np.arange(100),
                                                step_size=step_size/100., max_steps=100, recurse=False)
        trace = np.vstack((trace_north[::-1], trace_south))
        # convert all locations to geodetic coordinates
        tlat, tlon, talt = ecef_to_geodetic(trace[:,0], trace[:,1], trace[:,2])
        # determine location that is highest with respect to the geodetic Earth
        max_idx = np.argmax(talt)
        # collect outputs
        out_x.append(trace[max_idx,0])
        out_y.append(trace[max_idx,1])
        out_z.append(trace[max_idx,2])
        
    out_x = np.array(out_x)
    out_y = np.array(out_y)
    out_z = np.array(out_z)
    glat, glon, alt = ecef_to_geodetic(out_x, out_y, out_z)
    
    return out_x, out_y, out_z, glat, glon, alt


def scalars_for_mapping_ion_drifts(glats, glons, alts, dates, step_size=None, max_steps=None,
                                   e_field_scaling_only=False):
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
    e_field_scaling_only : boolean (False)
        If True, method only calculates the electric field scalar, ignoring 
        changes in magnitude of B. Note ion velocity related to E/B.
        
    Returns
    -------
    dict
        array-like of scalars for translating electric field. Keys are,
        'north_zonal_drifts_scalar', 'north_mer_drifts_scalar', and similarly
        for southern locations. 'equator_mer_drifts_scalar' and 
        'equator_zonal_drifts_scalar' cover the mappings to the equator.
    
    """
    
    import pandas

    if step_size is None:
        step_size = 10.
    if max_steps is None:
        max_steps = 1000
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
        # going to try and form close loops via field line integration
        # start at location of interest, map down to northern and southern footpoints
        # then take symmetric steps along meridional and zonal directions and trace back
        # from location of interest, step along field line directions until we intersect 
        # or hit the distance of closest approach to the return field line
        # with the known distances of footpoint steps, and the closet approach distance
        # we can determine the scalar mapping of one location to another
        
        ivm.date = date
        ivm.yr, ivm.doy = pysat.utils.getyrdoy(date)
        double_date = float(ivm.yr) + float(ivm.doy) / 366.

        print (glat, glon, alt)
        # get location of apex for s/c field line
        apex_x, apex_y, apex_z, apex_lat, apex_lon, apex_alt = apex_location_info([glat], [glon], [alt], [date])

        # trace to northern footpoint
        sc_root = np.array([ecef_x, ecef_y, ecef_z])
        trace_north = field_line_trace(sc_root, double_date, 1., 120., steps=steps,
                                       step_size=step_size, max_steps=max_steps)
        # southern tracing
        trace_south = field_line_trace(sc_root, double_date, -1., 120., steps=steps,
                                       step_size=step_size, max_steps=max_steps)
        # footpoint location
        north_ftpnt = trace_north[-1, :]
        nft_glat, nft_glon, nft_alt = ecef_to_geodetic(*north_ftpnt)
        south_ftpnt = trace_south[-1, :]
        sft_glat, sft_glon, sft_alt = ecef_to_geodetic(*north_ftpnt)

        # determine scalar for zonal ion drifts (meridional electric fields)
        # take step from northern footpoint along + meridional direction
        north_plus_mer = step_along_mag_unit_vector(north_ftpnt[0], north_ftpnt[1], north_ftpnt[2], 
                                                    date, direction='meridional')
        # trace this back to southern footpoint
        trace_south_plus_mer = field_line_trace(north_plus_mer, double_date, -1., 0., steps=steps,
                                                step_size=step_size, max_steps=max_steps)
        # take half step from northern along - meridional direction
        north_minus_mer = step_along_mag_unit_vector(north_ftpnt[0], north_ftpnt[1], north_ftpnt[2], date, 
                                                     direction='meridional', scalar=-1)
        # trace this back to southern footpoint
        trace_south_minus_mer = field_line_trace(north_minus_mer, double_date, -1., 0., steps=steps,
                                                 step_size=step_size, max_steps=max_steps)
        # need to determine where the intersection of field line coming back from north footpoint + mer is
        # in relation to the meridional direction from the s/c location. 
        pos_mer_step_size, _, _ = intersection_field_line_and_unit_vector_projection(sc_root,
                                                                                  trace_south_plus_mer,
                                                                                  1, date, direction='meridional')        
        # take half step from S/C along - meridional direction 
        minus_mer_step_size, _, _ = intersection_field_line_and_unit_vector_projection(sc_root,
                                                                                    trace_south_minus_mer,
                                                                                    -1, date, direction='meridional')
        # scalar for the northern footpoint electric field based on distances
        full_mer_sc_step = pos_mer_step_size + minus_mer_step_size
        if e_field_scaling_only:
            north_ftpnt_zon_drifts_scalar.append(full_mer_sc_step/50.)
        else:
            # for drift also need to include the magnetic field, drift = E/B
            tbn, tbe, tbd, b_sc = igrf.igrf12syn(0, double_date, 1, alt, np.deg2rad(90.-glat), np.deg2rad(glon))
            tbn, tbe, tbd, b_nft = igrf.igrf12syn(0, double_date, 1, nft_alt, np.deg2rad(90.-nft_glat), np.deg2rad(nft_glon))
            north_ftpnt_zon_drifts_scalar.append(full_mer_sc_step*b_sc/(50.*b_nft))
            

        # calculate zonal scalar to map ion drifts to the equator        
        # take step from s/c along + meridional direction
        # then get the apex location
        plus_mer = step_along_mag_unit_vector(ecef_x, ecef_y, ecef_z, date, direction='meridional')
        plus_mer_lat, plus_mer_lon, plus_mer_alt = ecef_to_geodetic(plus_mer[0], plus_mer[1], plus_mer[2])
        plus_apex_x, plus_apex_y, plus_apex_z, plus_apex_lat, plus_apex_lon, plus_apex_alt = \
                    apex_location_info([plus_mer_lat], [plus_mer_lon], [plus_mer_alt], [date])
        # take half step from s/c along - meridional direction
        # then get the apex location
        minus_mer = step_along_mag_unit_vector(ecef_x, ecef_y, ecef_z, date, direction='meridional', scalar=-1)
        minus_mer_lat, minus_mer_lon, minus_mer_alt = ecef_to_geodetic(minus_mer[0], minus_mer[1], minus_mer[2])
        minus_apex_x, minus_apex_y, minus_apex_z, minus_apex_lat, minus_apex_lon, minus_apex_alt = \
                    apex_location_info([minus_mer_lat], [minus_mer_lon], [minus_mer_alt], [date])
        # take difference in apex locations
        step_zon_apex = np.sqrt( (plus_apex_x-minus_apex_x)**2 + (plus_apex_y-minus_apex_y)**2 + (plus_apex_z-minus_apex_z)**2) 
        # compare difference in step sizes vs apex locations to get electric field scalar to the equator
        if e_field_scaling_only:
            eq_zon_drifts_scalar.append(50./step_zon_apex)
        else:
            # for drift also need to include the magnetic field, drift = E/B
            tbn, tbe, tbd, b_sc = igrf.igrf12syn(0, double_date, 1, alt, np.deg2rad(90.-glat), np.deg2rad(glon))
            tbn, tbe, tbd, b_eq = igrf.igrf12syn(0, double_date, 1, apex_alt, np.deg2rad(90.-apex_lat), np.deg2rad(apex_lon))
            eq_zon_drifts_scalar.append(50.*b_sc/(step_zon_apex*b_eq))
       

        # Now it is time to do the same calculation for the southern footpoint scalar
        south_plus_mer = step_along_mag_unit_vector(south_ftpnt[0], south_ftpnt[1], south_ftpnt[2], 
                                                    date, direction='meridional')
        # trace this back to northern footpoint
        trace_north_plus_mer = field_line_trace(south_plus_mer, double_date, 1., 0., steps=steps,
                                                step_size=step_size, max_steps=max_steps)
        # take half step from southern along - meridional direction
        south_minus_mer = step_along_mag_unit_vector(south_ftpnt[0], south_ftpnt[1], south_ftpnt[2], 
                                                     date, direction='meridional', scalar=-1)
        # trace this back to northern footpoint
        trace_north_minus_mer = field_line_trace(south_minus_mer, double_date, 1., 0., steps=steps,
                                                 step_size=step_size, max_steps=max_steps)
        pos_mer_step_size, _, _ = intersection_field_line_and_unit_vector_projection(sc_root, 
                                                                                    trace_north_plus_mer,
                                                                                    1, date, direction='meridional')
        minus_mer_step_size, _, _ = intersection_field_line_and_unit_vector_projection(sc_root, 
                                                                                    trace_north_minus_mer,
                                                                                    -1, date, direction='meridional')
        # scalar for the southern footpoint
        if e_field_scaling_only:
            south_ftpnt_zon_drifts_scalar.append((pos_mer_step_size + minus_mer_step_size)/ 50.)
        else:
            # for drift also need to include the magnetic field, drift = E/B
            tbn, tbe, tbd, b_sc = igrf.igrf12syn(0, double_date, 1, alt, np.deg2rad(90.-glat), np.deg2rad(glon))
            tbn, tbe, tbd, b_sft = igrf.igrf12syn(0, double_date, 1, sft_alt, np.deg2rad(90.-sft_glat), np.deg2rad(sft_glon))
            south_ftpnt_zon_drifts_scalar.append((pos_mer_step_size + minus_mer_step_size)*b_sc/(50.*b_sft))

        #############
        # Time for scaling in the meridional drifts, the zonal electric field
        #############

        # take half step along + zonal direction
        north_plus_zon = step_along_mag_unit_vector(north_ftpnt[0], north_ftpnt[1], north_ftpnt[2], date, direction='zonal')
        # trace this back to southern footpoint
        trace_south_plus_zon = field_line_trace(north_plus_zon, double_date, -1., 0., steps=steps,
                                                step_size=step_size, max_steps=max_steps)
        # take half step from northern along - zonal direction
        north_minus_zon = step_along_mag_unit_vector(north_ftpnt[0], north_ftpnt[1], north_ftpnt[2], date, direction='zonal', scalar=-1)
        # trace this back to southern footpoint
        trace_south_minus_zon = field_line_trace(north_minus_zon, double_date, -1., 0., steps=steps,
                                                 step_size=step_size, max_steps=max_steps)
        # get intersections
        pos_zon_step_size, _, _ = intersection_field_line_and_unit_vector_projection(sc_root,
                                                                                    trace_south_plus_zon,
                                                                                    1, date, direction='zonal')
        minus_zon_step_size, _, _ = intersection_field_line_and_unit_vector_projection(sc_root, 
                                                                                    trace_south_minus_zon,
                                                                                    -1, date, direction='zonal')
        # scalar for the northern footpoint
        full_zonal_sc_step = pos_zon_step_size + minus_zon_step_size
        if e_field_scaling_only:
            north_ftpnt_mer_drifts_scalar.append((full_zonal_sc_step) / 50.)
        else:
            # for drift also need to include the magnetic field, drift = E/B
            tbn, tbe, tbd, b_sc = igrf.igrf12syn(0, double_date, 1, alt, np.deg2rad(90.-glat), np.deg2rad(glon))
            tbn, tbe, tbd, b_nft = igrf.igrf12syn(0, double_date, 1, nft_alt, np.deg2rad(90.-nft_glat), np.deg2rad(nft_glon))
            north_ftpnt_mer_drifts_scalar.append(full_zonal_sc_step*b_sc/(50.*b_nft))
            

        # calculate meridional scalar to map ion drifts to the equator        
        # take step from s/c along + zonal direction
        # then get the apex location
        plus_zon = step_along_mag_unit_vector(ecef_x, ecef_y, ecef_z, date, direction='zonal')
        plus_zon_lat, plus_zon_lon, plus_zon_alt = ecef_to_geodetic(plus_zon[0], plus_zon[1], plus_zon[2])
        plus_apex_x, plus_apex_y, plus_apex_z, plus_apex_lat, plus_apex_lon, plus_apex_alt = apex_location_info([plus_zon_lat], [plus_zon_lon], [plus_zon_alt], [date])
        # take half step from s/c along - zonal direction
        # then get the apex location
        minus_zon = step_along_mag_unit_vector(ecef_x, ecef_y, ecef_z, date, direction='zonal', scalar=-1)
        minus_zon_lat, minus_zon_lon, minus_zon_alt = ecef_to_geodetic(minus_zon[0], minus_zon[1], minus_zon[2])
        minus_apex_x, minus_apex_y, minus_apex_z, minus_apex_lat, minus_apex_lon, minus_apex_alt = apex_location_info([minus_zon_lat], [minus_zon_lon], [minus_zon_alt], [date])
        # compare difference in step sizes vs apex locations to get electric field scalar to the equator
        step_zon_apex = np.sqrt( (plus_apex_x-minus_apex_x)**2 + (plus_apex_y-minus_apex_y)**2 + (plus_apex_z-minus_apex_z)**2) 

        if e_field_scaling_only:
            eq_mer_drifts_scalar.append(50./step_zon_apex)
        else:
            # for drift also need to include the magnetic field, drift = E/B
            tbn, tbe, tbd, b_sc = igrf.igrf12syn(0, double_date, 1, alt, np.deg2rad(90.-glat), np.deg2rad(glon))
            tbn, tbe, tbd, b_eq = igrf.igrf12syn(0, double_date, 1, apex_alt, np.deg2rad(90.-apex_lat), np.deg2rad(apex_lon))
            eq_mer_drifts_scalar.append(50.*b_sc/(step_zon_apex*b_eq))

        # Now it is time to do the same calculation for the southern footpoint
        south_plus_zon = step_along_mag_unit_vector(south_ftpnt[0], south_ftpnt[1], south_ftpnt[2], date, direction='zonal')
        # trace this back to northern footpoint
        trace_north_plus_zon = field_line_trace(south_plus_zon, double_date, 1., 0., steps=steps,
                                                step_size=step_size, max_steps=max_steps)
        # take half step from southern along - zonal direction
        south_minus_zon = step_along_mag_unit_vector(south_ftpnt[0], south_ftpnt[1], south_ftpnt[2], date, direction='zonal', scalar=-1)
        # trace this back to northern footpoint
        trace_north_minus_zon = field_line_trace(south_minus_zon, double_date, 1., 0., steps=steps,
                                                 step_size=step_size, max_steps=max_steps)
        # take half step from S/C along + zonal direction
        pos_zon_step_size, _, _ = intersection_field_line_and_unit_vector_projection(sc_root,  
                                                                                    trace_north_plus_zon,
                                                                                    1, date, direction='zonal')
        # take half step from S/C along - zonal direction
        minus_zon_step_size, _, _ = intersection_field_line_and_unit_vector_projection(sc_root, 
                                                                                    trace_north_minus_zon,
                                                                                    -1, date, direction='zonal')
        # scalar for the southern footpoint
        if e_field_scaling_only:
            south_ftpnt_mer_drifts_scalar.append((pos_zon_step_size + minus_zon_step_size) / 50.)
        else:
            # for drift also need to include the magnetic field, drift = E/B
            tbn, tbe, tbd, b_sc = igrf.igrf12syn(0, double_date, 1, alt, np.deg2rad(90.-glat), np.deg2rad(glon))
            tbn, tbe, tbd, b_sft = igrf.igrf12syn(0, double_date, 1, sft_alt, np.deg2rad(90.-sft_glat), np.deg2rad(sft_glon))
            south_ftpnt_mer_drifts_scalar.append((pos_zon_step_size + minus_zon_step_size)*b_sc/(50.*b_sft))

    out = {}
    out['north_zonal_drifts_scalar'] = north_ftpnt_zon_drifts_scalar
    out['south_zonal_drifts_scalar'] = south_ftpnt_zon_drifts_scalar
    out['north_mer_drifts_scalar'] = north_ftpnt_mer_drifts_scalar
    out['south_mer_drifts_scalar'] = south_ftpnt_mer_drifts_scalar
    out['equator_zonal_drifts_scalar'] = eq_zon_drifts_scalar
    out['equator_mer_drifts_scalar'] = eq_mer_drifts_scalar

    return out


