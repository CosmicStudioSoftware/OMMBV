"""
Supporting routines for coordinate conversions as well as vector operations and
transformations used in Space Science.
"""

import scipy
import scipy.integrate
import numpy as np
import datetime
import pysat
# import reference IGRF fortran code within the package
from OMMBV import igrf as igrf
import OMMBV.fortran_coords

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
    x = r*np.cos(np.deg2rad(latitude))*np.cos(np.deg2rad(longitude))
    y = r*np.cos(np.deg2rad(latitude))*np.sin(np.deg2rad(longitude))
    z = r*np.sin(np.deg2rad(latitude))

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

    r = np.sqrt(x**2 + y**2 + z**2)
    colatitude = np.rad2deg(np.arccos(z/r))
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

    ellip = np.sqrt(1. - earth_b**2/earth_a**2)
    r_n = earth_a/np.sqrt(1. - ellip**2*np.sin(np.deg2rad(latitude))**2)

    # colatitude = 90. - latitude
    x = (r_n + altitude)*np.cos(np.deg2rad(latitude))*np.cos(np.deg2rad(longitude))
    y = (r_n + altitude)*np.cos(np.deg2rad(latitude))*np.sin(np.deg2rad(longitude))
    z = (r_n*(1. - ellip**2) + altitude)*np.sin(np.deg2rad(latitude))

    return x, y, z

try:
    ecef_to_geodetic = OMMBV.fortran_coords.ecef_to_geodetic
except AttributeError:
    print('Unable to use Fortran version of ecef_to_geodetic. Please check installation.')

def python_ecef_to_geodetic(x, y, z, method=None):
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
    ellip = np.sqrt(1. - earth_b**2/earth_a**2)
    # first eccentricity squared
    e2 = ellip**2  # 6.6943799901377997E-3

    longitude = np.arctan2(y, x)
    # cylindrical radius
    p = np.sqrt(x**2 + y**2)

    # closed form solution
    # a source, http://www.epsg.org/Portals/0/373-07-2.pdf , page 96 section 2.2.1
    if method == 'closed':
        e_prime = np.sqrt((earth_a**2 - earth_b**2)/earth_b**2)
        theta = np.arctan2(z*earth_a, p*earth_b)
        latitude = np.arctan2(z + e_prime**2*earth_b*np.sin(theta)**3, p - e2*earth_a*np.cos(theta)**3)
        r_n = earth_a/np.sqrt(1. - e2*np.sin(latitude)**2)
        h = p/np.cos(latitude) - r_n

    # another possibility
    # http://ir.lib.ncku.edu.tw/bitstream/987654321/39750/1/3011200501001.pdf

    ## iterative method
    # http://www.oc.nps.edu/oc2902w/coord/coordcvt.pdf
    if method == 'iterative':
        latitude = np.arctan2(p, z)
        r_n = earth_a/np.sqrt(1. - e2*np.sin(latitude)**2)
        for i in np.arange(6):
            # print latitude
            r_n = earth_a/np.sqrt(1. - e2*np.sin(latitude)**2)
            h = p/np.cos(latitude) - r_n
            latitude = np.arctan(z/p/(1. - e2*(r_n/(r_n + h))))
            # print h
        # final ellipsoidal height update
        h = p/np.cos(latitude) - r_n

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
    up = x*np.cos(rlon)*np.cos(rlat) + y*np.sin(rlon)*np.cos(rlat) + z*np.sin(rlat)

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

    Returns
    -------
    x, y, z
        Vector projected onto new basis

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
                     max_steps=1E4, step_size=10., recursive_loop_count=None,
                     recurse=True, min_check_flag=False, stop_direction = 0):
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
    stop_direction : int
        Direction to check when termination height is reached;
        1 means high to low, -1 means low to high, 0 means check both directions.

    Returns
    -------
    numpy array
        2D array. [0,:] has the x,y,z location for initial point
        [:,0] is the x positions over the integration.
        Positions are reported in ECEF (km).


    """

    if recursive_loop_count is None:
        recursive_loop_count = 0
    # number of times integration routine must output step location
    if steps is None:
        steps = np.arange(max_steps)
    # ensure date is a float for IGRF call
    if not isinstance(date, float):
        # recast from datetime to float, as required by IGRF12 code
        doy = (date - datetime.datetime(date.year, 1, 1)).days
        # number of days in year, works for leap years
        num_doy_year = (datetime.datetime(date.year + 1, 1, 1) - datetime.datetime(date.year, 1, 1)).days
        date = float(date.year) + \
               (float(doy) + float(date.hour + date.minute/60. + date.second/3600.)/24.)/float(num_doy_year + 1)

# Termination event function for solve_ivp, integration will stop when we cross
# termination height from above or below.

    def height_termination(t, y, date, step_size, direction, height):

        return ecef_to_geodetic(y[0],y[1],y[2])[2] - height

    height_termination.direction = stop_direction
    height_termination.terminal = True
    init = np.transpose(init)

    bunch = scipy.integrate.solve_ivp(fun=igrf.igrf_step,
                                      y0=init.copy(),
                                      t_eval=steps,
                                      t_span=(steps[0], steps[-1]),
                                      args=(date, step_size, direction, height),
                                      rtol=1.E-11,
                                      atol=1.E-11,
                                      events = height_termination)


    trace_north = np.transpose(bunch.y)
    messg = bunch.message

    m1 = 'The solver successfully reached the end of the integration interval.'
    m2 = 'A termination event occurred.'

    if messg != m1 and messg != m2:
        raise RuntimeError("Field-Line trace not successful.")
    
    return trace_north
    '''
    # fortran integration gets close to target height
    if recurse & (z > check_height*1.000001):
        if (recursive_loop_count < 1000):
            # When we have not reached the reference height, call field_line_trace
            # again by taking check value as init - recursive call
            recursive_loop_count = recursive_loop_count + 1
            trace_north1 = field_line_trace(check, date, direction, height,
                                            step_size=step_size,
                                            max_steps=max_steps,
                                            recursive_loop_count=recursive_loop_count,
                                            steps=steps)
        else:
            raise RuntimeError("After 1000 iterations couldn't reach target altitude")
        # append new trace data to existing trace data
        # this return is taken as part of recursive loop
        return np.vstack((trace_north, trace_north1))
    else:
        # return results if we make it to the target altitude

        # filter points to terminate at point closest to target height
        # code also introduces a variable length return, though I suppose
        # that already exists with the recursive functionality
        # while this check is done innternally within Fortran integrand, if
        # that steps out early, the output we receive would be problematic.
        # Steps below provide an extra layer of security that output has some
        # semblance to expectations
        if min_check_flag:
            x, y, z = ecef_to_geodetic(trace_north[:, 0], trace_north[:, 1], trace_north[:, 2])
            idx = np.argmin(np.abs(check_height - z))
            if (z[idx] < check_height*1.001) and (idx > 0):
                trace_north = trace_north[:idx + 1, :]
    '''
    


def full_field_line(init, date, height, step_size=100., max_steps=1000,
                    steps=None, **kwargs):
    """Perform field line tracing using IGRF and scipy.integrate.odeint.

    Parameters
    ----------
    init : array-like of floats
        Position to begin field line tracing from in ECEF (x,y,z) km
    date : datetime or float
        Date to perform tracing on (year + day/365 + hours/24. + etc.)
        Accounts for leap year if datetime provided.
    height : float
        Altitude to terminate trace, geodetic WGS84 (km)
    max_steps : float
        Maximum number of steps along each direction that should be taken
    step_size : float
        Distance in km for each large integration step. Multiple substeps
        are taken as determined by scipy.integrate.odeint
    steps : array-like of ints or floats
        Number of steps along field line when field line trace positions should
        be reported. By default, each step is reported, plus origin;
            steps=np.arange(max_steps+1).
        Two traces are made, one north, the other south, thus the output array
        could have double max_steps, or more via recursion.

    Returns
    -------
    numpy array
        2D array. [0,:] has the x,y,z location for southern footpoint
        [:,0] is the x positions over the integration.
        Positions are reported in ECEF (km).


    """

    if steps is None:
        steps = np.arange(max_steps + 1)
    if len(steps) != max_steps + 1:
        raise ValueError('Length of steps must be max_steps+1.')

    # trace north, then south, and combine
    trace_south = field_line_trace(init, date, -1., height,
                                   steps=steps,
                                   step_size=step_size,
                                   max_steps=max_steps,
                                   **kwargs)
    trace_north = field_line_trace(init, date, 1., height,
                                   steps=steps,
                                   step_size=step_size,
                                   max_steps=max_steps,
                                   **kwargs)
    # order of field points is generally along the field line, south to north
    # don't want to include the initial point twice
    trace = np.vstack((trace_south[::-1][:-1, :], trace_north))
    return trace


def calculate_integrated_mag_drift_unit_vectors_ecef(latitude, longitude, altitude, datetimes,
                                                     steps=None, max_steps=1000, step_size=100.,
                                                     ref_height=120., filter_zonal=True):
    """Calculates field line integrated geomagnetic basis vectors.

     Unit vectors are expressed in ECEF coordinates.

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
    filter_zonal : bool
        If True, removes any field aligned component from the calculated
        zonal unit vector. Resulting coordinate system is not-orthogonal.

    Returns
    -------
    zon_x, zon_y, zon_z, fa_x, fa_y, fa_z, mer_x, mer_y, mer_z

    Note
    ----
        The zonal vector is calculated by field-line tracing from
        the input locations toward the footpoint locations at ref_height.
        The cross product of these two vectors is taken to define the plane of
        the magnetic field. This vector is not always orthogonal
        with the local field-aligned vector (IGRF), thus any component of the
        zonal vector with the field-aligned direction is removed (optional).
        The meridional unit vector is defined via the cross product of the
        zonal and field-aligned directions.

    """

    if steps is None:
        steps = np.arange(max_steps + 1)
    latitude = np.array(latitude)
    longitude = np.array(longitude)
    altitude = np.array(altitude)

    # calculate satellite position in ECEF coordinates
    ecef_x, ecef_y, ecef_z = geodetic_to_ecef(latitude, longitude, altitude)
    # also get position in geocentric coordinates
    geo_lat, geo_long, geo_alt = ecef_to_geocentric(ecef_x, ecef_y, ecef_z,
                                                    ref_height=0.)
    # geo_lat, geo_long, geo_alt = ecef_to_geodetic(ecef_x, ecef_y, ecef_z)

    # filter longitudes (could use pysat's function here)
    idx, = np.where(geo_long < 0)
    geo_long[idx] = geo_long[idx] + 360.
    # prepare output lists
    north_x = []
    north_y = []
    north_z = []
    south_x = []
    south_y = []
    south_z = []
    bn = []
    be = []
    bd = []

    for x, y, z, alt, colat, elong, time in zip(ecef_x, ecef_y, ecef_z,
                                                altitude, np.deg2rad(90. - latitude),
                                                np.deg2rad(longitude), datetimes):
        init = np.array([x, y, z])
        trace = full_field_line(init, time, ref_height, step_size=step_size,
                                max_steps=max_steps,
                                steps=steps)
        # store final location, full trace goes south to north
        trace_north = trace[-1, :]
        trace_south = trace[0, :]
        # recast from datetime to float, as required by IGRF12 code
        doy = (time - datetime.datetime(time.year, 1, 1)).days
        # number of days in year, works for leap years
        num_doy_year = (datetime.datetime(time.year + 1, 1, 1) - datetime.datetime(time.year, 1, 1)).days
        date = time.year + float(doy)/float(num_doy_year + 1)
        date += (time.hour + time.minute/60. + time.second/3600.)/24./float(num_doy_year + 1)
        # get IGRF field components
        # tbn, tbe, tbd, tbmag are in nT
        # geodetic input
        tbn, tbe, tbd, tbmag = igrf.igrf13syn(0, date, 1, alt, colat, elong)

        # collect outputs
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
    # normalize the vectors
    norm_foot = np.sqrt(zvx_foot**2 + zvy_foot**2 + zvz_foot**2)

    # calculate zonal vector
    zvx = zvx_foot/norm_foot
    zvy = zvy_foot/norm_foot
    zvz = zvz_foot/norm_foot

    if filter_zonal:
        # print ("Making magnetic vectors orthogonal")
        # remove any field aligned component to the zonal vector
        dot_fa = zvx*bx + zvy*by + zvz*bz
        zvx -= dot_fa*bx
        zvy -= dot_fa*by
        zvz -= dot_fa*bz
        zvx, zvy, zvz = normalize_vector(zvx, zvy, zvz)

    # compute meridional vector
    # cross product of zonal and magnetic unit vector
    mx, my, mz = cross_product(zvx, zvy, zvz,
                               bx, by, bz)
    # add unit vectors for magnetic drifts in ecef coordinates
    return zvx, zvy, zvz, bx, by, bz, mx, my, mz


def magnetic_vector(x, y, z, dates, normalize=False):
    """Uses IGRF to calculate geomagnetic field.

    Parameters
    ----------
    x : array-like
        Position in ECEF (km), X
    y : array-like
        Position in ECEF (km), Y
    z : array-like
        Position in ECEF (km), Z
    normalize : bool (False)
        If True, return unit vector

    Returns
    -------
    array, array, array
        Magnetic field along ECEF directions

    """

    # prepare output lists
    bn = []
    be = []
    bd = []
    bm = []

    # need a double variable for time
    doy = np.array([(time - datetime.datetime(time.year, 1, 1)).days for time in dates])
    years = np.array([time.year for time in dates])
    num_doy_year = np.array(
        [(datetime.datetime(time.year + 1, 1, 1) - datetime.datetime(time.year, 1, 1)).days for time in dates])
    time = np.array([(time.hour + time.minute/60. + time.second/3600.)/24. for time in dates])
    ddates = years + (doy + time)/(num_doy_year + 1)

    # use geocentric coordinates for calculating magnetic field
    # transformation between it and ECEF is robust
    # geodetic translations introduce error
    latitudes, longitudes, altitudes = ecef_to_geocentric(x, y, z, ref_height=0.)

    for colat, elong, alt, date in zip(np.deg2rad(90. - latitudes),
                                       np.deg2rad(longitudes),
                                       altitudes,
                                       ddates):
        # tbn, tbe, tbd, tbmag are in nT
        tbn, tbe, tbd, tbmag = igrf.igrf13syn(0, date, 2, alt, colat, elong)

        # collect outputs
        bn.append(tbn)
        be.append(tbe)
        bd.append(tbd)
        bm.append(tbmag)
    # repackage
    bn = np.array(bn)
    be = np.array(be)
    bd = np.array(bd)
    bm = np.array(bm)

    if normalize:
        bn /= bm
        be /= bm
        bd /= bm

    # calculate magnetic unit vector
    bx, by, bz = enu_to_ecef_vector(be, bn, -bd, latitudes, longitudes)

    return bx, by, bz, bm


def calculate_geomagnetic_basis(latitude, longitude, altitude, datetimes):
    """Calculates local geomagnetic basis vectors and mapping scalars.

    Thin wrapper around calculate_mag_drift_unit_vectors_ecef set
    to default parameters and with more organization of the outputs.

    Parameters
    ----------
    latitude : array-like of floats (degrees) [-90., 90]
        Latitude of location, degrees, WGS84
    longitude : array-like of floats (degrees) [-180., 360.]
        Longitude of location, degrees, WGS84
    altitude : array-like of floats (km)
        Altitude of location, height above surface, WGS84
    datetimes : array-like of datetimes
        Time to calculate vectors

    Returns
    -------
    dict
        zon_x (y,z): zonal unit vector along ECEF X, Y, and Z directions
        fa_x (y,z): field-aligned unit vector along ECEF X, Y, and Z directions
        mer_x (y,z): meridional unit vector along ECEF X, Y, and Z directions

        d_zon_mag: D zonal vector magnitude
        d_fa_mag: D field-aligned vector magnitude
        d_mer_mag: D meridional vector magnitude

        d_zon_x (y,z) : D zonal vector components along ECEF X, Y, and Z directions
        d_mer_x (y,z) : D meridional vector components along ECEF X, Y, and Z directions
        d_fa_x (y,z) : D field aligned vector components along ECEF X, Y, and Z directions

        e_zon_mag: E zonal vector magnitude
        e_fa_mag: E field-aligned vector magnitude
        e_mer_mag: E meridional vector magnitude

        e_zon_x (y,z) : E zonal vector components along ECEF X, Y, and Z directions
        e_mer_x (y,z) : E meridional vector components along ECEF X, Y, and Z directions
        e_fa_x (y,z) : E field aligned vector components along ECEF X, Y, and Z directions

    """

    zx, zy, zz, fx, fy, fz, mx, my, mz, info_d = calculate_mag_drift_unit_vectors_ecef(latitude, longitude,
                                                                                       altitude, datetimes,
                                                                                       full_output=True)
    d_zon_mag = np.sqrt(info_d['d_zon_x']**2 + info_d['d_zon_y']**2 + info_d['d_zon_z']**2)
    d_fa_mag = np.sqrt(info_d['d_fa_x']**2 + info_d['d_fa_y']**2 + info_d['d_fa_z']**2)
    d_mer_mag = np.sqrt(info_d['d_mer_x']**2 + info_d['d_mer_y']**2 + info_d['d_mer_z']**2)
    e_zon_mag = np.sqrt(info_d['e_zon_x']**2 + info_d['e_zon_y']**2 + info_d['e_zon_z']**2)
    e_fa_mag = np.sqrt(info_d['e_fa_x']**2 + info_d['e_fa_y']**2 + info_d['e_fa_z']**2)
    e_mer_mag = np.sqrt(info_d['e_mer_x']**2 + info_d['e_mer_y']**2 + info_d['e_mer_z']**2)
    # assemble output dictionary
    out_d = {'zon_x': zx, 'zon_y': zy, 'zon_z': zz,
             'fa_x': fx, 'fa_y': fy, 'fa_z': fz,
             'mer_x': mx, 'mer_y': my, 'mer_z': mz,
             'd_zon_x': info_d['d_zon_x'], 'd_zon_y': info_d['d_zon_y'], 'd_zon_z': info_d['d_zon_z'],
             'd_fa_x': info_d['d_fa_x'], 'd_fa_y': info_d['d_fa_y'], 'd_fa_z': info_d['d_fa_z'],
             'd_mer_x': info_d['d_mer_x'], 'd_mer_y': info_d['d_mer_y'], 'd_mer_z': info_d['d_mer_z'],
             'e_zon_x': info_d['e_zon_x'], 'e_zon_y': info_d['e_zon_y'], 'e_zon_z': info_d['e_zon_z'],
             'e_fa_x': info_d['e_fa_x'], 'e_fa_y': info_d['e_fa_y'], 'e_fa_z': info_d['e_fa_z'],
             'e_mer_x': info_d['e_mer_x'], 'e_mer_y': info_d['e_mer_y'], 'e_mer_z': info_d['e_mer_z'],
             'd_zon_mag': d_zon_mag, 'd_fa_mag': d_fa_mag, 'd_mer_mag': d_mer_mag,
             'e_zon_mag': e_zon_mag, 'e_fa_mag': e_fa_mag, 'e_mer_mag': e_mer_mag}

    return out_d


def calculate_mag_drift_unit_vectors_ecef(latitude, longitude, altitude, datetimes,
                                          step_size=2., tol=1.E-4,
                                          tol_zonal_apex=1.E-4, max_loops=100,
                                          ecef_input=False, centered_diff=True,
                                          full_output=False, include_debug=False,
                                          scalar=1.,
                                          edge_steps=1, dstep_size=2.,
                                          max_steps=None, ref_height=None,
                                          steps=None, ):
    """Calculates local geomagnetic basis vectors and mapping scalars.

    Zonal - Generally Eastward (+East); lies along a surface of constant apex height
    Field Aligned - Generally Northward (+North); points along geomagnetic field
    Meridional - Generally Vertical (+Up); points along the gradient in apex height

    The apex height is the geodetic height of the field line at its highest point.
    Unit vectors are expressed in ECEF coordinates.

    Parameters
    ----------
    latitude : array-like of floats (degrees) [-90., 90]
        Latitude of location, degrees, WGS84
    longitude : array-like of floats (degrees) [-180., 360.]
        Longitude of location, degrees, WGS84
    altitude : array-like of floats (km)
        Altitude of location, height above surface, WGS84
    datetimes : array-like of datetimes
        Time to calculate vectors
    step_size : float
        Step size (km) to use when calculating changes in apex height
    tol : float
        Tolerance goal for the magnitude of the change in unit vectors per loop
    tol_zonal_apex : Maximum allowed change in apex height along
        zonal direction
    max_loops : int
        Maximum number of iterations
    ecef_input : bool (False)
        If True, inputs latitude, longitude, altitude are interpreted as
        x, y, and z in ECEF coordinates (km).
    full_output : bool (False)
        If True, return an additional dictionary with the E and D mapping
        vectors
    include_deubg : bool (False)
        If True, include stats about iterative process in optional dictionary.
        Requires full_output=True
    centered_diff : bool (True)
        If True, a symmetric centered difference is used when calculating
        the change in apex height along the zonal direction, used within
        the zonal unit vector calculation
    scalar : int
        Used to modify unit magnetic field within algorithm. Generally
        speaking, this should not be modified
    edge_steps : int (1)
        Number of steps taken when moving across field lines and calculating
        the change in apex location. This parameter impacts both runtime
        and accuracy of the D, E vectors.
    dstep_size : float (.016 km)
        Step size (km) used when calculting the expansion of field line surfaces.
        Generally, this should be the same as step_size.
    max_steps : int
        Deprecated
    ref_height : float
        Deprecated
    steps : list-like
        Deprecated

    Returns
    -------
    zon_x, zon_y, zon_z, fa_x, fa_y, fa_z, mer_x, mer_y, mer_z, (optional dictionary)

    Optional output dictionary
    --------------------------
    Full Output Parameters

    d_zon_x (y,z) : D zonal vector components along ECEF X, Y, and Z directions
    d_mer_x (y,z) : D meridional vector components along ECEF X, Y, and Z directions
    d_fa_x (y,z) : D field aligned vector components along ECEF X, Y, and Z directions

    e_zon_x (y,z) : E zonal vector components along ECEF X, Y, and Z directions
    e_mer_x (y,z) : E meridional vector components along ECEF X, Y, and Z directions
    e_fa_x (y,z) : E field aligned vector components along ECEF X, Y, and Z directions


    Debug Parameters

    diff_mer_apex : rate of change in apex height (km) along meridional vector
    diff_mer_vec : magnitude of vector change for last loop
    diff_zonal_apex : rate of change in apex height (km) along zonal vector
    diff_zonal_vec : magnitude of vector change for last loop
    loops : Number of loops
    vector_seed_type : Initial vector used for starting calculation (deprecated)

    Note
    ----
        The zonal and meridional vectors are calculated by using the observed
        apex-height gradient to rotate a pair of vectors orthogonal
        to eachother and the geomagnetic field such that one points along
        no change in apex height (zonal), the other along the max (meridional).
        The rotation angle theta is given by

            Tan(theta) = apex_height_diff_zonal/apex_height_diff_meridional

        The method terminates when successive updates to both the zonal and meridional
        unit vectors differ (magnitude of difference) by less than tol, and the
        change in apex_height from input location is less than tol_zonal_apex.

    """

    if max_steps is not None:
        raise DeprecationWarning('max_steps is no longer supported.')
    if ref_height is not None:
        raise DeprecationWarning('ref_height is no longer supported.')
    if steps is not None:
        raise DeprecationWarning('steps is no longer supported.')
    if step_size <= 0:
        raise ValueError('Step Size must be greater than 0.')

    ss = scalar

    if ecef_input:
        ecef_x, ecef_y, ecef_z = latitude, longitude, altitude
        # lat and long needed for initial zonal and meridional vector
        # generation later on
        latitude, longitude, altitude = ecef_to_geocentric(ecef_x, ecef_y, ecef_z)

    else:
        latitude = np.array(latitude)
        longitude = np.array(longitude)
        altitude = np.array(altitude)
        # ensure latitude reasonable
        idx, = np.where(np.abs(latitude) > 90.)
        if len(idx) > 0:
            raise RuntimeError('Latitude out of bounds [-90., 90.].')
        # ensure longitude reasonable
        idx, = np.where((longitude < -180.) | (longitude > 360.))
        if len(idx) > 0:
            print('Out of spec :', longitude[idx])
            raise RuntimeError('Longitude out of bounds [-180., 360.].')

        # calculate satellite position in ECEF coordinates
        ecef_x, ecef_y, ecef_z = geodetic_to_ecef(latitude, longitude, altitude)

    # get apex location for root point
    a_x, a_y, a_z, _, _, apex_root = apex_location_info(ecef_x, ecef_y, ecef_z,
                                                        datetimes,
                                                        return_geodetic=True,
                                                        ecef_input=True)
    bx, by, bz, bm = magnetic_vector(ecef_x, ecef_y, ecef_z, datetimes, normalize=True)

    bx, by, bz = ss*bx, ss*by, ss*bz
    # need a vector perpendicular to mag field
    # infinitely many
    # let's use the east vector as a great place to start
    tzx, tzy, tzz = enu_to_ecef_vector(ss*np.ones(len(bx)), np.zeros(len(bx)),
                                       np.zeros(len(bx)), latitude, longitude)
    init_type = np.zeros(len(bx)) - 1

    # get meridional from this
    tmx, tmy, tmz = cross_product(tzx, tzy, tzz, bx, by, bz)
    # normalize
    tmx, tmy, tmz = normalize_vector(tmx, tmy, tmz)
    # get orthogonal zonal now
    tzx, tzy, tzz = cross_product(bx, by, bz, tmx, tmy, tmz)
    # normalize
    tzx, tzy, tzz = normalize_vector(tzx, tzy, tzz)

    # loop variables
    loop_num = 0
    repeat_flag = True
    while repeat_flag:
        # get apex field height location info for both places
        # after taking step along these directions

        # zonal-ish direction
        ecef_xz, ecef_yz, ecef_zz = ecef_x + step_size*tzx, ecef_y + step_size*tzy, ecef_z + step_size*tzz
        _, _, _, _, _, apex_z = apex_location_info(ecef_xz, ecef_yz, ecef_zz,
                                                   datetimes,
                                                   return_geodetic=True,
                                                   ecef_input=True)
        if centered_diff:
            ecef_xz2, ecef_yz2, ecef_zz2 = ecef_x - step_size*tzx, ecef_y - step_size*tzy, ecef_z - step_size*tzz
            _, _, _, _, _, apex_z2 = apex_location_info(ecef_xz2, ecef_yz2, ecef_zz2,
                                                        datetimes,
                                                        return_geodetic=True,
                                                        ecef_input=True)
            diff_apex_z = apex_z - apex_z2
            diff_apex_z /= 2*step_size
        else:
            diff_apex_z = apex_z - apex_root
            diff_apex_z /= step_size

        # meridional-ish direction
        ecef_xm, ecef_ym, ecef_zm = ecef_x + step_size*tmx, ecef_y + step_size*tmy, ecef_z + step_size*tmz
        _, _, _, _, _, apex_m = apex_location_info(ecef_xm, ecef_ym, ecef_zm,
                                                   datetimes,
                                                   return_geodetic=True,
                                                   ecef_input=True)

        diff_apex_m = apex_m - apex_root
        diff_apex_m /= step_size

        # rotation angle
        theta = np.arctan2(diff_apex_z, diff_apex_m)
        # theta2 = np.pi/2. - np.arctan2(diff_apex_m, diff_apex_z)

        # rotate vectors around unit vector to align along desired gradients
        # zonal along no gradient, meridional along max
        # see wikipedia quaternion spatial rotation page for equation below
        # https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
        # precalculate some info
        ct = np.cos(theta)
        st = np.sin(theta)
        # zonal vector
        tzx2, tzy2, tzz2 = tzx*ct - tmx*st, tzy*ct - tmy*st, tzz*ct - tmz*st
        # meridional vector
        tmx2, tmy2, tmz2 = tmx*ct + tzx*st, tmy*ct + tzy*st, tmz*ct + tzz*st

        # track difference
        dx, dy, dz = (tzx2 - tzx)**2, (tzy2 - tzy)**2, (tzz2 - tzz)**2
        diff_z = np.sqrt(dx + dy + dz)
        dx, dy, dz = (tmx2 - tmx)**2, (tmy2 - tmy)**2, (tmz2 - tmz)**2
        diff_m = np.sqrt(dx + dy + dz)
        # take biggest difference
        diff = np.max([diff_z, diff_m])

        # store info into calculation vectors to refine next loop
        tzx, tzy, tzz = tzx2, tzy2, tzz2
        tmx, tmy, tmz = tmx2, tmy2, tmz2

        # check if we are done
        if (diff < tol) & (np.max(np.abs(diff_apex_z)) < tol_zonal_apex) & (loop_num > 1):
            repeat_flag = False

        loop_num += 1
        if loop_num > max_loops:
            tzx, tzy, tzz = np.nan*tzx, np.nan*tzy, np.nan*tzz
            tmx, tmy, tmz = np.nan*tzx, np.nan*tzy, np.nan*tzz
            estr = ' step_size ' + str(step_size) + ' diff_z ' + str(np.max(np.abs(diff_apex_z)))
            estr += ' diff ' + str(diff) + ' centered ' + str(centered_diff)
            raise RuntimeWarning("Didn't converge after reaching max_loops " + estr)

    # store temp arrays into output
    zx, zy, zz = tzx, tzy, tzz
    mx, my, mz = ss*tmx, ss*tmy, ss*tmz

    if full_output:
        # calculate expansion of zonal vector
        # recalculating zonal vector without centered difference
        # keeps locations along same apex height
        diff_apex_r, diff_h = apex_distance_after_local_step(ecef_x, ecef_y, ecef_z,
                                                             datetimes,
                                                             vector_direction='zonal',
                                                             ecef_input=True,
                                                             edge_length=dstep_size,
                                                             edge_steps=edge_steps,
                                                             return_geodetic=True)
        # need to translate to arc length
        radial_loc = np.sqrt(a_x**2 + a_y**2 + a_z**2)
        subtend_angle = np.arcsin(diff_apex_r/2./radial_loc)
        diff_apex_circ = radial_loc*2*subtend_angle
        grad_brb = diff_apex_circ/(2.*dstep_size)

        # get magnitude of magnetic field at root apex location
        bax, bay, baz, bam = magnetic_vector(a_x, a_y, a_z, datetimes,
                                             normalize=True)

        # d vectors
        d_fa_x, d_fa_y, d_fa_z = bam/bm*bx, bam/bm*by, bam/bm*bz
        d_zon_x, d_zon_y, d_zon_z = grad_brb*zx, grad_brb*zy, grad_brb*zz

        # get meridional that completes set
        d_mer_x, d_mer_y, d_mer_z = cross_product(d_zon_x, d_zon_y, d_zon_z,
                                                  d_fa_x, d_fa_y, d_fa_z)
        mag = d_mer_x**2 + d_mer_y**2 + d_mer_z**2
        d_mer_x, d_mer_y, d_mer_z = d_mer_x/mag, d_mer_y/mag, d_mer_z/mag

        # e vectors (Richmond nomenclature)
        e_zon_x, e_zon_y, e_zon_z = cross_product(d_fa_x, d_fa_y, d_fa_z,
                                                  d_mer_x, d_mer_y, d_mer_z)
        e_fa_x, e_fa_y, e_fa_z = cross_product(d_mer_x, d_mer_y, d_mer_z,
                                               d_zon_x, d_zon_y, d_zon_z)
        e_mer_x, e_mer_y, e_mer_z = cross_product(d_zon_x, d_zon_y, d_zon_z,
                                                  d_fa_x, d_fa_y, d_fa_z)

        outd = {
            'd_zon_x': d_zon_x,
            'd_zon_y': d_zon_y,
            'd_zon_z': d_zon_z,
            'd_mer_x': d_mer_x,
            'd_mer_y': d_mer_y,
            'd_mer_z': d_mer_z,
            'd_fa_x': d_fa_x,
            'd_fa_y': d_fa_y,
            'd_fa_z': d_fa_z,
            'e_zon_x': e_zon_x,
            'e_zon_y': e_zon_y,
            'e_zon_z': e_zon_z,
            'e_mer_x': e_mer_x,
            'e_mer_y': e_mer_y,
            'e_mer_z': e_mer_z,
            'e_fa_x': e_fa_x,
            'e_fa_y': e_fa_y,
            'e_fa_z': e_fa_z,
        }

        if include_debug:
            # calculate zonal gradient using latest vectors
            ecef_xz, ecef_yz, ecef_zz = ecef_x + dstep_size*zx, ecef_y + dstep_size*zy, ecef_z + dstep_size*zz
            a_x2, a_y2, a_z2, _, _, apex_z = apex_location_info(ecef_xz, ecef_yz, ecef_zz,
                                                                datetimes,
                                                                return_geodetic=True,
                                                                ecef_input=True)
            ecef_xz2, ecef_yz2, ecef_zz2 = ecef_x - dstep_size*zx, ecef_y - dstep_size*zy, ecef_z - dstep_size*zz
            _, _, _, _, _, apex_z2 = apex_location_info(ecef_xz2, ecef_yz2, ecef_zz2,
                                                        datetimes,
                                                        return_geodetic=True,
                                                        ecef_input=True)
            diff_apex_z = apex_z - apex_z2
            grad_zonal = diff_apex_z/(2.*dstep_size)

            # calculate meridional gradient using latest vectors
            ecef_xm, ecef_ym, ecef_zm = ecef_x + dstep_size*mx, ecef_y + dstep_size*my, ecef_z + dstep_size*mz
            _, _, _, _, _, apex_m = apex_location_info(ecef_xm, ecef_ym, ecef_zm,
                                                       datetimes,
                                                       return_geodetic=True,
                                                       ecef_input=True)
            ecef_xm2, ecef_ym2, ecef_zm2 = ecef_x - dstep_size*mx, ecef_y - dstep_size*my, ecef_z - dstep_size*mz
            _, _, _, _, _, apex_m2 = apex_location_info(ecef_xm2, ecef_ym2, ecef_zm2,
                                                        datetimes,
                                                        return_geodetic=True,
                                                        ecef_input=True)
            diff_apex_m = apex_m - apex_m2
            grad_apex = diff_apex_m/(2.*dstep_size)

            # # potentially higher accuracy method of getting height gradient magnitude
            # did not increase accuracy
            # leaving here as a reminder that this code path has been checked out
            # diff_apex_r, diff_h = apex_distance_after_local_step(ecef_x, ecef_y, ecef_z,
            #                                                 datetimes,
            #                                                 vector_direction='meridional',
            #                                                 ecef_input=True,
            #                                                 edge_length=dstep_size,
            #                                                 edge_steps=edge_steps,
            #                                                 return_geodetic=True)

            # second path D, E vectors
            mer_scal = grad_apex
            # d meridional vector via apex height gradient
            d_mer2_x, d_mer2_y, d_mer2_z = mer_scal*mx, mer_scal*my, mer_scal*mz
            # zonal to complete set (apex height gradient calculation is precise)
            # less so for zonal gradient
            d_zon2_x, d_zon2_y, d_zon2_z = cross_product(d_fa_x, d_fa_y, d_fa_z,
                                                         d_mer2_x, d_mer2_y, d_mer2_z)
            mag = d_zon2_x**2 + d_zon2_y**2 + d_zon2_z**2
            d_zon2_x, d_zon2_y, d_zon2_z = d_zon2_x/mag, d_zon2_y/mag, d_zon2_z/mag

            tempd = {'diff_zonal_apex': grad_zonal,
                     'diff_mer_apex': grad_apex,
                     'loops': loop_num,
                     'vector_seed_type': init_type,
                     'diff_zonal_vec': diff_z,
                     'diff_mer_vec': diff_m,
                     'd_zon2_x': d_zon2_x,
                     'd_zon2_y': d_zon2_y,
                     'd_zon2_z': d_zon2_z,
                     'd_mer2_x': d_mer2_x,
                     'd_mer2_y': d_mer2_y,
                     'd_mer2_z': d_mer2_z,
                     }
            for key in tempd.keys():
                outd[key] = tempd[key]
        return zx, zy, zz, bx, by, bz, mx, my, mz, outd

    # return unit vectors for magnetic drifts in ecef coordinates
    return zx, zy, zz, bx, by, bz, mx, my, mz


def step_along_mag_unit_vector(x, y, z, date, direction=None, num_steps=1.,
                               step_size=25., scalar=1):
    """
    Move along 'lines' formed by following the magnetic unit vector directions.

    Moving along the field is effectively the same as a field line trace though
    extended movement along a field should use the specific field_line_trace
    method.


    Parameters
    ----------
    x : ECEF-x (km)
        Location to step from in ECEF (km).
    y : ECEF-y (km)
        Location to step from in ECEF (km).
    z : ECEF-z (km)
        Location to step from in ECEF (km).
    date : list-like of datetimes
        Date and time for magnetic field
    direction : string
        String identifier for which unit vector direction to move along.
        Supported inputs, 'meridional', 'zonal', 'aligned'
    num_steps : int
        Number of steps to take along unit vector direction
    step_size = float
        Distance taken for each step (km)
    scalar : int
        Scalar modifier for step size distance. Input a -1 to move along
        negative unit vector direction.

    Returns
    -------
    np.array
        [x, y, z] of ECEF location after taking num_steps along direction,
        each step_size long.

    Notes
    -----
        centered_diff=True is passed along to calculate_mag_drift_unit_vectors_ecef
        when direction='meridional', while centered_diff=False is used
        for the 'zonal' direction. This ensures that when moving along the
        zonal direction there is a minimal change in apex height.

    """

    if direction == 'meridional':
        centered_diff = True
    else:
        centered_diff = False

    for i in np.arange(num_steps):
        # x, y, z in ECEF
        # get unit vector directions
        zvx, zvy, zvz, bx, by, bz, mx, my, mz = calculate_mag_drift_unit_vectors_ecef(
                                                            x, y, z, date,
                                                            step_size=step_size,
                                                            ecef_input=True,
                                                            centered_diff=centered_diff,
                                                            scalar=scalar)
        # pull out the direction we need
        if direction == 'meridional':
            ux, uy, uz = mx, my, mz
        elif direction == 'zonal':
            ux, uy, uz = zvx, zvy, zvz
        elif direction == 'aligned':
            ux, uy, uz = bx, by, bz

        # take steps along direction
        x = x + step_size*ux
        y = y + step_size*uy
        z = z + step_size*uz

    return x, y, z


def footpoint_location_info(glats, glons, alts, dates, step_size=100.,
                            num_steps=1000, return_geodetic=False,
                            ecef_input=False):
    """Return ECEF location of footpoints in Northern/Southern hemisphere

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
    step_size : float (100. km)
        Step size (km) used for tracing coarse field line
    num_steps : int (1.E-5 km)
        Number of steps passed along to field_line_trace as max_steps.
    ecef_input : bool
        If True, glats, glons, and alts are treated as x, y, z (ECEF).
    return_geodetic : bool
        If True, footpoint locations returned as lat, long, alt.

    Returns
    -------
    array(len(glats), 3), array(len(glats), 3)
        Northern and Southern ECEF X,Y,Z locations

    """

    # use input location and convert to ECEF
    if ecef_input:
        ecef_xs, ecef_ys, ecef_zs = glats, glons, alts
    else:
        ecef_xs, ecef_ys, ecef_zs = geodetic_to_ecef(glats, glons, alts)

    north_ftpnt = np.empty((len(ecef_xs), 3))
    south_ftpnt = np.empty((len(ecef_xs), 3))

    root = np.array([0, 0, 0])
    i = 0
    steps = np.arange(num_steps + 1)
    for ecef_x, ecef_y, ecef_z, date in zip(ecef_xs, ecef_ys, ecef_zs, dates):
        yr, doy = pysat.utils.time.getyrdoy(date)
        double_date = float(yr) + float(doy)/366.

        root[:] = (ecef_x, ecef_y, ecef_z)
        trace_north = field_line_trace(root, double_date, 1., 120.,
                                       steps=steps,
                                       step_size=step_size,
                                       max_steps=num_steps)
        # southern tracing
        trace_south = field_line_trace(root, double_date, -1., 120.,
                                       steps=steps,
                                       step_size=step_size,
                                       max_steps=num_steps)
        # footpoint location
        north_ftpnt[i, :] = trace_north[-1, :]
        south_ftpnt[i, :] = trace_south[-1, :]
        i += 1

    if return_geodetic:
        north_ftpnt[:, 0], north_ftpnt[:, 1], north_ftpnt[:, 2] = ecef_to_geodetic(
            north_ftpnt[:, 0], north_ftpnt[:, 1],
            north_ftpnt[:, 2])
        south_ftpnt[:, 0], south_ftpnt[:, 1], south_ftpnt[:, 2] = ecef_to_geodetic(
            south_ftpnt[:, 0], south_ftpnt[:, 1],
            south_ftpnt[:, 2])
    return north_ftpnt, south_ftpnt


def apex_location_info(glats, glons, alts, dates, step_size=100.,
                       fine_step_size=1.E-5, fine_max_steps=5,
                       return_geodetic=False, ecef_input=False):
    """Determine apex location for the field line passing through input point.

    Employs a two stage method. A broad step (step_size) field line trace spanning
    Northern/Southern footpoints is used to find the location with the largest
    geodetic (WGS84) height. A binary search higher resolution trace (goal fine_step_size)
    is then used to get a better fix on this location. Each loop, step_size halved.
    Greatest geodetic height is once again selected once the step_size is below
    fine_step_size.

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
    step_size : float (100. km)
        Step size (km) used for tracing coarse field line
    fine_step_size : float (1.E-5 km)
        Fine step size for refining apex location height
    fine_max_steps : int (1.E-5 km)
        Fine number of steps passed along to full_field_trace. Do not
        change unless youknow exactly what you are doing.
    return_geodetic: bool
        If True, also return location in geodetic coordinates
    ecef_input : bool
        If True, glats, glons, and alts are treated as x, y, z (ECEF).

    Returns
    -------
    (float, float, float, float, float, float)
        ECEF X (km), ECEF Y (km), ECEF Z (km),
    if return_geodetic, also includes:
        Geodetic Latitude (degrees),
        Geodetic Longitude (degrees),
        Geodetic Altitude (km)

    """

    # use input location and convert to ECEF
    if ecef_input:
        ecef_xs, ecef_ys, ecef_zs = glats, glons, alts
    else:
        ecef_xs, ecef_ys, ecef_zs = geodetic_to_ecef(glats, glons, alts)
    # prepare parameters for field line trace
    max_steps = 100
    _apex_coarse_steps = np.arange(max_steps + 1)
    # high resolution trace parameters
    _apex_fine_steps = np.arange(fine_max_steps + 1)
    # prepare output
    _apex_out_x = np.empty(len(ecef_xs))
    _apex_out_y = np.empty(len(ecef_xs))
    _apex_out_z = np.empty(len(ecef_xs))

    i = 0
    for ecef_x, ecef_y, ecef_z, date in zip(ecef_xs, ecef_ys, ecef_zs, dates):
        # to get the apex location we need to do a field line trace
        # then find the highest point
        trace = full_field_line(np.array([ecef_x, ecef_y, ecef_z]), date, 0.,
                                steps=_apex_coarse_steps,
                                step_size=step_size,
                                max_steps=max_steps)
        # convert all locations to geodetic coordinates
        tlat, tlon, talt = ecef_to_geodetic(trace[:, 0], trace[:, 1], trace[:, 2])
        # determine location that is highest with respect to the geodetic Earth
        max_idx = np.argmax(talt)
        # repeat using a high resolution trace one big step size each
        # direction around identified max
        # recurse False ensures only max_steps are taken
        new_step = step_size
        # print('start range', talt[max_idx-1:max_idx+2])
        while new_step > fine_step_size:
            new_step /= 2.
            trace = full_field_line(trace[max_idx, :], date, 0.,
                                    steps=_apex_fine_steps,
                                    step_size=new_step,
                                    max_steps=fine_max_steps,
                                    recurse=False)
            # convert all locations to geodetic coordinates
            tlat, tlon, talt = ecef_to_geodetic(trace[:, 0], trace[:, 1], trace[:, 2])
            # determine location that is highest with respect to the geodetic Earth
            # print(talt)
            max_idx = np.argmax(talt)

        # collect outputs
        _apex_out_x[i] = trace[max_idx, 0]
        _apex_out_y[i] = trace[max_idx, 1]
        _apex_out_z[i] = trace[max_idx, 2]
        i += 1

    if return_geodetic:
        glat, glon, alt = ecef_to_geodetic(_apex_out_x, _apex_out_y, _apex_out_z)
        return _apex_out_x, _apex_out_y, _apex_out_z, glat, glon, alt
    else:
        return _apex_out_x, _apex_out_y, _apex_out_z


def apex_edge_lengths_via_footpoint(*args, **kwargs):
    raise DeprecationWarning('This method now called apex_distance_after_footpoint_step.')
    apex_distance_after_footpoint_step(*args, **kwargs)


def apex_distance_after_footpoint_step(glats, glons, alts, dates, direction,
                                       vector_direction, step_size=None,
                                       max_steps=None, steps=None,
                                       edge_length=25., edge_steps=5,
                                       ecef_input=False):
    """
    Calculates the distance between apex locations after stepping along
    vector_direction.

    Using the input location, the footpoint location is calculated.
    From here, a step along both the positive and negative
    vector_directions is taken, and the apex locations for those points are calculated.
    The difference in position between these apex locations is the total centered
    distance between magnetic field lines at the magnetic apex when starting
    from the footpoints with a field line half distance of edge_length.

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
    direction : string
        'north' or 'south' for tracing through northern or
        southern footpoint locations
    vector_direction : string
        'meridional' or 'zonal' unit vector directions
    step_size : float (km)
        Step size (km) used for field line integration
    max_steps : int
        Number of steps taken for field line integration
    steps : np.array
        Integration steps array passed to full_field_line, np.arange(max_steps+1)
    edge_length : float (km)
        Half of total edge length (step) taken at footpoint location.
        edge_length step in both positive and negative directions.
    edge_steps : int
        Number of steps taken from footpoint towards new field line
        in a given direction (positive/negative) along unit vector
    ecef_input : bool (False)
        If True, latitude, longitude, and altitude are treated as
        ECEF positions (km).

    Returns
    -------
    np.array,
        A closed loop field line path through input location and footpoint in
        northern/southern hemisphere and back is taken. The return edge length
        through input location is provided.

    Note
    ----
        vector direction refers to the magnetic unit vector direction

    """

    if step_size is None:
        step_size = 100.
    if max_steps is None:
        max_steps = 1000
    if steps is None:
        steps = np.arange(max_steps + 1)

    # use spacecraft location to get ECEF
    if ecef_input:
        ecef_xs, ecef_ys, ecef_zs = glats, glons, alts
    else:
        ecef_xs, ecef_ys, ecef_zs = geodetic_to_ecef(glats, glons, alts)

    # prepare output
    apex_edge_length = []

    if direction == 'north':
        ftpnts, _ = footpoint_location_info(ecef_xs, ecef_ys, ecef_zs, dates,
                                            ecef_input=True)
    elif direction == 'south':
        _, ftpnts = footpoint_location_info(ecef_xs, ecef_ys, ecef_zs, dates,
                                            ecef_input=True)

    # take step from footpoint along + vector direction
    plus_x, plus_y, plus_z = step_along_mag_unit_vector(ftpnts[:, 0], ftpnts[:, 1], ftpnts[:, 2],
                                                        dates,
                                                        direction=vector_direction,
                                                        num_steps=edge_steps,
                                                        step_size=edge_length/edge_steps)
    plus_apex_x, plus_apex_y, plus_apex_z = \
        apex_location_info(plus_x, plus_y, plus_z,
                           dates, ecef_input=True)

    # take half step from first footpoint along - vector direction
    minus_x, minus_y, minus_z = step_along_mag_unit_vector(ftpnts[:, 0], ftpnts[:, 1], ftpnts[:, 2],
                                                           dates,
                                                           direction=vector_direction,
                                                           scalar=-1,
                                                           num_steps=edge_steps,
                                                           step_size=edge_length/edge_steps)
    minus_apex_x, minus_apex_y, minus_apex_z = \
        apex_location_info(minus_x, minus_y, minus_z,
                           dates, ecef_input=True)
    # take difference in apex locations
    apex_edge_length = np.sqrt((plus_apex_x - minus_apex_x)**2 +
                               (plus_apex_y - minus_apex_y)**2 +
                               (plus_apex_z - minus_apex_z)**2)

    return apex_edge_length


def apex_distance_after_local_step(glats, glons, alts, dates,
                                   vector_direction,
                                   edge_length=25.,
                                   edge_steps=5,
                                   ecef_input=False,
                                   return_geodetic=False):
    """
    Calculates the distance between apex locations mapping to the input location.

    Using the input location, the apex location is calculated. Also from the input
    location, a step along both the positive and negative
    vector_directions is taken, and the apex locations for those points are calculated.
    The difference in position between these apex locations is the total centered
    distance between magnetic field lines at the magnetic apex when starting
    locally with a field line half distance of edge_length.

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
    vector_direction : string
        'meridional' or 'zonal' unit vector directions
    edge_length : float (km)
        Half of total edge length (step) taken at footpoint location.
        edge_length step in both positive and negative directions.
    edge_steps : int
        Number of steps taken from footpoint towards new field line
        in a given direction (positive/negative) along unit vector

    Returns
    -------
    np.array
        The change in field line apex locations.


    Note
    ----
        vector direction refers to the magnetic unit vector direction

    """

    # use spacecraft location to get ECEF
    if ecef_input:
        ecef_xs, ecef_ys, ecef_zs = glats, glons, alts
    else:
        ecef_xs, ecef_ys, ecef_zs = geodetic_to_ecef(glats, glons, alts)

    # prepare output
    apex_edge_length = []

    # take step from s/c along + vector direction
    # then get the apex location
    plus_x, plus_y, plus_z = step_along_mag_unit_vector(ecef_xs, ecef_ys, ecef_zs, dates,
                                                        direction=vector_direction,
                                                        num_steps=edge_steps,
                                                        step_size=edge_length/edge_steps)

    # take half step from s/c along - vector direction
    # then get the apex location
    minus_x, minus_y, minus_z = step_along_mag_unit_vector(ecef_xs, ecef_ys, ecef_zs, dates,
                                                           direction=vector_direction,
                                                           scalar=-1,
                                                           num_steps=edge_steps,
                                                           step_size=edge_length/edge_steps)

    # get apex locations
    if return_geodetic:
        plus_apex_x, plus_apex_y, plus_apex_z, _, _, plus_h = \
            apex_location_info(plus_x, plus_y, plus_z, dates,
                               ecef_input=True,
                               return_geodetic=True)

        minus_apex_x, minus_apex_y, minus_apex_z, _, _, minus_h = \
            apex_location_info(minus_x, minus_y, minus_z, dates,
                               ecef_input=True,
                               return_geodetic=True)
    else:
        plus_apex_x, plus_apex_y, plus_apex_z = \
            apex_location_info(plus_x, plus_y, plus_z, dates,
                               ecef_input=True)

        minus_apex_x, minus_apex_y, minus_apex_z = \
            apex_location_info(minus_x, minus_y, minus_z, dates,
                               ecef_input=True)

    # take difference in apex locations
    apex_edge_length = np.sqrt((plus_apex_x - minus_apex_x)**2 +
                               (plus_apex_y - minus_apex_y)**2 +
                               (plus_apex_z - minus_apex_z)**2)
    if return_geodetic:
        return apex_edge_length, plus_h - minus_h
    else:
        return apex_edge_length


def scalars_for_mapping_ion_drifts(glats, glons, alts, dates,
                                   max_steps=None, e_field_scaling_only=None,
                                   edge_length=None, edge_steps=None,
                                   **kwargs):
    """
    Translates ion drifts and electric fields to equator and footpoints.

    All inputs are assumed to be 1D arrays.

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
    e_field_scaling_only : Deprecated
    max_steps : Deprecated
    edge_length : Deprecated
    edge_steps : Deprecated

    Returns
    -------
    dict
        array-like of scalars for translating ion drifts. Keys are,
        'north_zonal_drifts_scalar', 'north_mer_drifts_scalar', and similarly
        for southern locations. 'equator_mer_drifts_scalar' and
        'equator_zonal_drifts_scalar' cover the mappings to the equator.


    """

    if e_field_scaling_only is not None:
        raise DeprecationWarning('e_field_scaling_only no longer supported.')
    if max_steps is not None:
        raise DeprecationWarning('max_steps no longer supported.')
    if edge_length is not None:
        raise DeprecationWarning('edge_length no longer supported.')
    if edge_steps is not None:
        raise DeprecationWarning('edge_steps no longer supported.')

    # use spacecraft location to get ECEF
    ecef_xs, ecef_ys, ecef_zs = geodetic_to_ecef(glats, glons, alts)

    # get footpoint location information
    north_ftpnt, south_ftpnt = footpoint_location_info(ecef_xs, ecef_ys, ecef_zs,
                                                       dates, ecef_input=True)

    # prepare output memory
    out = {}

    # D and E vectors at user supplied location
    # good for mapping to magnetic equator
    _, _, _, _, _, _, _, _, _, infod = calculate_mag_drift_unit_vectors_ecef(ecef_xs, ecef_ys, ecef_zs, dates,
                                                                             full_output=True,
                                                                             include_debug=True,
                                                                             ecef_input=True,
                                                                             **kwargs)

    out['equator_zon_fields_scalar'] = np.sqrt(infod['e_zon_x']**2 + infod['e_zon_y']**2 + infod['e_zon_z']**2)
    out['equator_mer_fields_scalar'] = np.sqrt(infod['e_mer_x']**2 + infod['e_mer_y']**2 + infod['e_mer_z']**2)

    out['equator_zon_drifts_scalar'] = np.sqrt(infod['d_zon_x']**2 + infod['d_zon_y']**2 + infod['d_zon_z']**2)
    out['equator_mer_drifts_scalar'] = np.sqrt(infod['d_mer_x']**2 + infod['d_mer_y']**2 + infod['d_mer_z']**2)

    # D and E vectors at northern footpoint
    _, _, _, _, _, _, _, _, _, northd = calculate_mag_drift_unit_vectors_ecef(north_ftpnt[:, 0], north_ftpnt[:, 1],
                                                                              north_ftpnt[:, 2], dates,
                                                                              full_output=True,
                                                                              include_debug=True,
                                                                              ecef_input=True,
                                                                              **kwargs)

    # D and E vectors at northern footpoint
    _, _, _, _, _, _, _, _, _, southd = calculate_mag_drift_unit_vectors_ecef(south_ftpnt[:, 0], south_ftpnt[:, 1],
                                                                              south_ftpnt[:, 2], dates,
                                                                              full_output=True,
                                                                              include_debug=True,
                                                                              ecef_input=True,
                                                                              **kwargs)

    # prepare output
    # to map fields from r1 to r2, (E dot e1) d2
    out['north_mer_fields_scalar'] = np.sqrt(infod['e_mer_x']**2 + infod['e_mer_y']**2 + infod['e_mer_z']**2)
    out['north_mer_fields_scalar'] *= np.sqrt(northd['d_mer_x']**2 + northd['d_mer_y']**2 + northd['d_mer_z']**2)
    # to map drifts from r1 to r2, (v dot d1) e2
    out['north_mer_drifts_scalar'] = np.sqrt(infod['d_mer_x']**2 + infod['d_mer_y']**2 + infod['d_mer_z']**2)
    out['north_mer_drifts_scalar'] *= np.sqrt(northd['e_mer_x']**2 + northd['e_mer_y']**2 + northd['e_mer_z']**2)
    # to map fields from r1 to r2, (E dot e1) d2
    out['north_zon_fields_scalar'] = np.sqrt(infod['e_zon_x']**2 + infod['e_zon_y']**2 + infod['e_zon_z']**2)
    out['north_zon_fields_scalar'] *= np.sqrt(northd['d_zon_x']**2 + northd['d_zon_y']**2 + northd['d_zon_z']**2)
    # to map drifts from r1 to r2, (v dot d1) e2
    out['north_zon_drifts_scalar'] = np.sqrt(infod['d_zon_x']**2 + infod['d_zon_y']**2 + infod['d_zon_z']**2)
    out['north_zon_drifts_scalar'] *= np.sqrt(northd['e_zon_x']**2 + northd['e_zon_y']**2 + northd['e_zon_z']**2)

    # to map fields from r1 to r2, (E dot e1) d2
    out['south_mer_fields_scalar'] = np.sqrt(infod['e_mer_x']**2 + infod['e_mer_y']**2 + infod['e_mer_z']**2)
    out['south_mer_fields_scalar'] *= np.sqrt(southd['d_mer_x']**2 + southd['d_mer_y']**2 + southd['d_mer_z']**2)
    # to map drifts from r1 to r2, (v dot d1) e2
    out['south_mer_drifts_scalar'] = np.sqrt(infod['d_mer_x']**2 + infod['d_mer_y']**2 + infod['d_mer_z']**2)
    out['south_mer_drifts_scalar'] *= np.sqrt(southd['e_mer_x']**2 + southd['e_mer_y']**2 + southd['e_mer_z']**2)
    # to map fields from r1 to r2, (E dot e1) d2
    out['south_zon_fields_scalar'] = np.sqrt(infod['e_zon_x']**2 + infod['e_zon_y']**2 + infod['e_zon_z']**2)
    out['south_zon_fields_scalar'] *= np.sqrt(southd['d_zon_x']**2 + southd['d_zon_y']**2 + southd['d_zon_z']**2)
    # to map drifts from r1 to r2, (v dot d1) e2
    out['south_zon_drifts_scalar'] = np.sqrt(infod['d_zon_x']**2 + infod['d_zon_y']**2 + infod['d_zon_z']**2)
    out['south_zon_drifts_scalar'] *= np.sqrt(southd['e_zon_x']**2 + southd['e_zon_y']**2 + southd['e_zon_z']**2)

    return out


def heritage_scalars_for_mapping_ion_drifts(glats, glons, alts, dates, step_size=None,
                                            max_steps=None, e_field_scaling_only=False,
                                            edge_length=25., edge_steps=1,
                                            **kwargs):
    """
    Heritage technique for mapping ion drifts and electric fields.

    Use scalars_for_mapping_ion_drifts instead.

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
        array-like of scalars for translating ion drifts. Keys are,
        'north_zonal_drifts_scalar', 'north_mer_drifts_scalar', and similarly
        for southern locations. 'equator_mer_drifts_scalar' and
        'equator_zonal_drifts_scalar' cover the mappings to the equator.

    Note
    ----
        Directions refer to the ion motion direction e.g. the zonal
        scalar applies to zonal ion motions (meridional E field assuming ExB ion motion)

    """

    if step_size is None:
        step_size = 100.
    if max_steps is None:
        max_steps = 1000
    steps = np.arange(max_steps + 1)

    # use spacecraft location to get ECEF
    ecef_xs, ecef_ys, ecef_zs = geodetic_to_ecef(glats, glons, alts)

    # double edge length, used later
    double_edge = 2.*edge_length

    # prepare output
    eq_zon_drifts_scalar = []
    eq_mer_drifts_scalar = []
    # magnetic field info
    north_mag_scalar = []
    south_mag_scalar = []
    eq_mag_scalar = []
    out = {}
    # meridional e-field scalar map, can also be
    # zonal ion drift scalar map
    north_zon_drifts_scalar = apex_distance_after_footpoint_step(ecef_xs, ecef_ys, ecef_zs,
                                                                 dates, 'north',
                                                                 'meridional',
                                                                 step_size=step_size,
                                                                 max_steps=max_steps,
                                                                 edge_length=edge_length,
                                                                 edge_steps=edge_steps,
                                                                 steps=steps,
                                                                 ecef_input=True,
                                                                 **kwargs)

    north_mer_drifts_scalar = apex_distance_after_footpoint_step(ecef_xs, ecef_ys, ecef_zs,
                                                                 dates, 'north',
                                                                 'zonal',
                                                                 step_size=step_size,
                                                                 max_steps=max_steps,
                                                                 edge_length=edge_length,
                                                                 edge_steps=edge_steps,
                                                                 steps=steps,
                                                                 ecef_input=True,
                                                                 **kwargs)

    south_zon_drifts_scalar = apex_distance_after_footpoint_step(ecef_xs, ecef_ys, ecef_zs,
                                                                 dates, 'south',
                                                                 'meridional',
                                                                 step_size=step_size,
                                                                 max_steps=max_steps,
                                                                 edge_length=edge_length,
                                                                 edge_steps=edge_steps,
                                                                 steps=steps,
                                                                 ecef_input=True,
                                                                 **kwargs)

    south_mer_drifts_scalar = apex_distance_after_footpoint_step(ecef_xs, ecef_ys, ecef_zs,
                                                                 dates, 'south',
                                                                 'zonal',
                                                                 step_size=step_size,
                                                                 max_steps=max_steps,
                                                                 edge_length=edge_length,
                                                                 edge_steps=edge_steps,
                                                                 steps=steps,
                                                                 ecef_input=True,
                                                                 **kwargs)

    eq_zon_drifts_scalar = apex_distance_after_local_step(ecef_xs, ecef_ys, ecef_zs, dates,
                                                          'meridional',
                                                          edge_length=edge_length,
                                                          edge_steps=edge_steps,
                                                          ecef_input=True)
    eq_mer_drifts_scalar = apex_distance_after_local_step(ecef_xs, ecef_ys, ecef_zs, dates,
                                                          'zonal',
                                                          edge_length=edge_length,
                                                          edge_steps=edge_steps,
                                                          ecef_input=True)
    # ratio of apex height difference to step_size across footpoints
    # scales from equator to footpoint
    north_zon_drifts_scalar = north_zon_drifts_scalar/double_edge
    south_zon_drifts_scalar = south_zon_drifts_scalar/double_edge
    north_mer_drifts_scalar = north_mer_drifts_scalar/double_edge
    south_mer_drifts_scalar = south_mer_drifts_scalar/double_edge

    # equatorial
    # scale from s/c to equator
    eq_zon_drifts_scalar = double_edge/eq_zon_drifts_scalar
    eq_mer_drifts_scalar = double_edge/eq_mer_drifts_scalar

    # change scaling from equator to footpoint, to s/c to footpoint
    # via s/c to equator
    north_zon_drifts_scalar *= eq_zon_drifts_scalar
    south_zon_drifts_scalar *= eq_zon_drifts_scalar
    north_mer_drifts_scalar *= eq_mer_drifts_scalar
    south_mer_drifts_scalar *= eq_mer_drifts_scalar

    # prepare output
    out['north_mer_fields_scalar'] = north_zon_drifts_scalar
    out['south_mer_fields_scalar'] = south_zon_drifts_scalar
    out['north_zon_fields_scalar'] = north_mer_drifts_scalar
    out['south_zon_fields_scalar'] = south_mer_drifts_scalar
    out['equator_mer_fields_scalar'] = eq_zon_drifts_scalar
    out['equator_zon_fields_scalar'] = eq_mer_drifts_scalar

    if e_field_scaling_only:
        return out
    else:
        # onward and upward
        # figure out scaling for drifts based upon change in magnetic field
        # strength
        north_ftpnt = np.empty((len(ecef_xs), 3))
        south_ftpnt = np.empty((len(ecef_xs), 3))
        # get location of apex for s/c field line
        apex_xs, apex_ys, apex_zs = apex_location_info(ecef_xs, ecef_ys, ecef_zs,
                                                       dates, ecef_input=True)

        # magnetic field values at spacecraft
        _, _, _, b_sc = magnetic_vector(ecef_xs, ecef_ys, ecef_zs, dates)
        # magnetic field at apex
        _, _, _, b_apex = magnetic_vector(apex_xs, apex_ys, apex_zs, dates)

        north_ftpnt, south_ftpnt = footpoint_location_info(apex_xs, apex_ys,
                                                           apex_zs, dates,
                                                           ecef_input=True)

        # magnetic field at northern footpoint
        _, _, _, b_nft = magnetic_vector(north_ftpnt[:, 0], north_ftpnt[:, 1],
                                         north_ftpnt[:, 2], dates)

        # magnetic field at southern footpoint
        _, _, _, b_sft = magnetic_vector(south_ftpnt[:, 0], south_ftpnt[:, 1],
                                         south_ftpnt[:, 2], dates)
        # scalars account for change in magnetic field between locations
        south_mag_scalar = b_sc/b_sft
        north_mag_scalar = b_sc/b_nft
        eq_mag_scalar = b_sc/b_apex
        # apply to electric field scaling to get ion drift values
        north_zon_drifts_scalar = north_zon_drifts_scalar*north_mag_scalar
        south_zon_drifts_scalar = south_zon_drifts_scalar*south_mag_scalar
        north_mer_drifts_scalar = north_mer_drifts_scalar*north_mag_scalar
        south_mer_drifts_scalar = south_mer_drifts_scalar*south_mag_scalar
        # equatorial
        eq_zon_drifts_scalar = eq_zon_drifts_scalar*eq_mag_scalar
        eq_mer_drifts_scalar = eq_mer_drifts_scalar*eq_mag_scalar
        # output
        out['north_zonal_drifts_scalar'] = north_zon_drifts_scalar
        out['south_zonal_drifts_scalar'] = south_zon_drifts_scalar
        out['north_mer_drifts_scalar'] = north_mer_drifts_scalar
        out['south_mer_drifts_scalar'] = south_mer_drifts_scalar
        out['equator_zonal_drifts_scalar'] = eq_zon_drifts_scalar
        out['equator_mer_drifts_scalar'] = eq_mer_drifts_scalar

    return out
