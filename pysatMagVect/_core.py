
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
from pysatMagVect import igrf as igrf

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

import pysatMagVect.fortran_coords
ecef_to_geodetic = pysatMagVect.fortran_coords.ecef_to_geodetic

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
                     max_steps=1E4, step_size=10., recursive_loop_count=None,
                     recurse=True, min_check_flag=False):
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
    # number of times integration routine must output step location
    if steps is None:
        steps = np.arange(max_steps)
    # ensure date is a float for IGRF call
    if not isinstance(date, float):
        # recast from datetime to float, as required by IGRF12 code
        doy = (date - datetime.datetime(date.year,1,1)).days
        # number of days in year, works for leap years
        num_doy_year = (datetime.datetime(date.year+1,1,1) - datetime.datetime(date.year,1,1)).days
        date = float(date.year) + \
               (float(doy) + float(date.hour + date.minute/60. + date.second/3600.)/24.)/float(num_doy_year+1)

    # set altitude to terminate trace
    if height == 0:
        check_height = 1.
    else:
        check_height = height

    # perform trace
    trace_north, messg = scipy.integrate.odeint(igrf.igrf_step, init.copy(),
                                         steps,
                                         args=(date, step_size, direction, height),
                                         full_output=True,
                                         printmessg=False,
                                         # ixpr=False,
                                         rtol=1.E-11,
                                         atol=1.E-11) #,
                                         # mxstep=500)
    if messg['message'] != 'Integration successful.':
        raise RuntimeError("Field-Line trace not successful.")

    # calculate data to check that we reached final altitude
    check = trace_north[-1, :]
    x, y, z = ecef_to_geodetic(*check)

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
            x, y, z = ecef_to_geodetic(trace_north[:,0], trace_north[:,1], trace_north[:,2])
            idx = np.argmin(np.abs(check_height - z))
            if (z[idx] < check_height*1.001) and (idx > 0):
                trace_north = trace_north[:idx+1,:]
        return trace_north


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
        steps = np.arange(max_steps+1)
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
    trace = np.vstack((trace_south[::-1][:-1,:], trace_north))
    return trace


def calculate_integrated_mag_drift_unit_vectors_ecef(latitude, longitude, altitude, datetimes,
                                          steps=None, max_steps=1000, step_size=100.,
                                          ref_height=120., filter_zonal=True):
    """Calculates unit vectors expressing the ion drift coordinate system
    organized by the geomagnetic field. Unit vectors are expressed
    in ECEF coordinates.

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

    """

    if steps is None:
        steps = np.arange(max_steps+1)
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
    north_x = [];
    north_y = [];
    north_z = []
    south_x = [];
    south_y = [];
    south_z = []
    bn = [];
    be = [];
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
        doy = (time - datetime.datetime(time.year,1,1)).days
        # number of days in year, works for leap years
        num_doy_year = (datetime.datetime(time.year+1,1,1) - datetime.datetime(time.year,1,1)).days
        date = time.year + float(doy)/float(num_doy_year+1)
        date += (time.hour + time.minute/60. + time.second/3600.)/24./float(num_doy_year+1)
        # get IGRF field components
        # tbn, tbe, tbd, tbmag are in nT
        # geodetic input
        tbn, tbe, tbd, tbmag = igrf.igrf12syn(0, date, 1, alt, colat, elong)

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
    norm_foot = np.sqrt(zvx_foot ** 2 + zvy_foot ** 2 + zvz_foot ** 2)

    # calculate zonal vector
    zvx = zvx_foot / norm_foot
    zvy = zvy_foot / norm_foot
    zvz = zvz_foot / norm_foot

    if filter_zonal:
        # print ("Making magnetic vectors orthogonal")
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
    doy = np.array([(time - datetime.datetime(time.year,1,1)).days for time in dates])
    years = np.array([time.year for time in dates])
    num_doy_year = np.array([(datetime.datetime(time.year+1,1,1) - datetime.datetime(time.year,1,1)).days for time in dates])
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
        tbn, tbe, tbd, tbmag = igrf.igrf12syn(0, date, 2, alt, colat, elong)

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


def calculate_mag_drift_unit_vectors_ecef(latitude, longitude, altitude, datetimes,
                                          step_size=0.03, tol=1.E-4, max_loops=100,
                                          full_output=False, tol_zonal_apex=1.E-4,
                                          ecef_input=False, max_steps=None,
                                          ref_height=None, steps=None):
    """Calculates local geomagnetic unit vectors expressing the ion drift
    coordinate system organized by the geomagnetic field. Unit vectors are expressed
    in ECEF coordinates.

    Zonal - Generally Eastward (+East); lies along a surface of constant apex height
    Field Aligned - Generally Northward (+North); points along geomagnetic field
    Meridional - Generally Vertical (+Up); points along the gradient in apex height

    The apex height is the geodetic height of the field line at its highest point.

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
    max_loops : int
        Maximum number of iterations
    tol_zonal_apex : Maximum allowed change in apex height along
        zonal direction
    full_output : bool (False)
        If True, return an additional output parameter (dict) with stats
        about iterative process
    ecef_input : bool (False)
        If True, inputs latitude, longitude, altitude are interpreted as
        x, y, and z in ECEF coordinates (km).
    max_steps : int
        Deprecated
    ref_height : float
        Deprecated
    steps : list-like
        Deprecated

    Returns
    -------
    zon_x, zon_y, zon_z, fa_x, fa_y, fa_z, mer_x, mer_y, mer_z

    Optional output dictionary
    --------------------------
    diff_mer_apex : change in apex height (km) along meridional vector
    diff_mer_vec : magnitude of vector change for last loop
    diff_zonal_apex : change in apex height (km) along zonal vector
    diff_zonal_vec : magnitude of vector change for last loop
    loops : Number of loops
    vector_seed_type : Initial vector used for starting calculation

    """

    if max_steps is not None:
        raise DeprecationWarning('max_steps is no longer supported.')
    if ref_height is not None:
        raise DeprecationWarning('ref_height is no longer supported.')
    if steps is not None:
        raise DeprecationWarning('steps is no longer supported.')
    if step_size <= 0:
        raise ValueError('Step Size must be greater than 0.')

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
            raise RuntimeWarning('Latitude out of bounds [-90., 90.].')
        # ensure longitude reasonable
        idx, = np.where((longitude < -180.) | (longitude > 360.))
        if len(idx) > 0:
            print('Out of spec :', longitude[idx])
            raise RuntimeWarning('Longitude out of bounds [-180., 360.].')

        # calculate satellite position in ECEF coordinates
        ecef_x, ecef_y, ecef_z = geodetic_to_ecef(latitude, longitude, altitude)

    bx, by, bz, bm = magnetic_vector(ecef_x, ecef_y, ecef_z, datetimes, normalize=True)

    # get apex location for root point
    _, _, _, _, _, apex_root = apex_location_info(ecef_x, ecef_y, ecef_z,
                                                  datetimes,
                                                  return_geodetic=True,
                                                  ecef_input=True)

    # need a vector perpendicular to mag field
    # infinitely many
    # let's use the east vector as a great place to start
    tzx, tzy, tzz = enu_to_ecef_vector(np.ones(len(bx)), np.zeros(len(bx)),
                                       np.zeros(len(bx)), latitude, longitude)
    init_type = np.zeros(len(bx)) - 1
    # make sure this vector is well constrained
    # avoid locations where bz near zero
    idx0, = np.where(np.abs(bz) >= 1.E-10)
    if len(idx0) > 0:
        idx, = np.where((np.abs(tzx[idx0]) >= np.abs(tzy[idx0])))
        if len(idx) > 0:
            # force y component to zero and use z-x plane
            tzy[idx0[idx]] = 0
            tzz[idx0[idx]] = -tzx[idx0[idx]]*bx[idx0[idx]]/bz[idx0[idx]]
            init_type[idx0[idx]] = 0
        idx, = np.where((np.abs(tzy[idx0]) > np.abs(tzx[idx0])))
        if len(idx) > 0:
            # force x component to zero and use z-y plane
            tzx[idx0[idx]] = 0
            tzz[idx0[idx]] = -tzy[idx0[idx]]*by[idx0[idx]]/bz[idx0[idx]]
            init_type[idx0[idx]] = 1

    # dealing with locations with small bz
    idx0, = np.where(np.abs(bz) < 1.E-10)
    if len(idx0) > 0:
        idx, = np.where((np.abs(bx[idx0]) > np.abs(by[idx0])) )
        if len(idx) > 0:
            # force z component to zero and use y-x plane
            tzz[idx0[idx]] = 0
            tzx[idx0[idx]] = -tzy[idx0[idx]]*by[idx0[idx]]/bx[idx0[idx]]
            init_type[idx0[idx]] = 2
        idx, = np.where((np.abs(by[idx0]) >= np.abs(bx[idx0])) )
        if len(idx) > 0:
            # force z component to zero and use x-y plane
            tzz[idx0[idx]] = 0
            tzy[idx0[idx]] = -tzx[idx0[idx]]*bx[idx0[idx]]/by[idx0[idx]]
            init_type[idx0[idx]]= 3

    # normalize these vectors
    tzx, tzy, tzz = normalize_vector(tzx, tzy, tzz)

    # need another perpendicular vector to both
    tmx, tmy, tmz = cross_product(tzx, tzy, tzz, bx, by, bz)

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

        # precalculate some info
        ct = np.cos(theta)
        st = np.sin(theta)

        # rotate vectors around unit vector to align along desired gradients
        # zonal along no gradient, meridional along max
        # see wikipedia quaternion spatial rotation page for equation below
        # https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
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
            print('lats:', latitude)
            print('longs', longitude)
            print('alts', altitude)
            raise RuntimeWarning("Didn't converge after reaching max_loops")
            tzx, tzy, tzz = np.nan*tzx, np.nan*tzy, np.nan*tzz
            tmx, tmy, tmz = np.nan*tzx, np.nan*tzy, np.nan*tzz

    zx, zy, zz = tzx, tzy, tzz
    mx, my, mz = tmx, tmy, tmz
    if full_output:
        outd = {'diff_zonal_apex' : diff_apex_z,
                'diff_mer_apex' : diff_apex_m,
                'loops' : loop_num,
                'vector_seed_type' : init_type,
                'diff_zonal_vec' : diff_z,
                'diff_mer_vec' : diff_m
                }
        return zx, zy, zz, bx, by, bz, mx, my, mz, outd

    # return unit vectors for magnetic drifts in ecef coordinates
    return zx, zy, zz, bx, by, bz, mx, my, mz

def step_until_intersect(pos, field_line, sign, time,  direction=None,
                        step_size_goal=5., field_step_size=None,
                        tol=1.E-2, initial_step_factor=3.):
    """Starting at pos, method steps along magnetic unit vector direction
    towards the supplied field line trace. Determines the distance of
    closest approach to field line.

    Routine is used when calculting the mapping of electric fields along
    magnetic field lines. Voltage remains constant along the field but the
    distance between field lines does not. This routine may be used to form the
    last leg when trying to trace out a closed field line loop.

    Routine will create a high resolution field line trace (tol km step size)
    around the location of closest approach (using the field line supplied by
    user) to better determine where the intersection occurs.

    Parameters
    ----------
    pos : array-like
        X, Y, and Z ECEF (km) locations to start from
    field_line : array-like (:,3)
        X, Y, and Z ECEF (km) locations of field line trace, produced by the
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
    step_size_goal : float (5 km)
        step size goal that method will try to match when stepping towards field line.
    field_step_size : float
        step size used (km) for the field_line supplied
    tol : float (.01 km)
        tolerance (uncertainty) allowed (km) for calculating closest
        approach distance

    Returns
    -------
    (float, array, float)
        Total distance taken along vector direction; the position after taking
        the step [x, y, z] in ECEF; distance of closest approach from input pos
        towards the input field line trace.

    """

    field_copy = field_line
    # set a high last minimum distance to ensure first loop does better than this
    best_min_dist = 2500000.
    # scalar is the distance along unit vector line that we are taking
    scalar = 1.
    # repeat boolean
    repeat = True
    # first run boolean
    first = True
    # factor is a divisor applied to the path length step size used when
    # looking for location of closest approach
    factor = 0
    # distance for each step
    step = np.NaN
    while repeat:
        # take a total step along magnetic unit vector
        # try to take steps near user provided step_size_goal
        unit_steps = np.abs(scalar//step_size_goal)
        if unit_steps == 0:
            unit_steps = 1
        # first time through, step_size is 1
        pos_step = step_along_mag_unit_vector(pos[0], pos[1], pos[2], time,
                                              direction=direction,
                                              num_steps=unit_steps,
                                              step_size=np.abs(scalar)/unit_steps,
                                              scalar=sign)
        # find closest point along field line trace
        diff = field_copy - pos_step
        diff_mag = np.sqrt((diff ** 2).sum(axis=1))
        min_idx = np.argmin(diff_mag)
        if first:
            # first time through, the minimum distance just determined
            # uses the coarse field line supplied by the user
            # Now, make a high resolution field line trace around closest distance
            # Take a field step size in each direction to ensure minimum is covered
            # maintain accuracy of high res trace below to be .01 km
            init = field_copy[min_idx,:]
            field_copy = full_field_line(init, time, 0.,
                                         step_size=tol,
                                         max_steps=np.ceil(field_step_size/tol),
                                         recurse=False)
            # difference with position
            diff = field_copy - pos_step
            diff_mag = np.sqrt((diff ** 2).sum(axis=1))
            # find closest one
            min_idx = np.argmin(diff_mag)
            # no longer first run through
            first = False
            # calculate step size for path integration
            step = diff_mag[min_idx]/initial_step_factor
            step_factor = step/(2.**factor)*(-1)**factor

        # pull out distance of closest point
        min_dist = diff_mag[min_idx]

        # check how the solution is doing
        if min_dist > best_min_dist:
            # last step we took made the solution worse
            if np.abs(step_factor) < tol:
                # we've tried enough, stop looping
                repeat = False
            else:
                # undo the last step
                scalar = scalar - step_factor
                # decrease the size of the step
                # and turn around
                factor = factor + 1.
                step_factor = step/(2.**factor)*(-1)**factor
                # perform step
                scalar = scalar + step_factor
        else:
            # things got better
            # we have a new standard to judge against
            # store it
            best_min_dist = min_dist
            best_scalar = scalar
            best_pos_step = pos_step
            # perform another step
            # same size, same direction
            scalar = scalar + step_factor
    # return magnitude of step
    return best_scalar, best_pos_step, best_min_dist


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
        String identifier for which unit vector directino to move along.
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

    """

    for i in np.arange(num_steps):
        # x, y, z in ECEF
        # get unit vector directions
        zvx, zvy, zvz, bx, by, bz, mx, my, mz = calculate_mag_drift_unit_vectors_ecef(
                                                        x, y, z, date,
                                                        step_size=step_size,
                                                        ecef_input=True)
        # pull out the direction we need
        if direction == 'meridional':
            ux, uy, uz = mx, my, mz
        elif direction == 'zonal':
            ux, uy, uz = zvx, zvy, zvz
        elif direction == 'aligned':
            ux, uy, uz = bx, by, bz

        # take steps along direction
        x = x + step_size*ux*scalar
        y = y + step_size*uy*scalar
        z = z + step_size*uz*scalar

    return x, y, z


# create global variables here to reduce overhead in creating input arrays
# _apex_fine_steps = np.arange(6)
# _apex_coarse_steps = np.arange(101)
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
    _apex_coarse_steps = np.arange(max_steps+1)
    # high resolution trace parameters
    _apex_fine_steps = np.arange(fine_max_steps+1)
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
        tlat, tlon, talt = ecef_to_geodetic(trace[:,0], trace[:,1], trace[:,2])
        # determine location that is highest with respect to the geodetic Earth
        max_idx = np.argmax(talt)
        # repeat using a high resolution trace one big step size each
        # direction around identified max
        # recurse False ensures only max_steps are taken
        new_step = step_size
        # print('start range', talt[max_idx-1:max_idx+2])
        while new_step > fine_step_size:
            new_step /= 2.
            trace = full_field_line(trace[max_idx,:], date, 0.,
                                    steps=_apex_fine_steps,
                                    step_size=new_step,
                                    max_steps=fine_max_steps,
                                    recurse=False)
            # convert all locations to geodetic coordinates
            tlat, tlon, talt = ecef_to_geodetic(trace[:,0], trace[:,1], trace[:,2])
            # determine location that is highest with respect to the geodetic Earth
            # print(talt)
            max_idx = np.argmax(talt)

        # collect outputs
        _apex_out_x[i] = trace[max_idx,0]
        _apex_out_y[i] = trace[max_idx,1]
        _apex_out_z[i] = trace[max_idx,2]
        i += 1

    if return_geodetic:
        glat, glon, alt = ecef_to_geodetic(_apex_out_x, _apex_out_y, _apex_out_z)
        return _apex_out_x, _apex_out_y, _apex_out_z, glat, glon, alt
    else:
        return _apex_out_x, _apex_out_y, _apex_out_z

def closed_loop_edge_lengths_via_footpoint(glats, glons, alts, dates, direction,
                                           vector_direction, step_size=None,
                                           max_steps=None, edge_length=25.,
                                           edge_steps=5, **kwargs):
    """
    Forms closed loop integration along mag field, satrting at input
    points and goes through footpoint. At footpoint, steps along vector direction
    in both positive and negative directions, then traces back to opposite
    footpoint. Back at input location, steps toward those new field lines
    (edge_length) along vector direction until hitting distance of minimum
    approach. Loops don't always close. Returns total edge distance
    that goes through input location, along with the distances of closest approach.

    Note
    ----
        vector direction refers to the magnetic unit vector direction

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
    edge_length : float (km)
        Half of total edge length (step) taken at footpoint location.
        edge_length step in both positive and negative directions.
    edge_steps : int
        Number of steps taken from footpoint towards new field line
        in a given direction (positive/negative) along unit vector
    **kwargs : additional keywords
        Passed along to step_until_intersect

    Returns
    -------
    np.array, np.array, np.array
        A closed loop field line path through input location and footpoint in
        northern/southern hemisphere and back is taken. The return edge length
        through input location is provided.

        The distances of closest approach for the positive step along vector
        direction, and the negative step are returned.


    """

    if step_size is None:
        step_size = 100.
    if max_steps is None:
        max_steps = 1000
    steps = np.arange(max_steps+1)

    if direction == 'south':
        direct = -1
    elif direction == 'north':
        direct = 1
    # use spacecraft location to get ECEF
    ecef_xs, ecef_ys, ecef_zs = geodetic_to_ecef(glats, glons, alts)

    # prepare output
    full_local_step = []
    min_distance_plus = []
    min_distance_minus = []

    for ecef_x, ecef_y, ecef_z, glat, glon, alt, date in zip(ecef_xs, ecef_ys, ecef_zs,
                                                             glats, glons, alts,
                                                             dates):
        # going to try and form close loops via field line integration
        # start at location of interest, map down to northern or southern
        # footpoints then take symmetric steps along meridional and zonal
        # directions and trace back from location of interest, step along
        # field line directions until we intersect or hit the distance of
        # closest approach to the return field line with the known
        # distances of footpoint steps, and the closet approach distance
        # we can determine the scalar mapping of one location to another

        yr, doy = pysat.utils.time.getyrdoy(date)
        double_date = float(yr) + float(doy) / 366.

        # trace to footpoint, starting with input location
        sc_root = np.array([ecef_x, ecef_y, ecef_z])
        trace = field_line_trace(sc_root, double_date, direct, 120.,
                                 steps=steps,
                                 step_size=step_size,
                                 max_steps=max_steps)
        # pull out footpoint location
        ftpnt = trace[-1, :]
        ft_glat, ft_glon, ft_alt = ecef_to_geodetic(*ftpnt)
        if np.isnan([ft_glat, ft_glon, ft_alt]).any():
            raise RuntimeError('Unable to find footpoint location, NaN.')
        if ft_alt < 0:
            raise RuntimeError('Footpoint altitude negative.')
        # take step from footpoint along + vector direction
        plus_step = step_along_mag_unit_vector(ftpnt[0], ftpnt[1], ftpnt[2],
                                               date,
                                               direction=vector_direction,
                                               num_steps=edge_steps,
                                               step_size=edge_length/edge_steps)
        # trace this back to other footpoint
        other_plus = field_line_trace(plus_step, double_date, -1*direct, 0.,
                                      steps=steps,
                                      step_size=step_size,
                                      max_steps=max_steps)
        # take half step from first footpoint along - vector direction
        minus_step = step_along_mag_unit_vector(ftpnt[0], ftpnt[1], ftpnt[2],
                                               date,
                                               direction=vector_direction,
                                               scalar=-1,
                                               num_steps=edge_steps,
                                               step_size=edge_length/edge_steps)
        # trace this back to other footpoint
        other_minus = field_line_trace(minus_step, double_date, -1*direct, 0.,
                                       steps=steps,
                                       step_size=step_size,
                                       max_steps=max_steps)
        # need to determine where the intersection of field line coming back from
        # footpoint through postive vector direction step and back
        # in relation to the vector direction from the s/c location.
        pos_edge_length, _, mind_pos = step_until_intersect(sc_root,
                                        other_plus,
                                        1, date,
                                        direction=vector_direction,
                                        field_step_size=step_size,
                                        step_size_goal=edge_length/edge_steps,
                                        **kwargs)
        # take half step from S/C along - vector direction
        minus_edge_length, _, mind_minus = step_until_intersect(sc_root,
                                        other_minus,
                                        -1, date,
                                        direction=vector_direction,
                                        field_step_size=step_size,
                                        step_size_goal=edge_length/edge_steps,
                                        **kwargs)
        # collect outputs
        full_local_step.append(pos_edge_length + minus_edge_length)
        min_distance_plus.append(mind_pos)
        min_distance_minus.append(mind_minus)

    return np.array(full_local_step), np.array(min_distance_plus), np.array(min_distance_minus)

def apex_edge_lengths_via_footpoint(glats, glons, alts, dates, direction,
                                    vector_direction, step_size=None,
                                    max_steps=None, steps=None,
                                    edge_length=25., edge_steps=5,
                                    ecef_input=False):
    """
    Calculates the distance between apex locations mapping to the input location
    footpoints.

    Using the input location, the footpoint location is calculated.
    From here, a step along both the positive and negative
    vector_directions is taken, and the apex locations for those points are calculated.
    The difference in position between these apex locations is the total centered
    distance between magnetic field lines at the magnetic apex when starting
    from the footpoints with a field line half distance of edge_length.

    Note
    ----
        vector direction refers to the magnetic unit vector direction

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



    """

    if step_size is None:
        step_size = 100.
    if max_steps is None:
        max_steps = 1000
    if steps is None:
        steps = np.arange(max_steps+1)

    if direction == 'south':
        direct = -1
    elif direction == 'north':
        direct = 1

    # use spacecraft location to get ECEF
    if ecef_input:
        ecef_xs, ecef_ys, ecef_zs = glats, glons, alts
    else:
        ecef_xs, ecef_ys, ecef_zs = geodetic_to_ecef(glats, glons, alts)

    # prepare output
    apex_edge_length = []
    sc_root = np.array([0, 0, 0])
    ftpnts = np.zeros((len(ecef_xs), 3))
    i = 0
    for ecef_x, ecef_y, ecef_z, date in zip(ecef_xs, ecef_ys, ecef_zs, dates):
        # start at location of interest, map down to northern or southern
        # footpoints then take symmetric steps along meridional and zonal
        # directions and then get apex locations for these fields.
        # The difference between apex location used as field line distance.
        yr, doy = pysat.utils.time.getyrdoy(date)
        double_date = float(yr) + float(doy) / 366.

        # trace to footpoint, starting with input location
        sc_root[:] = [ecef_x, ecef_y, ecef_z]
        trace = field_line_trace(sc_root, double_date, direct, 120.,
                                 steps=steps,
                                 step_size=step_size,
                                 max_steps=max_steps)
        # pull out footpoint location
        ftpnts[i,:] = trace[-1, :]
        i += 1

    # take step from footpoint along + vector direction
    plus_x, plus_y, plus_z = step_along_mag_unit_vector(ftpnts[:,0], ftpnts[:,1], ftpnts[:,2],
                                            dates,
                                            direction=vector_direction,
                                            num_steps=edge_steps,
                                            step_size=edge_length/edge_steps)
    plus_apex_x, plus_apex_y, plus_apex_z = \
                apex_location_info(plus_x, plus_y, plus_z,
                                   dates, ecef_input=True)

    # take half step from first footpoint along - vector direction
    minus_x, minus_y, minus_z = step_along_mag_unit_vector(ftpnts[:,0], ftpnts[:,1], ftpnts[:,2],
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

def closed_loop_edge_lengths_via_equator(glats, glons, alts, dates,
                                         vector_direction,
                                         edge_length=25.,
                                         edge_steps=5,
                                         ecef_input=False):
    """
    Calculates the distance between apex locations mapping to the input location.

    Using the input location, the apex location is calculated. Also from the input
    location, a step along both the positive and negative
    vector_directions is taken, and the apex locations for those points are calculated.
    The difference in position between these apex locations is the total centered
    distance between magnetic field lines at the magnetic apex when starting
    locally with a field line half distance of edge_length.

    An alternative method has been implemented, then commented out.
    This technique takes multiple steps from the origin apex towards the apex
    locations identified along vector_direction. In principle this is more accurate
    but more computationally intensive, similar to the footpoint model.
    A comparison is planned.


    Note
    ----
        vector direction refers to the magnetic unit vector direction

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
    step_size : float (km)
        Step size (km) used for field line integration
    max_steps : int
        Number of steps taken for field line integration
    edge_length : float (km)
        Half of total edge length (step) taken at footpoint location.
        edge_length step in both positive and negative directions.
    edge_steps : int
        Number of steps taken from footpoint towards new field line
        in a given direction (positive/negative) along unit vector

    Returns
    -------
    np.array, ### np.array, np.array
        The change in field line apex locations.

        ## Pending ## The return edge length through input location is provided.

        ## Pending ## The distances of closest approach for the positive step
                      along vector direction, and the negative step are returned.


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
    plus_apex_x, plus_apex_y, plus_apex_z = \
                apex_location_info(plus_x, plus_y, plus_z, dates,
                                   ecef_input=True)

    # take half step from s/c along - vector direction
    # then get the apex location
    minus_x, minus_y, minus_z = step_along_mag_unit_vector(ecef_xs, ecef_ys, ecef_zs, dates,
                                       direction=vector_direction,
                                       scalar=-1,
                                       num_steps=edge_steps,
                                       step_size=edge_length/edge_steps)
    minus_apex_x, minus_apex_y, minus_apex_z = \
                apex_location_info(minus_x, minus_y, minus_z, dates,
                                   ecef_input=True)

    # take difference in apex locations
    apex_edge_length = np.sqrt((plus_apex_x - minus_apex_x)**2 +
                               (plus_apex_y - minus_apex_y)**2 +
                               (plus_apex_z - minus_apex_z)**2)

    return apex_edge_length



def heritage_scalars_for_mapping_ion_drifts(glats, glons, alts, dates, step_size=None,
                                   max_steps=None, e_field_scaling_only=False,
                                   **kwargs):
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
        array-like of scalars for translating ion drifts. Keys are,
        'north_zonal_drifts_scalar', 'north_mer_drifts_scalar', and similarly
        for southern locations. 'equator_mer_drifts_scalar' and
        'equator_zonal_drifts_scalar' cover the mappings to the equator.

    """

    if step_size is None:
        step_size = 100.
    if max_steps is None:
        max_steps = 1000
    steps = np.arange(max_steps+1)

    # use spacecraft location to get ECEF
    ecef_xs, ecef_ys, ecef_zs = geodetic_to_ecef(glats, glons, alts)

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
    # print ('Starting Northern')
    north_zon_drifts_scalar, mind_plus, mind_minus = closed_loop_edge_lengths_via_footpoint(glats,
                                                        glons, alts, dates, 'north',
                                                        'meridional',
                                                        step_size=step_size,
                                                        max_steps=max_steps,
                                                        edge_length=25.,
                                                        edge_steps=5,
                                                        **kwargs)

    north_mer_drifts_scalar, mind_plus, mind_minus = closed_loop_edge_lengths_via_footpoint(glats,
                                                        glons, alts, dates, 'north',
                                                        'zonal',
                                                        step_size=step_size,
                                                        max_steps=max_steps,
                                                        edge_length=25.,
                                                        edge_steps=1,
                                                        **kwargs)

    # print ('Starting Southern')
    south_zon_drifts_scalar, mind_plus, mind_minus = closed_loop_edge_lengths_via_footpoint(glats,
                                                        glons, alts, dates, 'south',
                                                        'meridional',
                                                        step_size=step_size,
                                                        max_steps=max_steps,
                                                        edge_length=25.,
                                                        edge_steps=5,
                                                        **kwargs)

    south_mer_drifts_scalar, mind_plus, mind_minus = closed_loop_edge_lengths_via_footpoint(glats,
                                                        glons, alts, dates, 'south',
                                                        'zonal',
                                                        step_size=step_size,
                                                        max_steps=max_steps,
                                                        edge_length=25.,
                                                        edge_steps=1,
                                                        **kwargs)
    # print ('Starting Equatorial')
    # , step_zon_apex2, mind_plus, mind_minus
    eq_zon_drifts_scalar = closed_loop_edge_lengths_via_equator(glats, glons, alts, dates,
                                                        'meridional',
                                                        edge_length=25.,
                                                        edge_steps=5)
    # , step_mer_apex2, mind_plus, mind_minus
    eq_mer_drifts_scalar = closed_loop_edge_lengths_via_equator(glats, glons, alts, dates,
                                                        'zonal',
                                                        edge_length=25.,
                                                        edge_steps=1)
    # print ('Done with core')
    north_zon_drifts_scalar = north_zon_drifts_scalar/50.
    south_zon_drifts_scalar = south_zon_drifts_scalar/50.
    north_mer_drifts_scalar = north_mer_drifts_scalar/50.
    south_mer_drifts_scalar = south_mer_drifts_scalar/50.
    # equatorial
    eq_zon_drifts_scalar = 50./eq_zon_drifts_scalar
    eq_mer_drifts_scalar = 50./eq_mer_drifts_scalar

    if e_field_scaling_only:
        # prepare output
        out['north_mer_fields_scalar'] = north_zon_drifts_scalar
        out['south_mer_fields_scalar'] = south_zon_drifts_scalar
        out['north_zon_fields_scalar'] = north_mer_drifts_scalar
        out['south_zon_fields_scalar'] = south_mer_drifts_scalar
        out['equator_mer_fields_scalar'] = eq_zon_drifts_scalar
        out['equator_zon_fields_scalar'] = eq_mer_drifts_scalar

    else:
        # figure out scaling for drifts based upon change in magnetic field
        # strength
        for ecef_x, ecef_y, ecef_z, glat, glon, alt, date in zip(ecef_xs, ecef_ys, ecef_zs,
                                                                glats, glons, alts,
                                                                dates):
            yr, doy = pysat.utils.time.getyrdoy(date)
            double_date = float(yr) + float(doy) / 366.
            # get location of apex for s/c field line
            apex_x, apex_y, apex_z, apex_lat, apex_lon, apex_alt = apex_location_info(
                                                                        [glat], [glon],
                                                                        [alt], [date],
                                                                        return_geodetic=True)
            # trace to northern footpoint
            sc_root = np.array([ecef_x, ecef_y, ecef_z])
            trace_north = field_line_trace(sc_root, double_date, 1., 120.,
                                        steps=steps,
                                        step_size=step_size,
                                        max_steps=max_steps)
            # southern tracing
            trace_south = field_line_trace(sc_root, double_date, -1., 120.,
                                        steps=steps,
                                        step_size=step_size,
                                        max_steps=max_steps)
            # footpoint location
            north_ftpnt = trace_north[-1, :]
            nft_glat, nft_glon, nft_alt = ecef_to_geodetic(*north_ftpnt)
            south_ftpnt = trace_south[-1, :]
            sft_glat, sft_glon, sft_alt = ecef_to_geodetic(*south_ftpnt)

            # scalar for the northern footpoint electric field based on distances
            # for drift also need to include the magnetic field, drift = E/B
            tbn, tbe, tbd, b_sc = igrf.igrf12syn(0, double_date, 1, alt,
                                                np.deg2rad(90.-glat),
                                                np.deg2rad(glon))
            # get mag field and scalar for northern footpoint
            tbn, tbe, tbd, b_nft = igrf.igrf12syn(0, double_date, 1, nft_alt,
                                                np.deg2rad(90.-nft_glat),
                                                np.deg2rad(nft_glon))
            north_mag_scalar.append(b_sc/b_nft)
            # equatorial values
            tbn, tbe, tbd, b_eq = igrf.igrf12syn(0, double_date, 1, apex_alt,
                                                 np.deg2rad(90.-apex_lat),
                                                 np.deg2rad(apex_lon))
            eq_mag_scalar.append(b_sc/b_eq)
            # scalar for the southern footpoint
            tbn, tbe, tbd, b_sft = igrf.igrf12syn(0, double_date, 1, sft_alt,
                                                  np.deg2rad(90.-sft_glat),
                                                  np.deg2rad(sft_glon))
            south_mag_scalar.append(b_sc/b_sft)

        # make E-Field scalars to drifts
        # lists to arrays
        north_mag_scalar = np.array(north_mag_scalar)
        south_mag_scalar = np.array(south_mag_scalar)
        eq_mag_scalar = np.array(eq_mag_scalar)
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

def scalars_for_mapping_ion_drifts(glats, glons, alts, dates, step_size=None,
                                   max_steps=None, e_field_scaling_only=False,
                                   edge_length=25., edge_steps=1,
                                   **kwargs):
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
        array-like of scalars for translating ion drifts. Keys are,
        'north_zonal_drifts_scalar', 'north_mer_drifts_scalar', and similarly
        for southern locations. 'equator_mer_drifts_scalar' and
        'equator_zonal_drifts_scalar' cover the mappings to the equator.

    """

    if step_size is None:
        step_size = 100.
    if max_steps is None:
        max_steps = 1000
    steps = np.arange(max_steps+1)

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
    # print ('Starting Northern')
    north_zon_drifts_scalar = apex_edge_lengths_via_footpoint(ecef_xs, ecef_ys, ecef_zs,
                                                dates, 'north',
                                                'meridional',
                                                step_size=step_size,
                                                max_steps=max_steps,
                                                edge_length=edge_length,
                                                edge_steps=edge_steps,
                                                steps=steps,
                                                ecef_input=True,
                                                **kwargs)

    north_mer_drifts_scalar = apex_edge_lengths_via_footpoint(ecef_xs, ecef_ys, ecef_zs,
                                                dates, 'north',
                                                'zonal',
                                                step_size=step_size,
                                                max_steps=max_steps,
                                                edge_length=edge_length,
                                                edge_steps=edge_steps,
                                                steps=steps,
                                                ecef_input=True,
                                                **kwargs)

    # print ('Starting Southern')
    south_zon_drifts_scalar = apex_edge_lengths_via_footpoint(ecef_xs, ecef_ys, ecef_zs,
                                                dates, 'south',
                                                'meridional',
                                                step_size=step_size,
                                                max_steps=max_steps,
                                                edge_length=edge_length,
                                                edge_steps=edge_steps,
                                                steps=steps,
                                                ecef_input=True,
                                                **kwargs)

    south_mer_drifts_scalar = apex_edge_lengths_via_footpoint(ecef_xs, ecef_ys, ecef_zs,
                                                dates, 'south',
                                                'zonal',
                                                step_size=step_size,
                                                max_steps=max_steps,
                                                edge_length=edge_length,
                                                edge_steps=edge_steps,
                                                steps=steps,
                                                ecef_input=True,
                                                **kwargs)
    # print ('Starting Equatorial')
    eq_zon_drifts_scalar = closed_loop_edge_lengths_via_equator(ecef_xs, ecef_ys, ecef_zs, dates,
                                                        'meridional',
                                                        edge_length=edge_length,
                                                        edge_steps=edge_steps,
                                                        ecef_input=True)
    eq_mer_drifts_scalar = closed_loop_edge_lengths_via_equator(ecef_xs, ecef_ys, ecef_zs, dates,
                                                        'zonal',
                                                        edge_length=edge_length,
                                                        edge_steps=edge_steps,
                                                        ecef_input=True)
    # print ('Done with core')
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

        sc_root = np.array([0, 0, 0])
        i = 0
        for apex_x, apex_y, apex_z, date in zip(apex_xs, apex_ys, apex_zs, dates):
            yr, doy = pysat.utils.time.getyrdoy(date)
            double_date = float(yr) + float(doy) / 366.

            sc_root[:] = (apex_x, apex_y, apex_z)
            trace_north = field_line_trace(sc_root, double_date, 1., 120.,
                                        steps=steps,
                                        step_size=step_size,
                                        max_steps=max_steps)
            # southern tracing
            trace_south = field_line_trace(sc_root, double_date, -1., 120.,
                                        steps=steps,
                                        step_size=step_size,
                                        max_steps=max_steps)
            # footpoint location
            north_ftpnt[i, :] = trace_north[-1, :]
            south_ftpnt[i, :] = trace_south[-1, :]
            i += 1

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
#         for ecef_x, ecef_y, ecef_z, glat, glon, alt, date in zip(ecef_xs, ecef_ys, ecef_zs,
#                                                                 glats, glons, alts,
#                                                                 dates):
#             yr, doy = pysat.utils.time.getyrdoy(date)
#             double_date = float(yr) + float(doy) / 366.
#             # get location of apex for s/c field line
#             apex_x, apex_y, apex_z, apex_lat, apex_lon, apex_alt = apex_location_info(
#                                                                         [glat], [glon],
#                                                                         [alt], [date],
#                                                                         return_geodetic=True)
#             # trace to northern footpoint
#             sc_root = np.array([ecef_x, ecef_y, ecef_z])
#             trace_north = field_line_trace(sc_root, double_date, 1., 120.,
#                                         steps=steps,
#                                         step_size=step_size,
#                                         max_steps=max_steps)
#             # southern tracing
#             trace_south = field_line_trace(sc_root, double_date, -1., 120.,
#                                         steps=steps,
#                                         step_size=step_size,
#                                         max_steps=max_steps)
#             # footpoint location
#             north_ftpnt = trace_north[-1, :]
#             nft_glat, nft_glon, nft_alt = ecef_to_geodetic(*north_ftpnt)
#             south_ftpnt = trace_south[-1, :]
#             sft_glat, sft_glon, sft_alt = ecef_to_geodetic(*south_ftpnt)
#
#             # scalar for the northern footpoint electric field based on distances
#             # for drift also need to include the magnetic field, drift = E/B
#             tbn, tbe, tbd, b_sc = igrf.igrf12syn(0, double_date, 1, alt,
#                                                 np.deg2rad(90.-glat),
#                                                 np.deg2rad(glon))
#             # get mag field and scalar for northern footpoint
#             tbn, tbe, tbd, b_nft = igrf.igrf12syn(0, double_date, 1, nft_alt,
#                                                 np.deg2rad(90.-nft_glat),
#                                                 np.deg2rad(nft_glon))
#             north_mag_scalar.append(b_sc/b_nft)
#             # equatorial values
#             tbn, tbe, tbd, b_eq = igrf.igrf12syn(0, double_date, 1, apex_alt,
#                                                  np.deg2rad(90.-apex_lat),
#                                                  np.deg2rad(apex_lon))
#             eq_mag_scalar.append(b_sc/b_eq)
#             # scalar for the southern footpoint
#             tbn, tbe, tbd, b_sft = igrf.igrf12syn(0, double_date, 1, sft_alt,
#                                                   np.deg2rad(90.-sft_glat),
#                                                   np.deg2rad(sft_glon))
#             south_mag_scalar.append(b_sc/b_sft)
#
#         # make E-Field scalars to drifts
#         # lists to arrays
#         north_mag_scalar = np.array(north_mag_scalar)
#         south_mag_scalar = np.array(south_mag_scalar)
#         eq_mag_scalar = np.array(eq_mag_scalar)
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
