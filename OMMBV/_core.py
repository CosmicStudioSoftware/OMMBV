"""
Supporting routines for coordinate conversions as well as vector operations and
transformations used in Space Science.
"""

import scipy
import scipy.integrate
import numpy as np
import datetime as dt
import warnings


# Import reference IGRF fortran code within the package if not on RTD
try:
    from OMMBV import igrf as igrf, enu_to_ecef_vector, normalize_vector, cross_product
except:
    pass

import OMMBV.trans as trans
import OMMBV.utils
from OMMBV import vector



def field_line_trace(init, date, direction, height, steps=None,
                     max_steps=1E4, step_size=10., recursive_loop_count=None,
                     recurse=True, min_check_flag=False):
    """Perform field line tracing using IGRF and scipy.integrate.odeint.

    After 500 recursive iterations this method will increase the step size by
    3% every subsequent iteration.

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
    steps : array-like of ints, floats, or NoneType
        Number of steps along field line when field line trace positions should
        be reported. By default, each step is reported;
        steps=np.arange(max_steps). (default=None)
    max_steps : float
        Maximum number of steps along field line that should be taken
        (default=1E4)
    step_size : float
        Distance in km for each large integration step. Multiple substeps
        are taken as determined by scipy.integrate.odeint. (default=10.)
    recursive_loop_count : int
        Used internally by the method. Not intended for the user.
    recurse : bool
        If True, method will recursively call itself to complete field line
        trace as necessary. (default=True)
    min_check_flag : bool
        If True, performs an additional check that the field line
        tracing reached the target altitude. Removes any redundant
        steps below the target altitude. (default=False)

    Returns
    -------
    numpy array
        2D array. [0,:] has the x,y,z location for initial point
        [:,0] is the x positions over the integration.
        Positions are reported in ECEF (km).


    """

    if recursive_loop_count is None:
        recursive_loop_count = 0

    # Number of times integration routine must output step location
    if steps is None:
        steps = np.arange(max_steps)

    # Ensure date is a float for IGRF call
    if isinstance(date, dt.datetime):
        date = OMMBV.utils.datetimes_to_doubles([date])[0]

    # Set altitude to terminate trace
    if height == 0:
        check_height = 1.
    else:
        check_height = height

    # Perform trace
    trace_north, messg = scipy.integrate.odeint(igrf.igrf_step, init.copy(),
                                                steps,
                                                args=(date, step_size,
                                                      direction, height),
                                                full_output=True,
                                                printmessg=False,
                                                ixpr=False,
                                                rtol=1.E-11,
                                                atol=1.E-11)

    if messg['message'] != 'Integration successful.':
        estr = "Field-Line trace not successful."
        warnings.warn(estr)
        return np.full((1, 3), np.nan)

    # Calculate data to check that we reached final altitude
    check = trace_north[-1, :]
    x, y, z = trans.ecef_to_geodetic(*check)

    # Fortran integration gets close to target height
    loop_step = step_size
    if recurse & (z > check_height*1.000001):
        if recursive_loop_count > 500:
            loop_step *= 1.03
        if recursive_loop_count < 1000:
            # When we have not reached the reference height, call
            # field_line_trace again by taking check value as init.
            # Recursive call.
            recursive_loop_count = recursive_loop_count + 1
            trace_north1 = field_line_trace(check, date, direction, height,
                                            step_size=loop_step,
                                            max_steps=max_steps,
                                            recursive_loop_count=recursive_loop_count,
                                            steps=steps)
        else:
            estr = "After 1000 iterations couldn't reach target altitude"
            warnings.warn(estr)
            return np.full((1, 3), np.nan)

        # Append new trace data to existing trace data
        # this return is taken as part of recursive loop
        if np.isnan(trace_north1[-1, 0]):
            return trace_north1
        else:
            return np.vstack((trace_north, trace_north1))

    else:
        # return results if we make it to the target altitude

        # filter points to terminate at point closest to target height
        # code also introduces a variable length return, though I suppose
        # that already exists with the recursive functionality
        # while this check is done internally within Fortran integrand, if
        # that steps out early, the output we receive would be problematic.
        # Steps below provide an extra layer of security that output has some
        # semblance to expectations
        if min_check_flag:
            x, y, z = trans.ecef_to_geodetic(trace_north[:, 0],
                                             trace_north[:, 1],
                                             trace_north[:, 2])
            idx = np.argmin(np.abs(check_height - z))
            if (z[idx] < check_height * 1.001) and (idx > 0):
                trace_north = trace_north[:idx + 1, :]

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
    if not np.isnan(trace_north[-1, 0]) and (not np.isnan(trace_south[-1, 0])):
        trace = np.vstack((trace_south[::-1][:-1, :], trace_north))
    else:
        trace = np.full((1, 3), np.nan)
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
    ecef_x, ecef_y, ecef_z = trans.geodetic_to_ecef(latitude, longitude,
                                                    altitude)
    # also get position in geocentric coordinates
    geo_lat, geo_long, geo_alt = trans.ecef_to_geocentric(ecef_x, ecef_y,
                                                          ecef_z,
                                                          ref_height=0.)
    # geo_lat, geo_long, geo_alt = trans.ecef_to_geodetic(ecef_x, ecef_y,
    # ecef_z)

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

    ddates = OMMBV.utils.datetimes_to_doubles(datetimes)

    for x, y, z, alt, colat, elong, time in zip(ecef_x, ecef_y, ecef_z,
                                                altitude, np.deg2rad(90. - latitude),
                                                np.deg2rad(longitude), ddates):
        init = np.array([x, y, z], dtype=np.float64)
        trace = full_field_line(init, time, ref_height, step_size=step_size,
                                max_steps=max_steps,
                                steps=steps)
        # store final location, full trace goes south to north
        trace_north = trace[-1, :]
        trace_south = trace[0, :]

        # get IGRF field components
        # tbn, tbe, tbd, tbmag are in nT
        # geodetic input
        tbn, tbe, tbd, tbmag = igrf.igrf13syn(0, time, 1, alt, colat, elong)

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
    north_x, north_y, north_z = vector.normalize(north_x, north_y, north_z)
    south_x = south_x - ecef_x
    south_y = south_y - ecef_y
    south_z = south_z - ecef_z
    south_x, south_y, south_z = vector.normalize(south_x, south_y, south_z)
    # calculate magnetic unit vector
    bx, by, bz = enu_to_ecef_vector(be, bn, -bd, geo_lat, geo_long)
    bx, by, bz = vector.normalize(bx, by, bz)

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
        zvx, zvy, zvz = vector.normalize(zvx, zvy, zvz)

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

    # Prepare output lists
    bn = []
    be = []
    bd = []
    bm = []

    # Need a double variable for time.
    ddates = OMMBV.utils.datetimes_to_doubles(dates)

    # Use geocentric coordinates for calculating magnetic field since
    # transformation between it and ECEF is robust. The geodetic translations
    # introduce error.
    latitudes, longitudes, altitudes = trans.ecef_to_geocentric(x, y, z,
                                                                ref_height=0.)

    # Calculate magnetic field value for all user provided locations
    colats = np.deg2rad(90. - latitudes)
    longs = np.deg2rad(longitudes)
    for colat, elong, alt, date in zip(colats, longs, altitudes, ddates):
        # tbn, tbe, tbd, tbmag are in nT
        tbn, tbe, tbd, tbmag = igrf.igrf13syn(0, date, 2, alt, colat, elong)

        # Collect outputs
        bn.append(tbn)
        be.append(tbe)
        bd.append(tbd)
        bm.append(tbmag)

    # Repackage
    bn = np.array(bn, dtype=np.float64)
    be = np.array(be, dtype=np.float64)
    bd = np.array(bd, dtype=np.float64)
    bm = np.array(bm, dtype=np.float64)

    # Convert to ECEF basis
    bx, by, bz = enu_to_ecef_vector(be, bn, -bd, latitudes, longitudes)

    if normalize:
        bx /= bm
        by /= bm
        bz /= bm

    return bx, by, bz, bm


def apex_location_info(glats, glons, alts, dates, step_size=100.,
                       fine_step_size=1.E-5, fine_max_steps=5,
                       return_geodetic=False, ecef_input=False):
    """Determine apex location for the field line passing through input point.

    Employs a two stage method. A broad step (`step_size`) field line trace
    spanning Northern/Southern footpoints is used to find the location with
    the largest geodetic (WGS84) height. A binary search higher resolution
    trace is then used to get a better fix on this location. Each loop,
    the step_size halved. Greatest geodetic height is once
    again selected once the step_size is below `fine_step_size`.

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
    step_size : float
        Step size (km) used for tracing coarse field line. (default=100)
    fine_step_size : float
        Fine step size (km) for refining apex location height. (default=1.E-5)
    fine_max_steps : int
        Fine number of steps passed along to full_field_trace. Do not
        change, generally. (default=5)
    return_geodetic: bool
        If True, also return location in geodetic coordinates
    ecef_input : bool
        If True, glats, glons, and alts are treated as x, y, z (ECEF).
        (default=False)

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
        ecef_xs, ecef_ys, ecef_zs = trans.geodetic_to_ecef(glats, glons, alts)

    # Prepare parameters for field line trace
    max_steps = 100
    apex_coarse_steps = np.arange(max_steps + 1)

    # High resolution trace parameters
    apex_fine_steps = np.arange(fine_max_steps + 1)

    # Prepare output
    apex_out_x = np.full(len(ecef_xs), np.nan, dtype=np.float64)
    apex_out_y = np.full(len(ecef_xs), np.nan, dtype=np.float64)
    apex_out_z = np.full(len(ecef_xs), np.nan, dtype=np.float64)

    i = 0
    for ecef_x, ecef_y, ecef_z, date in zip(ecef_xs, ecef_ys, ecef_zs, dates):

        # Ensure starting location is valid
        if np.any(np.isnan([ecef_x, ecef_y, ecef_z])):
            apex_out_x[i] = np.nan
            apex_out_y[i] = np.nan
            apex_out_z[i] = np.nan
            i += 1
            continue

        # To get the apex location we need to do a field line trace
        # then find the highest point.
        trace = full_field_line(np.array([ecef_x, ecef_y, ecef_z],
                                         dtype=np.float64),
                                date, 0.,
                                steps=apex_coarse_steps,
                                step_size=step_size,
                                max_steps=max_steps)

        # Convert all locations to geodetic coordinates
        tlat, tlon, talt = trans.ecef_to_geodetic(trace[:, 0], trace[:, 1],
                                                  trace[:, 2])

        # Determine location that is highest with respect to the geodetic Earth
        max_idx = np.argmax(talt)

        # Repeat using a high resolution trace one big step size each
        # direction around identified max. It is possible internal functions
        # had to increase step size. Get step estimate from trace.
        if max_idx > 0:
            new_step = np.sqrt((trace[max_idx, 0] - trace[max_idx - 1, 0])**2
                               + (trace[max_idx, 1] - trace[max_idx - 1, 1])**2
                               + (trace[max_idx, 2] - trace[max_idx - 1, 2])**2)
        elif max_idx < len(talt) - 1:
            new_step = np.sqrt((trace[max_idx, 0] - trace[max_idx + 1, 0])**2
                               + (trace[max_idx, 1] - trace[max_idx + 1, 1])**2
                               + (trace[max_idx, 2] - trace[max_idx + 1, 2])**2)
        else:
            new_step = step_size

        while new_step > fine_step_size:
            new_step /= 2.

            # Setting recurse False ensures only max_steps are taken.
            trace = full_field_line(trace[max_idx, :], date, 0.,
                                    steps=apex_fine_steps,
                                    step_size=new_step,
                                    max_steps=fine_max_steps,
                                    recurse=False)

            # Convert all locations to geodetic coordinates
            tlat, tlon, talt = trans.ecef_to_geodetic(trace[:, 0], trace[:, 1],
                                                      trace[:, 2])
            # Determine location that is highest with respect to the
            # geodetic Earth.
            max_idx = np.argmax(talt)

        # Collect outputs
        apex_out_x[i] = trace[max_idx, 0]
        apex_out_y[i] = trace[max_idx, 1]
        apex_out_z[i] = trace[max_idx, 2]
        i += 1

    if return_geodetic:
        glat, glon, alt = trans.ecef_to_geodetic(apex_out_x, apex_out_y,
                                                 apex_out_z)
        return apex_out_x, apex_out_y, apex_out_z, glat, glon, alt
    else:
        return apex_out_x, apex_out_y, apex_out_z


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

        d_zon_x (y,z) : D zonal vector components. ECEF X, Y, and Z directions
        d_mer_x (y,z) : D meridional vector components. ECEF X, Y, and Z.
        d_fa_x (y,z) : D field aligned vector components.  ECEF X, Y, and Z.

        e_zon_mag: E zonal vector magnitude
        e_fa_mag: E field-aligned vector magnitude
        e_mer_mag: E meridional vector magnitude

        e_zon_x (y,z) : E zonal vector components. ECEF X, Y, and Z directions.
        e_mer_x (y,z) : E meridional vector components. ECEF X, Y, and Z.
        e_fa_x (y,z) : E field aligned vector components. ECEF X, Y, and Z.

    """

    (zx, zy, zz,
     fx, fy, fz,
     mx, my, mz,
     info) = calculate_mag_drift_unit_vectors_ecef(latitude, longitude,
                                                     altitude, datetimes,
                                                     full_output=True)

    d_zon_mag = np.sqrt(info['d_zon_x']**2 + info['d_zon_y']**2
                        + info['d_zon_z']**2)
    d_fa_mag = np.sqrt(info['d_fa_x']**2 + info['d_fa_y']**2
                       + info['d_fa_z']**2)
    d_mer_mag = np.sqrt(info['d_mer_x']**2 + info['d_mer_y']**2
                        + info['d_mer_z']**2)

    e_zon_mag = np.sqrt(info['e_zon_x']**2 + info['e_zon_y']**2
                        + info['e_zon_z']**2)
    e_fa_mag = np.sqrt(info['e_fa_x']**2 + info['e_fa_y']**2
                       + info['e_fa_z']**2)
    e_mer_mag = np.sqrt(info['e_mer_x']**2 + info['e_mer_y']**2
                        + info['e_mer_z']**2)

    # Assemble output dictionary
    out_d = {'zon_x': zx, 'zon_y': zy, 'zon_z': zz,
             'fa_x': fx, 'fa_y': fy, 'fa_z': fz,
             'mer_x': mx, 'mer_y': my, 'mer_z': mz,
             'd_zon_x': info['d_zon_x'], 'd_zon_y': info['d_zon_y'],
             'd_zon_z': info['d_zon_z'],
             'd_fa_x': info['d_fa_x'], 'd_fa_y': info['d_fa_y'],
             'd_fa_z': info['d_fa_z'],
             'd_mer_x': info['d_mer_x'], 'd_mer_y': info['d_mer_y'],
             'd_mer_z': info['d_mer_z'],
             'e_zon_x': info['e_zon_x'], 'e_zon_y': info['e_zon_y'],
             'e_zon_z': info['e_zon_z'],
             'e_fa_x': info['e_fa_x'], 'e_fa_y': info['e_fa_y'],
             'e_fa_z': info['e_fa_z'],
             'e_mer_x': info['e_mer_x'], 'e_mer_y': info['e_mer_y'],
             'e_mer_z': info['e_mer_z'],
             'd_zon_mag': d_zon_mag, 'd_fa_mag': d_fa_mag, 'd_mer_mag': d_mer_mag,
             'e_zon_mag': e_zon_mag, 'e_fa_mag': e_fa_mag, 'e_mer_mag': e_mer_mag}

    return out_d


def calculate_mag_drift_unit_vectors_ecef(latitude, longitude, altitude,
                                          datetimes, step_size=0.5, tol=1.E-4,
                                          tol_zonal_apex=1.E-4, max_loops=10,
                                          ecef_input=False, centered_diff=True,
                                          full_output=False,
                                          include_debug=False,
                                          scalar=None, edge_steps=None,
                                          dstep_size=0.5, max_steps=None,
                                          ref_height=None, steps=None,
                                          pole_tol=1.E-5,
                                          location_info=apex_location_info):
    """Calculates local geomagnetic basis vectors and mapping scalars.

    Zonal - Generally Eastward (+East); surface of constant apex height
    Field Aligned - Generally Northward (+North); points along geomagnetic field
    Meridional - Generally Vertical (+Up); gradient in apex height

    The apex height is the geodetic height of the field line at its highest
    point. Unit vectors are expressed in Earth Centered Earth Fixed (ECEF)
    coordinates.

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
        (default=0.5)
    tol : float
        Tolerance goal for the magnitude of the change in unit vectors per loop
        (default=1.E-4)
    tol_zonal_apex : float
        Maximum allowed change in apex height along zonal direction
        (default=1.E-4)
    max_loops : int
        Maximum number of iterations (default=100)
    ecef_input : bool
        If True, inputs latitude, longitude, altitude are interpreted as
        x, y, and z in ECEF coordinates (km). (default=False)
    full_output : bool
        If True, return an additional dictionary with the E and D mapping
        vectors. (default=False)
    include_debug : bool
        If True, include stats about iterative process in optional dictionary.
        Requires full_output=True. (default=False)
    centered_diff : bool
        If True, a symmetric centered difference is used when calculating
        the change in apex height along the zonal direction, used within
        the zonal unit vector calculation. (default=True)
    scalar : NoneType
        Deprecated.
    edge_steps : NoneType
        Deprecated.
    dstep_size : float
        Step size (km) used when calculating the expansion of field line
        surfaces. Generally, this should be the same as step_size. (default=0.5)
    max_steps : NoneType
        Deprecated
    ref_height : NoneType
        Deprecated
    steps : NoneType
        Deprecated
    location_info : function
        Function used to determine a consistent relative position along a
        field line. Should not generally be modified.
        (default=apex_location_info)
    pole_tol : float
        When upward component of magnetic is within `pole_tol` of 1, the
        system will treat location as a pole and will not attempt to
        calculate unit vectors and scalars. (default=1.E-5)

    Returns
    -------
    zon_x, zon_y, zon_z, fa_x, fa_y, fa_z, mer_x, mer_y, mer_z, (optional dictionary)

    Optional output dictionary
    --------------------------
    Full Output Parameters

    d_zon_x (y,z) : D zonal vector components along ECEF X, Y, and Z directions
    d_mer_x (y,z) : D meridional vector components along ECEF X, Y, and Z
    d_fa_x (y,z) : D field aligned vector components along ECEF X, Y, and Z

    e_zon_x (y,z) : E zonal vector components along ECEF X, Y, and Z directions
    e_mer_x (y,z) : E meridional vector components along ECEF X, Y, and Z
    e_fa_x (y,z) : E field aligned vector components along ECEF X, Y, and Z


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

        The method terminates when successive updates to both the zonal and
        meridional unit vectors differ (magnitude of difference) by less than
        `tol`, and the change in apex_height from input location is less than
        `tol_zonal_apex`.

    """

    # Check for deprecated inputs
    if max_steps is not None:
        warnings.warn('max_steps is no longer supported.', DeprecationWarning)
    if ref_height is not None:
        warnings.warn('ref_height is no longer supported.', DeprecationWarning)
    if steps is not None:
        warnings.warn('steps is no longer supported.', DeprecationWarning)
    if scalar is not None:
        warnings.warn('scalar is no longer supported.', DeprecationWarning)
    if edge_steps is not None:
        warnings.warn('edge_steps is no longer supported.', DeprecationWarning)

    # Check for reasonable user inputs
    if step_size <= 0:
        raise ValueError('Step Size must be greater than 0.')

    if ecef_input:
        ecef_x, ecef_y, ecef_z = latitude, longitude, altitude
        # lat and long needed for initial zonal and meridional vector
        # generation later on
        latitude, longitude, altitude = trans.ecef_to_geodetic(ecef_x, ecef_y,
                                                               ecef_z)
    else:
        latitude = np.array(latitude, dtype=np.float64)
        longitude = np.array(longitude, dtype=np.float64)
        altitude = np.array(altitude, dtype=np.float64)

        # Ensure latitude reasonable
        idx, = np.where(np.abs(latitude) > 90.)
        if len(idx) > 0:
            raise RuntimeError('Latitude out of bounds [-90., 90.].')

        # Ensure longitude reasonable
        idx, = np.where((longitude < -180.) | (longitude > 360.))
        if len(idx) > 0:
            print('Out of spec :', longitude[idx])
            raise RuntimeError('Longitude out of bounds [-180., 360.].')

        # Calculate satellite position in ECEF coordinates
        ecef_x, ecef_y, ecef_z = trans.geodetic_to_ecef(latitude, longitude,
                                                  altitude)

    # Check for reasonable user inputs
    if len(latitude) != len(longitude) != len(altitude) != len(datetimes):
        estr = ''.join(['All inputs (latitude, longitude, altitude, datetimes)',
                        ' must have the same length.'])
        raise ValueError(estr)

    # Begin method calculation.

    # Magnetic field at root location
    bx, by, bz, bm = magnetic_vector(ecef_x, ecef_y, ecef_z, datetimes,
                                     normalize=True)

    # If magnetic field is pointed purely upward, then full basis can't
    # be calculated. Check for this condition and store locations.
    be, bn, bu = vector.ecef_to_enu_vector(bx, by, bz, latitude, longitude)
    null_idx, = np.where(np.abs(bu) > 1. - pole_tol)

    # To start, need a vector perpendicular to mag field. There are infinitely
    # many, thus, let's use the east vector as a start.
    tzx, tzy, tzz = enu_to_ecef_vector(np.ones(len(bx)), np.zeros(len(bx)),
                                       np.zeros(len(bx)), latitude, longitude)
    init_type = np.zeros(len(bx)) - 1

    # Get meridional direction via cross with field-aligned and normalize
    tmx, tmy, tmz = cross_product(tzx, tzy, tzz, bx, by, bz)
    tmx, tmy, tmz = vector.normalize(tmx, tmy, tmz)

    # Get orthogonal zonal now, and normalize.
    tzx, tzy, tzz = cross_product(bx, by, bz, tmx, tmy, tmz)
    tzx, tzy, tzz = vector.normalize(tzx, tzy, tzz)

    # Set null meridional/zonal vectors, as well as starting locations, to nan.
    if len(null_idx) > 0:
        tzx[null_idx], tzy[null_idx], tzz[null_idx] = np.nan, np.nan, np.nan
        tmx[null_idx], tmy[null_idx], tmz[null_idx] = np.nan, np.nan, np.nan
        ecef_x[null_idx], ecef_y[null_idx], ecef_z[null_idx] = (np.nan, np.nan,
                                                                np.nan)

    # Get apex location for root point.
    a_x, a_y, a_z, _, _, apex_root = location_info(ecef_x, ecef_y, ecef_z,
                                                   datetimes,
                                                   return_geodetic=True,
                                                   ecef_input=True)

    # Initialize loop variables
    loop_num = 0
    repeat_flag = True

    # Iteratively determine the vector basis directions.
    while repeat_flag:
        # Take a step along current zonal vector and calculate apex height
        # of new location. Depending upon user settings, this is either
        # a single step or steps along both positive and negative directions.

        # Positive zonal step.
        ecef_xz, ecef_yz, ecef_zz = (ecef_x + step_size * tzx,
                                     ecef_y + step_size * tzy,
                                     ecef_z + step_size * tzz)
        _, _, _, _, _, apex_z = location_info(ecef_xz, ecef_yz, ecef_zz,
                                              datetimes,
                                              return_geodetic=True,
                                              ecef_input=True)
        if centered_diff:
            # Negative step
            ecef_xz2, ecef_yz2, ecef_zz2 = (ecef_x - step_size * tzx,
                                            ecef_y - step_size * tzy,
                                            ecef_z - step_size * tzz)
            _, _, _, _, _, apex_z2 = location_info(ecef_xz2, ecef_yz2, ecef_zz2,
                                                   datetimes,
                                                   return_geodetic=True,
                                                   ecef_input=True)
            # Gradient in apex height
            diff_apex_z = apex_z - apex_z2
            diff_apex_z /= 2. * step_size
        else:
            # Gradient in apex height
            diff_apex_z = apex_z - apex_root
            diff_apex_z /= step_size

        # Meridional-ish direction, positive step.
        ecef_xm, ecef_ym, ecef_zm = (ecef_x + step_size * tmx,
                                     ecef_y + step_size * tmy,
                                     ecef_z + step_size * tmz)
        _, _, _, _, _, apex_m = location_info(ecef_xm, ecef_ym, ecef_zm,
                                              datetimes,
                                              return_geodetic=True,
                                              ecef_input=True)

        # Meridional gradient in apex height
        diff_apex_m = apex_m - apex_root
        diff_apex_m /= step_size

        # Use observed gradients to calculate a rotation angle that aligns
        # the zonal and meridional directions along desired directions.
        # zonal along no gradient, meridional along max.
        theta = np.arctan2(diff_apex_z, diff_apex_m)

        # See wikipedia quaternion spatial rotation page for equation below
        # https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
        # Precalculate some info.
        ct = np.cos(theta)
        st = np.sin(theta)

        # Zonal vector
        tzx2, tzy2, tzz2 = (tzx * ct - tmx * st, tzy * ct - tmy * st,
                            tzz * ct - tmz * st)

        # Meridional vector
        tmx2, tmy2, tmz2 = (tmx * ct + tzx * st, tmy * ct + tzy * st,
                            tmz * ct + tzz * st)

        # Track difference in vectors
        dx, dy, dz = (tzx2 - tzx)**2, (tzy2 - tzy)**2, (tzz2 - tzz)**2
        diff_z = np.sqrt(dx + dy + dz)
        dx, dy, dz = (tmx2 - tmx)**2, (tmy2 - tmy)**2, (tmz2 - tmz)**2
        diff_m = np.sqrt(dx + dy + dz)

        # Take biggest difference
        diff = np.nanmax([diff_z, diff_m])

        # Store info into calculation vectors to refine next loop
        tzx, tzy, tzz = tzx2, tzy2, tzz2
        tmx, tmy, tmz = tmx2, tmy2, tmz2

        # Check if we are done
        if np.isnan(diff):
            repeat_flag = False

        if (diff < tol) & (np.nanmax(np.abs(diff_apex_z))
                           < tol_zonal_apex) & (loop_num > 1):
            repeat_flag = False

        loop_num += 1
        if loop_num > max_loops:
            # Identify all locations with tolerance failure
            idx1, = np.where(diff_m >= tol)
            idx2, = np.where(diff_apex_z >= tol_zonal_apex)
            idx = np.hstack((idx1, idx2))

            # Assign nan to vectors
            tzx[idx], tzy[idx], tzz[idx] = np.nan, np.nan, np.nan
            tmx[idx], tmy[idx], tmz[idx] = np.nan, np.nan, np.nan

            estr = ''.join((str(len(idx)), ' locations did not converge.',
                            ' Setting to NaN.'))
            warnings.warn(estr)

    # Store temp arrays into output
    zx, zy, zz = tzx, tzy, tzz
    mx, my, mz = tmx, tmy, tmz

    # Set null meridional/zonal vectors to nan.
    if len(null_idx) > 0:
        zx[null_idx], zy[null_idx], zz[null_idx] = np.nan, np.nan, np.nan
        mx[null_idx], my[null_idx], mz[null_idx] = np.nan, np.nan, np.nan

    if full_output:

        # Calculate zonal gradient and field line expansion using latest vectors
        # Positive zonal step
        ecef_xz, ecef_yz, ecef_zz = (ecef_x + dstep_size * zx,
                                     ecef_y + dstep_size * zy,
                                     ecef_z + dstep_size * zz)
        # Corresponding apex location
        t_x1, t_y1, t_z1, _, _, apex_z = location_info(ecef_xz, ecef_yz, ecef_zz,
                                                       datetimes,
                                                       return_geodetic=True,
                                                       ecef_input=True)

        # Negative zonal step
        ecef_xz2, ecef_yz2, ecef_zz2 = (ecef_x - dstep_size * zx,
                                        ecef_y - dstep_size * zy,
                                        ecef_z - dstep_size * zz)
        # Corresponding apex location
        (t_x2, t_y2, t_z2,
         _, _, apex_z2) = location_info(ecef_xz2, ecef_yz2, ecef_zz2,
                                        datetimes, return_geodetic=True,
                                        ecef_input=True)

        # Basis vectors at apex location
        (azx, azy, azz, _, _, _,
         amx, amy, amz) = calculate_mag_drift_unit_vectors_ecef(a_x, a_y, a_z,
                                                                datetimes,
                                                                ecef_input=True,
                                                                step_size=step_size,
                                                                tol=tol,
                                                                tol_zonal_apex=tol_zonal_apex)

        # Get distance between apex points along apex zonal direction
        dist = (t_x1 - t_x2) * azx + (t_y1 - t_y2) * azy + (t_z1 - t_z2) * azz

        # Calculate gradient in field line separation distance
        grad_brb = dist / (2. * dstep_size)

        # Calculate gradient in apex height
        diff_apex_z = apex_z - apex_z2
        grad_zonal = diff_apex_z / (2. * dstep_size)

        # get magnitude of magnetic field at root apex location
        bax, bay, baz, bam = magnetic_vector(a_x, a_y, a_z, datetimes,
                                             normalize=True)

        # d vectors
        # Field-Aligned
        d_fa_x, d_fa_y, d_fa_z = bam / bm * bx, bam / bm * by, bam / bm * bz

        # Set null field-aligned vectors to nan.
        if len(null_idx) > 0:
            d_fa_x[null_idx], d_fa_y[null_idx], d_fa_z[null_idx] = (np.nan,
                                                                    np.nan,
                                                                    np.nan)

        # Zonal
        d_zon_x, d_zon_y, d_zon_z = grad_brb * zx, grad_brb * zy, grad_brb * zz

        # Calculate meridional that completes set
        d_mer_x, d_mer_y, d_mer_z = cross_product(d_zon_x, d_zon_y, d_zon_z,
                                                  d_fa_x, d_fa_y, d_fa_z)
        mag = d_mer_x**2 + d_mer_y**2 + d_mer_z**2
        d_mer_x, d_mer_y, d_mer_z = d_mer_x / mag, d_mer_y / mag, d_mer_z / mag

        # e vectors (Richmond nomenclature)
        # Zonal
        e_zon_x, e_zon_y, e_zon_z = cross_product(d_fa_x, d_fa_y, d_fa_z,
                                                  d_mer_x, d_mer_y, d_mer_z)
        # Field-Aligned
        e_fa_x, e_fa_y, e_fa_z = cross_product(d_mer_x, d_mer_y, d_mer_z,
                                               d_zon_x, d_zon_y, d_zon_z)
        # Meridional
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

            # Calculate meridional gradient using latest vectors
            # Positive step
            ecef_xm, ecef_ym, ecef_zm = (ecef_x + dstep_size*mx,
                                         ecef_y + dstep_size*my,
                                         ecef_z + dstep_size*mz)
            t_x1, t_y1, t_z1 = location_info(ecef_xm, ecef_ym, ecef_zm,
                                             datetimes, ecef_input=True)

            # Negative step
            ecef_xm2, ecef_ym2, ecef_zm2 = (ecef_x - dstep_size * mx,
                                            ecef_y - dstep_size * my,
                                            ecef_z - dstep_size * mz)
            t_x2, t_y2, t_z2 = location_info(ecef_xm2, ecef_ym2, ecef_zm2,
                                             datetimes, ecef_input=True)

            # Get distance between apex locations along meridional vector
            diff_apex_m = (t_x1 - t_x2) * amx + (t_y1 - t_y2) * amy + (t_z1 - t_z2) * amz

            # Apex height gradient
            grad_apex = diff_apex_m / (2. * dstep_size)

            # second path D, E vectors
            # d meridional vector via apex height gradient
            d_mer2_x, d_mer2_y, d_mer2_z = (grad_apex * mx, grad_apex * my,
                                            grad_apex * mz)

            # Zonal to complete set
            d_zon2_x, d_zon2_y, d_zon2_z = cross_product(d_fa_x, d_fa_y, d_fa_z,
                                                         d_mer2_x, d_mer2_y,
                                                         d_mer2_z)
            mag = d_zon2_x**2 + d_zon2_y**2 + d_zon2_z**2
            d_zon2_x, d_zon2_y, d_zon2_z = (d_zon2_x / mag, d_zon2_y / mag,
                                            d_zon2_z / mag)

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
                               step_size=25., scalar=None):
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
    step_size : float
        Distance taken for each step (km) (default=25.)
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

    if scalar is not None:
        estr = 'Scalar has been deprecated'
        warnings.warn(estr, DeprecationWarning)

    if direction == 'meridional':
        centered_diff = True
    else:
        centered_diff = False

    for i in np.arange(num_steps):
        # x, y, z in ECEF
        # get unit vector directions
        (zvx, zvy, zvz,
         bx, by, bz,
         mx, my, mz) = calculate_mag_drift_unit_vectors_ecef(x, y, z, date,
                                                             step_size=step_size,
                                                             ecef_input=True,
                                                             centered_diff=centered_diff)
        # Pull out the direction we need
        if direction == 'meridional':
            ux, uy, uz = mx, my, mz
        elif direction == 'zonal':
            ux, uy, uz = zvx, zvy, zvz
        elif direction == 'aligned':
            ux, uy, uz = bx, by, bz

        # Take steps along direction
        x = x + step_size * ux
        y = y + step_size * uy
        z = z + step_size * uz

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
        ecef_xs, ecef_ys, ecef_zs = trans.geodetic_to_ecef(glats, glons, alts)

    north_ftpnt = np.empty((len(ecef_xs), 3), dtype=np.float64)
    south_ftpnt = np.empty((len(ecef_xs), 3), dtype=np.float64)

    # Get dates in relevant format
    ddates = OMMBV.utils.datetimes_to_doubles(dates)

    root = np.array([0., 0., 0.], dtype=np.float64)
    i = 0
    steps = np.arange(num_steps + 1)
    for ecef_x, ecef_y, ecef_z, date in zip(ecef_xs, ecef_ys, ecef_zs, ddates):

        root[:] = (ecef_x, ecef_y, ecef_z)
        trace_north = field_line_trace(root, date, 1., 120.,
                                       steps=steps,
                                       step_size=step_size,
                                       max_steps=num_steps)
        # Southern tracing
        trace_south = field_line_trace(root, date, -1., 120.,
                                       steps=steps,
                                       step_size=step_size,
                                       max_steps=num_steps)
        # Footpoint location
        north_ftpnt[i, :] = trace_north[-1, :]
        south_ftpnt[i, :] = trace_south[-1, :]
        i += 1

    if return_geodetic:
        (north_ftpnt[:, 0],
         north_ftpnt[:, 1],
         north_ftpnt[:, 2]) = trans.ecef_to_geodetic(north_ftpnt[:, 0],
                                                     north_ftpnt[:, 1],
                                                     north_ftpnt[:, 2])
        (south_ftpnt[:, 0],
         south_ftpnt[:, 1],
         south_ftpnt[:, 2]) = trans.ecef_to_geodetic(south_ftpnt[:, 0],
                                                     south_ftpnt[:, 1],
                                                     south_ftpnt[:, 2])

    return north_ftpnt, south_ftpnt


def apex_edge_lengths_via_footpoint(*args, **kwargs):
    estr = 'This method now called `apex_distance_after_footpoint_step`.'
    warnings.warn(estr, DeprecationWarning)
    apex_distance_after_footpoint_step(*args, **kwargs)
    return


def apex_distance_after_footpoint_step(glats, glons, alts, dates, direction,
                                       vector_direction, step_size=100.,
                                       max_steps=1000, steps=None,
                                       edge_length=25., edge_steps=5,
                                       ecef_input=False):
    """
    Calculate distance between apex locations after `vector_direction` step.

    Using the input location, the footpoint location is calculated.
    From here, a step along both the positive and negative
    `vector_direction` is taken, and the apex locations for those points are
    calculated. The difference in position between these apex locations is the
    total centered distance between magnetic field lines at the magnetic apex
    when starting from the footpoints with a field line half distance of
    `edge_length`.

    Parameters
    ----------
    glats : list-like of floats
        Geodetic (WGS84) latitude in degrees.
    glons : list-like of floats
        Geodetic (WGS84) longitude in degrees.
    alts : list-like of floats
        Geodetic (WGS84) altitude, height above surface in km.
    dates : list-like of datetimes
        Date and time for determination of scalars.
    direction : str
        'north' or 'south' for tracing towards northern (field-aligned) or
        southern (anti field-aligned) footpoint locations.
    vector_direction : str
        'meridional' or 'zonal' magnetic unit vector directions.
    step_size : float
        Step size (km) used for field line integration. (default=100.)
    max_steps : int
        Number of steps taken for field line integration. (default=1000)
    steps : np.array or NoneType
        Integration steps array passed to full_field_line.
        (default=np.arange(max_steps+1) if steps is None)
    edge_length : float (km)
        Half of total edge length (step) taken at footpoint location.
        edge_length step in both positive and negative directions.
    edge_steps : int
        Number of steps taken from footpoint towards new field line
        in a given direction (positive/negative) along unit vector.
        (default=5)
    ecef_input : bool
        If True, latitude, longitude, and altitude are treated as
        ECEF positions (km). (default=False)

    Returns
    -------
    np.array,
        A closed loop field line path through input location and footpoint in
        northern/southern hemisphere and back is taken. The return edge length
        through input location is provided.

    """

    if steps is None:
        steps = np.arange(max_steps + 1)

    # Use spacecraft location to get ECEF
    if ecef_input:
        ecef_xs, ecef_ys, ecef_zs = glats, glons, alts
    else:
        ecef_xs, ecef_ys, ecef_zs = trans.geodetic_to_ecef(glats, glons, alts)

    if direction == 'north':
        ftpnts, _ = footpoint_location_info(ecef_xs, ecef_ys, ecef_zs, dates,
                                            ecef_input=True,
                                            step_size=step_size,
                                            max_steps=max_steps,
                                            steps=steps)
    elif direction == 'south':
        _, ftpnts = footpoint_location_info(ecef_xs, ecef_ys, ecef_zs, dates,
                                            ecef_input=True,
                                            step_size=step_size,
                                            max_steps=max_steps,
                                            steps=steps)

    # Take step from footpoint along + vector direction
    plus_x, plus_y, plus_z = step_along_mag_unit_vector(ftpnts[:, 0], ftpnts[:, 1], ftpnts[:, 2],
                                                        dates,
                                                        direction=vector_direction,
                                                        num_steps=edge_steps,
                                                        step_size=edge_length/edge_steps)

    plus_apex_x, plus_apex_y, plus_apex_z = apex_location_info(plus_x, plus_y,
                                                               plus_z, dates,
                                                               ecef_input=True,
                                                               step_size=step_size,
                                                               steps=steps)

    # Take half step from first footpoint along - vector direction
    minus_x, minus_y, minus_z = step_along_mag_unit_vector(ftpnts[:, 0], ftpnts[:, 1], ftpnts[:, 2],
                                                           dates,
                                                           direction=vector_direction,
                                                           scalar=-1,
                                                           num_steps=edge_steps,
                                                           step_size=edge_length/edge_steps)
    (minus_apex_x,
     minus_apex_y,
     minus_apex_z) = apex_location_info(minus_x, minus_y, minus_z, dates,
                                        ecef_input=True, step_size=step_size,
                                        steps=steps)

    # Take difference in apex locations
    apex_edge_length = np.sqrt((plus_apex_x - minus_apex_x)**2 +
                               (plus_apex_y - minus_apex_y)**2 +
                               (plus_apex_z - minus_apex_z)**2)

    return apex_edge_length


def apex_distance_after_local_step(glats, glons, alts, dates,
                                   vector_direction,
                                   edge_length=25.,
                                   edge_steps=5,
                                   ecef_input=False,
                                   return_geodetic=False,
                                   location_info=apex_location_info):
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
        ecef_xs, ecef_ys, ecef_zs = trans.geodetic_to_ecef(glats, glons, alts)

    # take step from s/c along + vector direction
    # then get the apex location
    plus_x, plus_y, plus_z = step_along_mag_unit_vector(ecef_xs, ecef_ys,
                                                        ecef_zs, dates,
                                                        direction=vector_direction,
                                                        num_steps=edge_steps,
                                                        step_size=edge_length/edge_steps)

    # take half step from s/c along - vector direction
    # then get the apex location
    minus_x, minus_y, minus_z = step_along_mag_unit_vector(ecef_xs, ecef_ys,
                                                           ecef_zs, dates,
                                                           direction=vector_direction,
                                                           scalar=-1,
                                                           num_steps=edge_steps,
                                                           step_size=edge_length/edge_steps)

    # get apex locations
    if return_geodetic:
        plus_apex_x, plus_apex_y, plus_apex_z, _, _, plus_h = \
            location_info(plus_x, plus_y, plus_z, dates,
                               ecef_input=True,
                               return_geodetic=True)

        minus_apex_x, minus_apex_y, minus_apex_z, _, _, minus_h = \
            location_info(minus_x, minus_y, minus_z, dates,
                               ecef_input=True,
                               return_geodetic=True)
    else:
        plus_apex_x, plus_apex_y, plus_apex_z = \
            location_info(plus_x, plus_y, plus_z, dates,
                               ecef_input=True)

        minus_apex_x, minus_apex_y, minus_apex_z = \
            location_info(minus_x, minus_y, minus_z, dates,
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
    ecef_xs, ecef_ys, ecef_zs = trans.geodetic_to_ecef(glats, glons, alts)

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


def heritage_scalars_for_mapping_ion_drifts(glats, glons, alts, dates,
                                            step_size=None, max_steps=None,
                                            e_field_scaling_only=False,
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
    ecef_xs, ecef_ys, ecef_zs = trans.geodetic_to_ecef(glats, glons, alts)

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


def geocentric_to_ecef(latitude, longitude, altitude):
    """Convert geocentric coordinates into ECEF

    .. deprecated:: 0.5.6
       Function moved to `OMMBV.trans.geocentric_to_ecef`, this wrapper will
       be removed in 0.6

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
    x, y, z : np.array
        x, y, z ECEF locations in km

    """

    warnings.warn("".join(["Function moved to `OMMBV.trans`, deprecated ",
                           "wrapper will be removed in OMMBV 0.6+"]),
                  DeprecationWarning, stacklevel=2)

    return trans.geocentric_to_ecef(latitude, longitude, altitude)


def ecef_to_geocentric(x, y, z, ref_height=trans.earth_geo_radius):
    """Convert ECEF into geocentric coordinates

    .. deprecated:: 0.5.6
       Function moved to `OMMBV.trans.ecef_to_geocentric`, this wrapper will
       be removed in 0.6

    Parameters
    ----------
    x : float or array_like
        ECEF-X in km
    y : float or array_like
        ECEF-Y in km
    z : float or array_like
        ECEF-Z in km
    ref_height : float or array_like
        Reference radius used for calculating height in km.
        (default=trans.earth_geo_radius)

    Returns
    -------
    latitude, longitude, altitude : np.array
        Locations in latitude (degrees), longitude (degrees), and
        altitude above `reference_height` in km.

    """
    warnings.warn("".join(["Function moved to `OMMBV.trans`, deprecated ",
                           "wrapper will be removed in OMMBV 0.6+"]),
                  DeprecationWarning, stacklevel=2)

    return trans.ecef_to_geocentric(x, y, z, ref_height=ref_height)


def geodetic_to_ecef(latitude, longitude, altitude):
    """Convert WGS84 geodetic coordinates into ECEF

    .. deprecated:: 0.5.6
       Function moved to `OMMBV.trans.geodetic_to_ecef`, this wrapper will
       be removed in 0.6

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
    x, y, z : np.array
        x, y, z ECEF locations in km

    """

    warnings.warn("".join(["Function moved to `OMMBV.trans`, deprecated ",
                           "wrapper will be removed in OMMBV 0.6+"]),
                  DeprecationWarning, stacklevel=2)

    return trans.geodetic_to_ecef(latitude, longitude, altitude)


def python_ecef_to_geodetic(x, y, z, method='closed'):
    """Convert ECEF into Geodetic WGS84 coordinates using Python code.

    .. deprecated:: 0.5.6
       Function moved to `OMMBV.trans.geodetic_to_ecef`, this wrapper will
       be removed in 0.6

    Parameters
    ----------
    x : float or array_like
        ECEF-X in km
    y : float or array_like
        ECEF-Y in km
    z : float or array_like
        ECEF-Z in km
    method : str
        Supports 'iterative' or 'closed' to select method of conversion.
        'closed' for mathematical solution (page 96 section 2.2.1,
        http://www.epsg.org/Portals/0/373-07-2.pdf) or 'iterative'
        (http://www.oc.nps.edu/oc2902w/coord/coordcvt.pdf). (default = 'closed')

    Returns
    -------
    latitude, longitude, altitude : np.array
        Locations in latitude (degrees), longitude (degrees), and
        altitude above WGS84 (km)

    """

    warnings.warn("".join(["Function moved to `OMMBV.trans`, deprecated ",
                           "wrapper will be removed in OMMBV 0.6+"]),
                  DeprecationWarning, stacklevel=2)

    if method is None:
        warnings.warn("".join(["`method` must be a string value in ",
                               "0.6.0+. Setting to function default."]),
                      DeprecationWarning, stacklevel=2)
        method = 'closed'

    return trans.python_ecef_to_geodetic(x, y, z, method=method)
