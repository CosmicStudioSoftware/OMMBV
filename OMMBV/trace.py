"""Support field-line tracing."""

import datetime as dt
import numpy as np
import scipy
import scipy.integrate
import warnings


try:
    def _stub_fcn():
        """Magnetic field/step function stub."""
        return

    _stub_fcn()
    # Import reference IGRF fortran code, if possible.
    # RTD doesn't support Fortran.
    from OMMBV import igrf
    from OMMBV import sources
    ommbv_mag_fcn = igrf.igrf13syn
    ommbv_step_fcn = sources.igrf_step
except Exception:
    ommbv_mag_fcn, ommbv_step_fcn = _stub_fcn, _stub_fcn

from OMMBV import trans
import OMMBV.utils
from OMMBV import vector


def field_line_trace(init, date, direction, height, steps=None,
                     max_steps=1E4, step_size=10., recursive_loop_count=None,
                     recurse=True, min_check_flag=False,
                     step_fcn=ommbv_step_fcn):
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
    step_fcn : function
        Function used by `scipy.integrate.odeint` to step along
        the magnetic field. (default=`OMMBV.sources.igrf_step`)

    Returns
    -------
    numpy array
        2D array. [0,:] has the x,y,z location for initial point
        [:,0] is the x positions over the integration.
        Positions are reported in ECEF (km).

    Note
    ----
    After 500 recursive iterations this method will increase the step size by
    3% every subsequent iteration. Fails at 900 iterations.

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
    trace_north, messg = scipy.integrate.odeint(step_fcn,
                                                init.copy(),
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
    if recurse & (z > check_height * 1.000001):
        if recursive_loop_count > 500:
            loop_step *= 1.03
        if recursive_loop_count < 900:
            # When we have not reached the reference height, call
            # `field_line_trace` again by taking check value as init.
            # Recursive call.
            recursive_loop_count = recursive_loop_count + 1
            rlc = recursive_loop_count
            trace_north1 = field_line_trace(check, date, direction, height,
                                            step_size=loop_step,
                                            max_steps=max_steps,
                                            recursive_loop_count=rlc,
                                            steps=steps,
                                            step_fcn=step_fcn)
        else:
            estr = "After 900 iterations couldn't reach target altitude"
            warnings.warn(estr)
            return np.full((1, 3), np.nan)

        # Append new trace data to existing trace data
        # this return is taken as part of recursive loop
        if np.isnan(trace_north1[-1, 0]):
            return trace_north1
        else:
            return np.vstack((trace_north, trace_north1))

    else:

        # Optionally filter points to terminate at point closest to target
        # height. While this check is done internally within Fortran integrand,
        # if that steps out early, the output we receive could be problematic.
        # Steps below provide an extra layer of security that output has some
        # semblance to expectations.
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
        are taken as determined by `scipy.integrate.odeint`.
    steps : array-like of ints or floats
        Number of steps along field line when field line trace positions should
        be reported. By default, each step is reported, plus origin;
            steps=np.arange(max_steps+1).
        Two traces are made, one north, the other south, thus the output array
        could have double max_steps, or more via recursion.
    **kwargs : Additional keywords
        Passed to `trace.field_line_trace`.

    Returns
    -------
    numpy array
        2D array. [0,:] has the x,y,z location for southern footpoint
        [:,0] is the x positions over the integration.
        Positions are reported in ECEF (km).


    """

    if steps is None:
        steps = np.arange(max_steps + 1)
    elif len(steps) != max_steps + 1:
        raise ValueError('Length of steps must be `max_steps`+1.')

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

    # Order of field points is along the field line, south to north.
    # Don't want to include the initial point twice.
    if not np.isnan(trace_north[-1, 0]) and (not np.isnan(trace_south[-1, 0])):
        trace = np.vstack((trace_south[::-1][:-1, :], trace_north))
    else:
        trace = np.full((1, 3), np.nan)
    return trace


def apex_location_info(glats, glons, alts, dates, step_size=100.,
                       fine_step_size=1.E-5, fine_max_steps=5,
                       return_geodetic=False, ecef_input=False,
                       validate_input=False, max_steps=None,
                       **kwargs):
    """Determine apex location for the field line passing through input point.

    Employs a two stage method. A broad step (`step_size`) field line trace
    spanning Northern/Southern footpoints is used to find the location with
    the largest geodetic (WGS84) height. A binary search higher resolution
    trace is then used to get a better fix on this location. Each loop,
    the `step_size` halved. Greatest geodetic height is once
    again selected once the `step_size` is below `fine_step_size`.

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
    validate_input : bool
        If True, will check input locations to ensure they are above
        the surface of the Earth. (default=False)
    **kwargs : Additional keywords
        Passed to `trace.full_field_line`.

    Returns
    -------
    (float, float, float, float, float, float)
        ECEF X (km), ECEF Y (km), ECEF Z (km),
    if return_geodetic, also includes:
        Geodetic Latitude (degrees),
        Geodetic Longitude (degrees),
        Geodetic Altitude (km)

    Warnings
    --------
    The altitude of the input point must be at least 1 km above the Earth.
    Set `validate_input` to True to test inputs and set locations that
    don't satisfy requirements to np.nan.

    Does not work at the geographic pole due to the use of the
    `ecef_to_geodetic` transformation.

    """
    # Use input location and convert to ECEF
    if ecef_input:
        ecef_xs, ecef_ys, ecef_zs = glats, glons, alts
    else:
        ecef_xs, ecef_ys, ecef_zs = trans.geodetic_to_ecef(glats, glons, alts)

    if validate_input:
        # Filter starting locations to those at least 1 km above the Earth.
        tlats, tlons, talts = trans.ecef_to_geodetic(ecef_xs, ecef_ys, ecef_zs)
        idx = np.where(talts < 1.)
        ecef_xs[idx], ecef_ys[idx], ecef_zs[idx] = np.nan, np.nan, np.nan

    # Prepare parameters for field line trace
    if max_steps is None:
        # Auto-generate `max_steps` number via guess. Ensure at least 100.
        max_steps = 10000. // step_size
        max_steps = max_steps if max_steps >= 100 else 100

    # Generate memory array for integration.
    apex_coarse_steps = np.arange(max_steps + 1)

    # High resolution array for integration.
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
                                max_steps=max_steps,
                                **kwargs)

        # Ensure trace is valid. trace is [[np.nan, np.nan, np.nan]] when
        # `full_field_line` fails.
        if np.any(np.isnan(trace[0, :])):
            apex_out_x[i], apex_out_y[i], apex_out_z[i] = np.nan, np.nan, np.nan
            i += 1
            continue

        # Convert all locations to geodetic coordinates
        tlat, tlon, talt = trans.ecef_to_geodetic(trace[:, 0], trace[:, 1],
                                                  trace[:, 2])

        # Determine location that is highest with respect to the geodetic Earth
        max_idx = np.argmax(talt)

        # Repeat using a high resolution trace one big step size each
        # direction around identified max. It is possible internal functions
        # had to increase step size. Get step estimate from trace. Note
        # that the apex is away from the footpoints, hence it is not at the
        # start of the trace field line array, so `max_idx` >= 1.
        nstep = np.sqrt((trace[max_idx, 0] - trace[max_idx - 1, 0])**2
                        + (trace[max_idx, 1] - trace[max_idx - 1, 1])**2
                        + (trace[max_idx, 2] - trace[max_idx - 1, 2])**2)

        # Iteratively search for max using a smaller step size each round.
        while nstep > fine_step_size:
            nstep /= 2.

            # Setting recurse False ensures only max_steps are taken.
            trace = full_field_line(trace[max_idx, :], date, 0.,
                                    steps=apex_fine_steps,
                                    step_size=nstep,
                                    max_steps=fine_max_steps,
                                    recurse=False,
                                    **kwargs)

            # Convert all locations to geodetic coordinates.
            tlat, tlon, talt = trans.ecef_to_geodetic(trace[:, 0], trace[:, 1],
                                                      trace[:, 2])

            # Determine location that is highest with respect to geodetic Earth.
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


def footpoint_location_info(glats, glons, alts, dates, step_size=100.,
                            num_steps=1000, return_geodetic=False,
                            ecef_input=False, height=120., **kwargs):
    """Return ECEF location of footpoints in Northern/Southern hemisphere.

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
        Number of steps passed along to `field_line_trace` as max_steps.
    return_geodetic : bool
        If True, footpoint locations returned as lat, long, alt.
    ecef_input : bool
        If True, glats, glons, and alts are treated as x, y, z (ECEF).
    height : float
        Geodetic altitude in km to consider footpoint. (default=120.)
    **kwargs : Additional keywords
        Passed to `trace.field_line_trace`.

    Returns
    -------
    array(len(glats), 3), array(len(glats), 3)
        Northern and Southern ECEF X,Y,Z locations

    """

    if ecef_input:
        ecef_xs, ecef_ys, ecef_zs = glats, glons, alts
    else:
        ecef_xs, ecef_ys, ecef_zs = trans.geodetic_to_ecef(glats, glons, alts)

    # Init memory
    north_ftpnt = np.empty((len(ecef_xs), 3), dtype=np.float64)
    south_ftpnt = np.empty((len(ecef_xs), 3), dtype=np.float64)
    root = np.array([0., 0., 0.], dtype=np.float64)
    steps = np.arange(num_steps + 1)

    # Get dates in relevant format
    ddates = OMMBV.utils.datetimes_to_doubles(dates)

    i = 0
    for ecef_x, ecef_y, ecef_z, date in zip(ecef_xs, ecef_ys, ecef_zs, ddates):
        root[:] = (ecef_x, ecef_y, ecef_z)
        # Trace north
        trace_north = field_line_trace(root, date, 1., height,
                                       steps=steps,
                                       step_size=step_size,
                                       max_steps=num_steps,
                                       **kwargs)
        # Southern tracing
        trace_south = field_line_trace(root, date, -1., height,
                                       steps=steps,
                                       step_size=step_size,
                                       max_steps=num_steps,
                                       **kwargs)
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


def magnetic_vector(x, y, z, dates, normalize=False, mag_fcn=ommbv_mag_fcn):
    """Use IGRF to calculate geomagnetic field.

    Parameters
    ----------
    x : array-like of floats
        Position in ECEF (km), X
    y : array-like of floats
        Position in ECEF (km), Y
    z : array-like of floats
        Position in ECEF (km), Z
    dates : array-like of datetimes
        Datetimes to calculate magnetic field
    normalize : bool
        If True, return unit vector. (default=False)
    mag_fcn : function
        Function that returns magnetic field from a model.
        The call signature must match that of `igrf.igrf13syn`.
        (default=OMMBV.igrf.igrf13syn)

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
        tbn, tbe, tbd, tbmag = mag_fcn(0, date, 2, alt, colat, elong)

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
    bx, by, bz = vector.enu_to_ecef(be, bn, -bd, latitudes, longitudes)

    if normalize:
        bx /= bm
        by /= bm
        bz /= bm

    return bx, by, bz, bm
