"""Heritage functions for OMMBV. Will be removed in v2.0.0."""

import numpy as np
import warnings

from OMMBV import step_along_mag_unit_vector
from OMMBV.trace import apex_location_info
from OMMBV.trace import footpoint_location_info
from OMMBV.trace import full_field_line
from OMMBV.trace import magnetic_vector
import OMMBV.trans as trans
import OMMBV.utils
from OMMBV import vector


def calculate_integrated_mag_drift_unit_vectors_ecef(latitude, longitude,
                                                     altitude, datetimes,
                                                     steps=None,
                                                     max_steps=1000,
                                                     step_size=100.,
                                                     ref_height=120.,
                                                     filter_zonal=True):
    """Calculate field line integrated geomagnetic basis vectors.

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

    # Calculate satellite position in ECEF coordinates
    ecef_x, ecef_y, ecef_z = trans.geodetic_to_ecef(latitude, longitude,
                                                    altitude)
    # Also get position in geocentric coordinates
    geo_lat, geo_long, geo_alt = trans.ecef_to_geocentric(ecef_x, ecef_y,
                                                          ecef_z,
                                                          ref_height=0.)
    # geo_lat, geo_long, geo_alt = trans.ecef_to_geodetic(ecef_x, ecef_y,
    # ecef_z)

    # Filter longitudes (could use pysat's function here)
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
                                                altitude,
                                                np.deg2rad(90. - latitude),
                                                np.deg2rad(longitude),
                                                datetimes):
        init = np.array([x, y, z], dtype=np.float64)
        trace = full_field_line(init, time, ref_height, step_size=step_size,
                                max_steps=max_steps,
                                steps=steps)
        # Store final location, full trace goes south to north
        trace_north = trace[-1, :]
        trace_south = trace[0, :]

        # Get IGRF field components. tbn, tbe, tbd, tbmag are in nT.
        # Geodetic input.
        tbx, tby, tbz, tbmag = OMMBV.trace.magnetic_vector([x], [y], [z],
                                                           [time])

        lat = -np.rad2deg(colat + np.pi / 2.)
        lon = np.rad2deg(elong)
        tbe, tbn, tbd = OMMBV.vector.ecef_to_enu(tbx, tby, tbz, lat, lon)
        tbn, tbe, tbd = tbn[0], tbe[0], -tbd[0]

        # Collect outputs
        south_x.append(trace_south[0])
        south_y.append(trace_south[1])
        south_z.append(trace_south[2])
        north_x.append(trace_north[0])
        north_y.append(trace_north[1])
        north_z.append(trace_north[2])

        bn.append(tbn)
        be.append(tbe)
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

    # Calculate vector from satellite to northern/southern footpoints
    north_x = north_x - ecef_x
    north_y = north_y - ecef_y
    north_z = north_z - ecef_z
    north_x, north_y, north_z = vector.normalize(north_x, north_y, north_z)
    south_x = south_x - ecef_x
    south_y = south_y - ecef_y
    south_z = south_z - ecef_z
    south_x, south_y, south_z = vector.normalize(south_x, south_y, south_z)

    # Calculate magnetic unit vector
    bx, by, bz = vector.enu_to_ecef(be, bn, -bd, geo_lat, geo_long)
    bx, by, bz = vector.normalize(bx, by, bz)

    # Take cross product of southward/northward vectors to get the zonal vector
    zvx_foot, zvy_foot, zvz_foot = OMMBV.vector.cross_product(south_x, south_y,
                                                              south_z, north_x,
                                                              north_y, north_z)
    # Normalize the vectors
    norm_foot = np.sqrt(zvx_foot**2 + zvy_foot**2 + zvz_foot**2)

    # Calculate zonal vector
    zvx = zvx_foot / norm_foot
    zvy = zvy_foot / norm_foot
    zvz = zvz_foot / norm_foot

    if filter_zonal:
        # Remove any field aligned component to the zonal vector
        dot_fa = zvx * bx + zvy * by + zvz * bz
        zvx -= dot_fa * bx
        zvy -= dot_fa * by
        zvz -= dot_fa * bz
        zvx, zvy, zvz = vector.normalize(zvx, zvy, zvz)

    # Compute meridional vector
    # Cross product of zonal and magnetic unit vector
    mx, my, mz = vector.cross_product(zvx, zvy, zvz, bx, by, bz)

    # Add unit vectors for magnetic drifts in ecef coordinates
    return zvx, zvy, zvz, bx, by, bz, mx, my, mz


def apex_edge_lengths_via_footpoint(*args, **kwargs):
    """Calculate distance between apex locations.

    .. deprecated:: 1.0.0
       Function moved to `apex_distance_after_footpoint_step`,
       this wrapper will be removed after v1.0.0.

    """
    estr = ''.join(['This method now called `apex_distance_after_',
                    'footpoint_step`. Wrapper will be removed after OMMBV ',
                    'v1.0.0.'])
    warnings.warn(estr, DeprecationWarning, stacklevel=2)
    apex_distance_after_footpoint_step(*args, **kwargs)
    return


def apex_distance_after_footpoint_step(glats, glons, alts, dates, direction,
                                       vector_direction, step_size=100.,
                                       max_steps=1000, steps=None,
                                       edge_length=25., edge_steps=5,
                                       ecef_input=False):
    """Calculate distance between apex locations after `vector_direction` step.

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
        (default=25.)
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
                                            step_size=step_size)
    elif direction == 'south':
        _, ftpnts = footpoint_location_info(ecef_xs, ecef_ys, ecef_zs, dates,
                                            ecef_input=True,
                                            step_size=step_size)

    # Take step from footpoint along + vector direction
    (plus_x,
     plus_y,
     plus_z) = step_along_mag_unit_vector(ftpnts[:, 0], ftpnts[:, 1],
                                          ftpnts[:, 2], dates,
                                          direction=vector_direction,
                                          num_steps=edge_steps,
                                          step_size=edge_length / edge_steps)

    (plus_apex_x,
     plus_apex_y,
     plus_apex_z) = apex_location_info(plus_x, plus_y, plus_z, dates,
                                       ecef_input=True, step_size=step_size)

    # Take half step from first footpoint along - vector direction
    (minus_x,
     minus_y,
     minus_z) = step_along_mag_unit_vector(ftpnts[:, 0], ftpnts[:, 1],
                                           ftpnts[:, 2], dates,
                                           direction=vector_direction,
                                           scalar=-1, num_steps=edge_steps,
                                           step_size=edge_length / edge_steps)
    (minus_apex_x,
     minus_apex_y,
     minus_apex_z) = apex_location_info(minus_x, minus_y, minus_z, dates,
                                        ecef_input=True, step_size=step_size)

    # Take difference in apex locations
    apex_edge_length = np.sqrt((plus_apex_x - minus_apex_x)**2
                               + (plus_apex_y - minus_apex_y)**2
                               + (plus_apex_z - minus_apex_z)**2)

    return apex_edge_length


def apex_distance_after_local_step(glats, glons, alts, dates,
                                   vector_direction,
                                   edge_length=25.,
                                   edge_steps=5,
                                   ecef_input=False,
                                   return_geodetic=False,
                                   location_info=apex_location_info):
    """Calculate the distance between apex locations after local step.

    Using the input location, the apex location is calculated. Also from the
    input location, a step along both the positive and negative
    `vector_direction` is taken, and the apex locations for those points are
    calculated. The difference in position between these apex locations is
    the total centered distance between magnetic field lines at the magnetic
    apex when starting locally with a field line half distance of `edge_length`.

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

    if ecef_input:
        ecef_xs, ecef_ys, ecef_zs = glats, glons, alts
    else:
        ecef_xs, ecef_ys, ecef_zs = trans.geodetic_to_ecef(glats, glons, alts)

    # Take step along + vector direction, then get the apex location.
    (plus_x,
     plus_y,
     plus_z) = step_along_mag_unit_vector(ecef_xs, ecef_ys, ecef_zs, dates,
                                          direction=vector_direction,
                                          num_steps=edge_steps,
                                          step_size=edge_length / edge_steps)

    # Take step along + vector direction, then get the apex location.
    (minus_x,
     minus_y,
     minus_z) = step_along_mag_unit_vector(ecef_xs, ecef_ys, ecef_zs, dates,
                                           direction=vector_direction,
                                           scalar=-1, num_steps=edge_steps,
                                           step_size=edge_length / edge_steps)

    # Get apex locations
    if return_geodetic:
        (plus_apex_x, plus_apex_y, plus_apex_z,
         _, _, plus_h) = location_info(plus_x, plus_y, plus_z, dates,
                                       ecef_input=True, return_geodetic=True)

        (minus_apex_x, minus_apex_y, minus_apex_z,
         _, _, minus_h) = location_info(minus_x, minus_y, minus_z, dates,
                                        ecef_input=True, return_geodetic=True)
    else:
        (plus_apex_x,
         plus_apex_y,
         plus_apex_z) = location_info(plus_x, plus_y, plus_z, dates,
                                      ecef_input=True)

        (minus_apex_x,
         minus_apex_y,
         minus_apex_z) = location_info(minus_x, minus_y, minus_z, dates,
                                       ecef_input=True)

    # take difference in apex locations
    apex_edge_length = np.sqrt((plus_apex_x - minus_apex_x)**2
                               + (plus_apex_y - minus_apex_y)**2
                               + (plus_apex_z - minus_apex_z)**2)

    if return_geodetic:
        return apex_edge_length, plus_h - minus_h
    else:
        return apex_edge_length


def heritage_scalars_for_mapping_ion_drifts(glats, glons, alts, dates,
                                            step_size=None, max_steps=None,
                                            e_field_scaling_only=False,
                                            edge_length=25., edge_steps=1,
                                            **kwargs):
    """
    Heritage technique for mapping ion drifts and electric fields.

    .. deprecated:: 0.6.0
       Use `OMMBV.scalars_for_mapping_ion_drifts` instead. Will be removed
       after 1.0.0.

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
        scalar applies to zonal ion motions (meridional E field
        assuming ExB ion motion)

    """

    warnings.warn("Use `OMMBV.scalars_for_mapping_ion_drifts` instead.",
                  DeprecationWarning, stacklevel=2)

    if step_size is None:
        step_size = 100.
    if max_steps is None:
        max_steps = 1000
    steps = np.arange(max_steps + 1)

    ecef_xs, ecef_ys, ecef_zs = trans.geodetic_to_ecef(glats, glons, alts)

    # double edge length, used later
    double_edge = 2. * edge_length

    # Prepare output
    out = {}

    # Meridional e-field scalar map, can also be zonal ion drift scalar map
    adafs = apex_distance_after_footpoint_step
    north_zon_drifts_scalar = adafs(ecef_xs, ecef_ys, ecef_zs, dates, 'north',
                                    'meridional', step_size=step_size,
                                    max_steps=max_steps,
                                    edge_length=edge_length,
                                    edge_steps=edge_steps, steps=steps,
                                    ecef_input=True, **kwargs)

    north_mer_drifts_scalar = adafs(ecef_xs, ecef_ys, ecef_zs, dates, 'north',
                                    'zonal', step_size=step_size,
                                    max_steps=max_steps,
                                    edge_length=edge_length,
                                    edge_steps=edge_steps, steps=steps,
                                    ecef_input=True, **kwargs)

    south_zon_drifts_scalar = adafs(ecef_xs, ecef_ys, ecef_zs, dates, 'south',
                                    'meridional', step_size=step_size,
                                    max_steps=max_steps,
                                    edge_length=edge_length,
                                    edge_steps=edge_steps, steps=steps,
                                    ecef_input=True, **kwargs)

    south_mer_drifts_scalar = adafs(ecef_xs, ecef_ys, ecef_zs, dates, 'south',
                                    'zonal', step_size=step_size,
                                    max_steps=max_steps,
                                    edge_length=edge_length,
                                    edge_steps=edge_steps, steps=steps,
                                    ecef_input=True, **kwargs)

    adals = apex_distance_after_local_step
    eq_zon_drifts_scalar = adals(ecef_xs, ecef_ys, ecef_zs, dates,
                                 'meridional', edge_length=edge_length,
                                 edge_steps=edge_steps, ecef_input=True)
    eq_mer_drifts_scalar = adals(ecef_xs, ecef_ys, ecef_zs, dates, 'zonal',
                                 edge_length=edge_length, edge_steps=edge_steps,
                                 ecef_input=True)

    # Ratio of apex height difference to step_size across footpoints
    # scales from equator to footpoint
    north_zon_drifts_scalar = north_zon_drifts_scalar / double_edge
    south_zon_drifts_scalar = south_zon_drifts_scalar / double_edge
    north_mer_drifts_scalar = north_mer_drifts_scalar / double_edge
    south_mer_drifts_scalar = south_mer_drifts_scalar / double_edge

    # Equatorial
    # Scale from s/c to equator
    eq_zon_drifts_scalar = double_edge / eq_zon_drifts_scalar
    eq_mer_drifts_scalar = double_edge / eq_mer_drifts_scalar

    # change scaling from equator to footpoint, to s/c to footpoint
    # via s/c to equator
    north_zon_drifts_scalar *= eq_zon_drifts_scalar
    south_zon_drifts_scalar *= eq_zon_drifts_scalar
    north_mer_drifts_scalar *= eq_mer_drifts_scalar
    south_mer_drifts_scalar *= eq_mer_drifts_scalar

    # Prepare output
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

        # Get location of apex for s/c field line
        apex_xs, apex_ys, apex_zs = apex_location_info(ecef_xs, ecef_ys,
                                                       ecef_zs,
                                                       dates, ecef_input=True)

        # Magnetic field values at spacecraft
        _, _, _, b_sc = magnetic_vector(ecef_xs, ecef_ys, ecef_zs, dates)
        # Magnetic field at apex
        _, _, _, b_apex = magnetic_vector(apex_xs, apex_ys, apex_zs, dates)

        north_ftpnt, south_ftpnt = footpoint_location_info(apex_xs, apex_ys,
                                                           apex_zs, dates,
                                                           ecef_input=True)

        # Magnetic field at northern footpoint
        _, _, _, b_nft = magnetic_vector(north_ftpnt[:, 0], north_ftpnt[:, 1],
                                         north_ftpnt[:, 2], dates)

        # Magnetic field at southern footpoint
        _, _, _, b_sft = magnetic_vector(south_ftpnt[:, 0], south_ftpnt[:, 1],
                                         south_ftpnt[:, 2], dates)

        # Scalars account for change in magnetic field between locations
        south_mag_scalar = b_sc / b_sft
        north_mag_scalar = b_sc / b_nft
        eq_mag_scalar = b_sc / b_apex

        # Apply to electric field scaling to get ion drift values
        north_zon_drifts_scalar = north_zon_drifts_scalar * north_mag_scalar
        south_zon_drifts_scalar = south_zon_drifts_scalar * south_mag_scalar
        north_mer_drifts_scalar = north_mer_drifts_scalar * north_mag_scalar
        south_mer_drifts_scalar = south_mer_drifts_scalar * south_mag_scalar

        # Equatorial
        eq_zon_drifts_scalar = eq_zon_drifts_scalar * eq_mag_scalar
        eq_mer_drifts_scalar = eq_mer_drifts_scalar * eq_mag_scalar

        # Output
        out['north_zonal_drifts_scalar'] = north_zon_drifts_scalar
        out['south_zonal_drifts_scalar'] = south_zon_drifts_scalar
        out['north_mer_drifts_scalar'] = north_mer_drifts_scalar
        out['south_mer_drifts_scalar'] = south_mer_drifts_scalar
        out['equator_zonal_drifts_scalar'] = eq_zon_drifts_scalar
        out['equator_mer_drifts_scalar'] = eq_mer_drifts_scalar

    return out
