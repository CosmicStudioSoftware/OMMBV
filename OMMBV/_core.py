"""Primary functions for calculating magnetic basis vectors."""

import numpy as np
import warnings

from OMMBV.trace import apex_location_info
from OMMBV.trace import footpoint_location_info
from OMMBV.trace import magnetic_vector
from OMMBV import trans
from OMMBV import vector


def calculate_geomagnetic_basis(latitude, longitude, altitude, datetimes):
    """Calculate local geomagnetic basis vectors and mapping scalars.

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

    Note
    ----
        Thin wrapper around `calculate_mag_drift_unit_vectors_ecef` set
        to default parameters and with more organization of the outputs.

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
             'd_zon_mag': d_zon_mag, 'd_fa_mag': d_fa_mag,
             'd_mer_mag': d_mer_mag,
             'e_zon_mag': e_zon_mag, 'e_fa_mag': e_fa_mag,
             'e_mer_mag': e_mer_mag}

    return out_d


def calculate_mag_drift_unit_vectors_ecef(latitude, longitude, altitude,
                                          datetimes, step_size=0.5, tol=1.E-4,
                                          tol_zonal_apex=1.E-4, max_loops=15,
                                          ecef_input=False, centered_diff=True,
                                          full_output=False,
                                          include_debug=False,
                                          scalar=None, edge_steps=None,
                                          dstep_size=0.5, max_steps=None,
                                          ref_height=None, steps=None,
                                          pole_tol=1.E-5,
                                          location_info=apex_location_info,
                                          mag_fcn=None, step_fcn=None,
                                          min_loops=3, apex_kwargs=None):
    """Calculate local geomagnetic basis vectors and mapping scalars.

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
    mag_fcn : function
        Function used to get information on local magnetic field. If None,
        uses default functions for IGRF. (default=None).
    step_fcn : function
        Function used to step along magnetic field. If None,
        uses default functions for IGRF. (default=None).
    min_loops : int
        Minimum number of iterations to attempt when calculating basis.
        (default=3)
    apex_kwargs : dict or NoneType
        If dict supplied, passed to `apex_location_info` as keyword arguments.
        (default=None)

    Returns
    -------
    zon_x, zon_y, zon_z, fa_x, fa_y, fa_z, mer_x, mer_y, mer_z,
    (optional dictionary)

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
    depd = {'max_steps': max_steps, 'ref_height': ref_height,
            'steps': steps, 'scalar': scalar, 'edge_steps': edge_steps}
    for key in depd.keys():
        if depd[key] is not None:
            wstr = " ".join([key, "is deprecated, non-functional,",
                             "and will be removed after OMMBV v1.0.0."])
            warnings.warn(wstr, DeprecationWarning, stacklevel=2)

    # Account for potential alternate magnetic field and step functions.
    # Functions aren't assigned as default in function definition since they
    # are Fortran functions and places like RTD don't support Fortran easily.
    mag_kwargs = {} if mag_fcn is None else {'mag_fcn': mag_fcn}
    step_kwargs = {} if step_fcn is None else {'step_fcn': step_fcn}

    # Keywords intended for `apex_location_info`
    apex_kwargs = {} if apex_kwargs is None else apex_kwargs
    for key in apex_kwargs.keys():
        step_kwargs[key] = apex_kwargs[key]

    # Check for reasonable user inputs
    if step_size <= 0:
        raise ValueError('`step_size` must be greater than 0.')

    # Check for reasonable user inputs
    checks = [longitude, altitude, datetimes]
    for item in checks:
        if len(latitude) != len(item):
            estr = ''.join(['All inputs (latitude, longitude, altitude, ',
                            'datetimes) must have the same length.'])
            raise ValueError(estr)

    if ecef_input:
        ecef_x, ecef_y, ecef_z = latitude, longitude, altitude
        # Latitude and longitude position needed later for initial zonal and
        # meridional vector generation.
        latitude, longitude, altitude = trans.ecef_to_geodetic(ecef_x, ecef_y,
                                                               ecef_z)
    else:
        latitude = np.array(latitude, dtype=np.float64)
        longitude = np.array(longitude, dtype=np.float64)
        altitude = np.array(altitude, dtype=np.float64)

        # Ensure latitude reasonable
        idx, = np.where(np.abs(latitude) > 90.)
        if len(idx) > 0:
            raise ValueError('Latitude out of bounds [-90., 90.].')

        # Ensure longitude reasonable
        idx, = np.where((longitude < -180.) | (longitude > 360.))
        if len(idx) > 0:
            raise ValueError('Longitude out of bounds [-180., 360.].')

        # Calculate satellite position in ECEF coordinates
        ecef_x, ecef_y, ecef_z = trans.geodetic_to_ecef(latitude, longitude,
                                                        altitude)

    # Begin method calculation.

    # Magnetic field at root location
    bx, by, bz, bm = magnetic_vector(ecef_x, ecef_y, ecef_z, datetimes,
                                     normalize=True, **mag_kwargs)

    # If magnetic field is pointed purely upward, then full basis can't
    # be calculated. Check for this condition and store locations.
    be, bn, bu = vector.ecef_to_enu(bx, by, bz, latitude, longitude)
    null_idx, = np.where(np.abs(bu) > 1. - pole_tol)

    # To start, need a vector perpendicular to mag field. There are infinitely
    # many, thus, let's use the east vector as a start.
    tzx, tzy, tzz = vector.enu_to_ecef(np.ones(len(bx)), np.zeros(len(bx)),
                                       np.zeros(len(bx)), latitude, longitude)
    init_type = np.zeros(len(bx)) - 1

    # Get meridional direction via cross with field-aligned and normalize
    tmx, tmy, tmz = vector.cross_product(tzx, tzy, tzz, bx, by, bz)
    tmx, tmy, tmz = vector.normalize(tmx, tmy, tmz)

    # Get orthogonal zonal now, and normalize.
    tzx, tzy, tzz = vector.cross_product(bx, by, bz, tmx, tmy, tmz)
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
                                                   ecef_input=True,
                                                   **step_kwargs)

    # Initialize loop variables
    loop_num = 0
    repeat_flag = True

    # Iteratively determine the vector basis directions.
    while repeat_flag or (loop_num < min_loops):
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
                                              ecef_input=True,
                                              **step_kwargs)
        if centered_diff:
            # Negative step
            ecef_xz2, ecef_yz2, ecef_zz2 = (ecef_x - step_size * tzx,
                                            ecef_y - step_size * tzy,
                                            ecef_z - step_size * tzz)
            _, _, _, _, _, apex_z2 = location_info(ecef_xz2, ecef_yz2, ecef_zz2,
                                                   datetimes,
                                                   return_geodetic=True,
                                                   ecef_input=True,
                                                   **step_kwargs)
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
                                              ecef_input=True,
                                              **step_kwargs)

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

        # Check if we are done
        if np.isnan(diff):
            # All values are np.nan.
            loop_num = min_loops - 1
            repeat_flag = False

        if (diff < tol) & (np.nanmax(np.abs(diff_apex_z)) < tol_zonal_apex) \
                & (loop_num + 1 >= min_loops):
            # Reached terminating conditions
            repeat_flag = False
        else:
            # Store info into calculation vectors to refine next loop
            tzx, tzy, tzz = tzx2, tzy2, tzz2
            tmx, tmy, tmz = tmx2, tmy2, tmz2

        loop_num += 1
        if loop_num >= max_loops:
            # Identify all locations with tolerance failure
            idx1, = np.where(diff_m >= tol)
            idx2, = np.where(np.abs(diff_apex_z) >= tol_zonal_apex)
            idx = np.hstack((idx1, idx2))

            # Assign nan to vectors
            tzx[idx], tzy[idx], tzz[idx] = np.nan, np.nan, np.nan
            tmx[idx], tmy[idx], tmz[idx] = np.nan, np.nan, np.nan

            estr = ''.join((str(len(idx)), ' locations did not converge.',
                            ' Setting to NaN.'))
            warnings.warn(estr, RuntimeWarning)
            break

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
        t_x1, t_y1, t_z1, _, _, apex_z = location_info(ecef_xz, ecef_yz,
                                                       ecef_zz, datetimes,
                                                       return_geodetic=True,
                                                       ecef_input=True,
                                                       **step_kwargs)

        # Negative zonal step
        ecef_xz2, ecef_yz2, ecef_zz2 = (ecef_x - dstep_size * zx,
                                        ecef_y - dstep_size * zy,
                                        ecef_z - dstep_size * zz)
        # Corresponding apex location
        (t_x2, t_y2, t_z2,
         _, _, apex_z2) = location_info(ecef_xz2, ecef_yz2, ecef_zz2,
                                        datetimes, return_geodetic=True,
                                        ecef_input=True,
                                        **step_kwargs)

        # Basis vectors at apex location
        tolza = tol_zonal_apex
        (azx, azy, azz,
         _, _, _, amx, amy, amz
         ) = calculate_mag_drift_unit_vectors_ecef(a_x, a_y, a_z, datetimes,
                                                   ecef_input=True,
                                                   step_size=step_size,
                                                   tol=tol,
                                                   tol_zonal_apex=tolza,
                                                   mag_fcn=mag_fcn,
                                                   step_fcn=step_fcn)

        # Get distance between apex points along apex zonal direction
        dist = (t_x1 - t_x2) * azx + (t_y1 - t_y2) * azy + (t_z1 - t_z2) * azz

        # Calculate gradient in field line separation distance
        grad_brb = dist / (2. * dstep_size)

        # Calculate gradient in apex height
        diff_apex_z = apex_z - apex_z2
        grad_zonal = diff_apex_z / (2. * dstep_size)

        # Get magnitude of magnetic field at root apex location
        bax, bay, baz, bam = magnetic_vector(a_x, a_y, a_z, datetimes,
                                             normalize=True,
                                             **mag_kwargs)

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
        d_mer_x, d_mer_y, d_mer_z = vector.cross_product(d_zon_x, d_zon_y,
                                                         d_zon_z, d_fa_x,
                                                         d_fa_y, d_fa_z)
        mag = d_mer_x**2 + d_mer_y**2 + d_mer_z**2
        d_mer_x, d_mer_y, d_mer_z = d_mer_x / mag, d_mer_y / mag, d_mer_z / mag

        # e vectors (Richmond nomenclature)
        # Zonal
        e_zon_x, e_zon_y, e_zon_z = vector.cross_product(d_fa_x, d_fa_y, d_fa_z,
                                                         d_mer_x, d_mer_y,
                                                         d_mer_z)
        # Field-Aligned
        e_fa_x, e_fa_y, e_fa_z = vector.cross_product(d_mer_x, d_mer_y, d_mer_z,
                                                      d_zon_x, d_zon_y, d_zon_z)
        # Meridional
        e_mer_x, e_mer_y, e_mer_z = vector.cross_product(d_zon_x, d_zon_y,
                                                         d_zon_z, d_fa_x,
                                                         d_fa_y, d_fa_z)

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
            ecef_xm, ecef_ym, ecef_zm = (ecef_x + dstep_size * mx,
                                         ecef_y + dstep_size * my,
                                         ecef_z + dstep_size * mz)
            t_x1, t_y1, t_z1 = location_info(ecef_xm, ecef_ym, ecef_zm,
                                             datetimes, ecef_input=True,
                                             **step_kwargs)

            # Negative step
            ecef_xm2, ecef_ym2, ecef_zm2 = (ecef_x - dstep_size * mx,
                                            ecef_y - dstep_size * my,
                                            ecef_z - dstep_size * mz)
            t_x2, t_y2, t_z2 = location_info(ecef_xm2, ecef_ym2, ecef_zm2,
                                             datetimes, ecef_input=True,
                                             **step_kwargs)

            # Get distance between apex locations along meridional vector
            diff_apex_m = (t_x1 - t_x2) * amx + (t_y1 - t_y2) * amy \
                          + (t_z1 - t_z2) * amz

            # Apex height gradient
            grad_apex = diff_apex_m / (2. * dstep_size)

            # second path D, E vectors
            # d meridional vector via apex height gradient
            d_mer2_x, d_mer2_y, d_mer2_z = (grad_apex * mx, grad_apex * my,
                                            grad_apex * mz)

            # Zonal to complete set
            (d_zon2_x,
             d_zon2_y,
             d_zon2_z) = vector.cross_product(d_fa_x, d_fa_y, d_fa_z,
                                              d_mer2_x, d_mer2_y, d_mer2_z)
            mag = d_zon2_x**2 + d_zon2_y**2 + d_zon2_z**2
            d_zon2_x, d_zon2_y, d_zon2_z = (d_zon2_x / mag, d_zon2_y / mag,
                                            d_zon2_z / mag)

            tempd = {'grad_zonal_apex': grad_zonal,
                     'grad_mer_apex': grad_apex,
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


def step_along_mag_unit_vector(x, y, z, date, direction, num_steps=1,
                               step_size=25., scalar=1, **kwargs):
    """Move by following specified magnetic unit vector direction.

    Moving along the field is effectively the same as a field line trace though
    extended movement along a field should use the specific `field_line_trace`
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
    direction : str
        String identifier for which unit vector direction to move along.
        Supported inputs, 'meridional', 'zonal', 'aligned'
    num_steps : int
        Number of steps to take along unit vector direction (default=1)
    step_size : float
        Distance taken for each step (km) (default=25.)
    scalar : int
        Scalar modifier for step size distance. Input a -1 to move along
        negative unit vector direction. (default=1)
    **kwargs : Additional keywords
        Passed to `calculate_mag_drift_unit_vectors_ecef`.

    Returns
    -------
    np.array
        [x, y, z] of ECEF location after taking num_steps along direction,
        each step_size long.

    Note
    ----
        `centered_diff=True` is passed along to
        `calculate_mag_drift_unit_vectors_ecef` when direction='meridional',
        while `centered_diff=False` is used for the 'zonal' direction.
        This ensures that when moving along the zonal direction there is a
        minimal change in apex height.

    """

    if direction == 'meridional':
        centered_diff = True
    else:
        centered_diff = False

    for i in np.arange(num_steps):
        # Get unit vector directions; x, y, z in ECEF
        (zvx, zvy, zvz,
         bx, by, bz,
         mx, my, mz
         ) = calculate_mag_drift_unit_vectors_ecef(x, y, z, date,
                                                   step_size=step_size,
                                                   ecef_input=True,
                                                   centered_diff=centered_diff,
                                                   **kwargs)
        # Pull out the direction we need
        if direction == 'meridional':
            ux, uy, uz = mx, my, mz
        elif direction == 'zonal':
            ux, uy, uz = zvx, zvy, zvz
        elif direction == 'aligned':
            ux, uy, uz = bx, by, bz

        # Take step along direction
        x = x + step_size * ux * scalar
        y = y + step_size * uy * scalar
        z = z + step_size * uz * scalar

    return x, y, z


def scalars_for_mapping_ion_drifts(glats, glons, alts, dates,
                                   max_steps=None, e_field_scaling_only=None,
                                   edge_length=None, edge_steps=None,
                                   **kwargs):
    """Translate ion drifts and electric fields to equator and footpoints.

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
    **kwargs : Additional keywords
        Passed to `calculate_mag_drift_unit_vectors_ecef`. `step_fcn`, if
        present, is also passed to `footpoint_location_info`.

    Returns
    -------
    dict
        array-like of scalars for translating ion drifts. Keys are,
        'north_zonal_drifts_scalar', 'north_mer_drifts_scalar', and similarly
        for southern locations. 'equator_mer_drifts_scalar' and
        'equator_zonal_drifts_scalar' cover the mappings to the equator.


    """

    # Check for deprecated inputs
    depd = {'e_field_scaling_only': e_field_scaling_only,
            'max_steps': max_steps, 'edge_length': edge_length,
            'edge_steps': edge_steps}
    for key in depd.keys():
        if depd[key] is not None:
            wstr = " ".join([key, "is deprecated, non-functional,",
                             "and will be removed after OMMBV v1.0.0."])
            warnings.warn(wstr, DeprecationWarning, stacklevel=2)

    ecef_xs, ecef_ys, ecef_zs = trans.geodetic_to_ecef(glats, glons, alts)

    # Get footpoint location information, passing along relevant kwargs
    flid = {'step_fcn': kwargs['step_fcn']} if 'step_fcn' in kwargs else {}
    north_ftpnt, south_ftpnt = footpoint_location_info(ecef_xs, ecef_ys,
                                                       ecef_zs, dates,
                                                       ecef_input=True,
                                                       **flid)

    # Prepare output memory
    out = {}

    # D and E vectors at user supplied location. Good for mapping to
    # magnetic equator.
    (_, _, _, _, _, _, _, _, _,
     infod) = calculate_mag_drift_unit_vectors_ecef(ecef_xs, ecef_ys, ecef_zs,
                                                    dates, full_output=True,
                                                    include_debug=True,
                                                    ecef_input=True, **kwargs)

    out['equator_zon_fields_scalar'] = np.sqrt(infod['e_zon_x']**2
                                               + infod['e_zon_y']**2
                                               + infod['e_zon_z']**2)
    out['equator_mer_fields_scalar'] = np.sqrt(infod['e_mer_x']**2
                                               + infod['e_mer_y']**2
                                               + infod['e_mer_z']**2)

    out['equator_zon_drifts_scalar'] = np.sqrt(infod['d_zon_x']**2
                                               + infod['d_zon_y']**2
                                               + infod['d_zon_z']**2)
    out['equator_mer_drifts_scalar'] = np.sqrt(infod['d_mer_x']**2
                                               + infod['d_mer_y']**2
                                               + infod['d_mer_z']**2)

    # D and E vectors at northern footpoint
    (_, _, _, _, _, _, _, _, _,
     northd) = calculate_mag_drift_unit_vectors_ecef(north_ftpnt[:, 0],
                                                     north_ftpnt[:, 1],
                                                     north_ftpnt[:, 2], dates,
                                                     full_output=True,
                                                     include_debug=True,
                                                     ecef_input=True, **kwargs)

    # D and E vectors at northern footpoint
    (_, _, _, _, _, _, _, _, _,
     southd) = calculate_mag_drift_unit_vectors_ecef(south_ftpnt[:, 0],
                                                     south_ftpnt[:, 1],
                                                     south_ftpnt[:, 2], dates,
                                                     full_output=True,
                                                     include_debug=True,
                                                     ecef_input=True, **kwargs)

    # Prepare output.
    # To map fields from r1 to r2, (E dot e1) d2
    out['north_mer_fields_scalar'] = np.sqrt(infod['e_mer_x']**2
                                             + infod['e_mer_y']**2
                                             + infod['e_mer_z']**2)
    out['north_mer_fields_scalar'] *= np.sqrt(northd['d_mer_x']**2
                                              + northd['d_mer_y']**2
                                              + northd['d_mer_z']**2)

    # To map drifts from r1 to r2, (v dot d1) e2
    out['north_mer_drifts_scalar'] = np.sqrt(infod['d_mer_x']**2
                                             + infod['d_mer_y']**2
                                             + infod['d_mer_z']**2)
    out['north_mer_drifts_scalar'] *= np.sqrt(northd['e_mer_x']**2
                                              + northd['e_mer_y']**2
                                              + northd['e_mer_z']**2)

    # To map fields from r1 to r2, (E dot e1) d2
    out['north_zon_fields_scalar'] = np.sqrt(infod['e_zon_x']**2
                                             + infod['e_zon_y']**2
                                             + infod['e_zon_z']**2)
    out['north_zon_fields_scalar'] *= np.sqrt(northd['d_zon_x']**2
                                              + northd['d_zon_y']**2
                                              + northd['d_zon_z']**2)

    # To map drifts from r1 to r2, (v dot d1) e2
    out['north_zon_drifts_scalar'] = np.sqrt(infod['d_zon_x']**2
                                             + infod['d_zon_y']**2
                                             + infod['d_zon_z']**2)
    out['north_zon_drifts_scalar'] *= np.sqrt(northd['e_zon_x']**2
                                              + northd['e_zon_y']**2
                                              + northd['e_zon_z']**2)

    # To map fields from r1 to r2, (E dot e1) d2
    out['south_mer_fields_scalar'] = np.sqrt(infod['e_mer_x']**2
                                             + infod['e_mer_y']**2
                                             + infod['e_mer_z']**2)
    out['south_mer_fields_scalar'] *= np.sqrt(southd['d_mer_x']**2
                                              + southd['d_mer_y']**2
                                              + southd['d_mer_z']**2)

    # To map drifts from r1 to r2, (v dot d1) e2
    out['south_mer_drifts_scalar'] = np.sqrt(infod['d_mer_x']**2
                                             + infod['d_mer_y']**2
                                             + infod['d_mer_z']**2)
    out['south_mer_drifts_scalar'] *= np.sqrt(southd['e_mer_x']**2
                                              + southd['e_mer_y']**2
                                              + southd['e_mer_z']**2)

    # To map fields from r1 to r2, (E dot e1) d2
    out['south_zon_fields_scalar'] = np.sqrt(infod['e_zon_x']**2
                                             + infod['e_zon_y']**2
                                             + infod['e_zon_z']**2)
    out['south_zon_fields_scalar'] *= np.sqrt(southd['d_zon_x']**2
                                              + southd['d_zon_y']**2
                                              + southd['d_zon_z']**2)

    # To map drifts from r1 to r2, (v dot d1) e2
    out['south_zon_drifts_scalar'] = np.sqrt(infod['d_zon_x']**2
                                             + infod['d_zon_y']**2
                                             + infod['d_zon_z']**2)
    out['south_zon_drifts_scalar'] *= np.sqrt(southd['e_zon_x']**2
                                              + southd['e_zon_y']**2
                                              + southd['e_zon_z']**2)

    return out


def geocentric_to_ecef(latitude, longitude, altitude):
    """Convert geocentric coordinates into ECEF.

    .. deprecated:: 1.0.0
       Function moved to `OMMBV.trans.geocentric_to_ecef`, this wrapper will
       be removed after v1.0.0.

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
                           "wrapper will be removed after OMMBV v1.0.0."]),
                  DeprecationWarning, stacklevel=2)

    return trans.geocentric_to_ecef(latitude, longitude, altitude)


def ecef_to_geocentric(x, y, z, ref_height=trans.earth_geo_radius):
    """Convert ECEF into geocentric coordinates.

    .. deprecated:: 1.0.0
       Function moved to `OMMBV.trans.ecef_to_geocentric`, this wrapper will
       be removed after v1.0.0.

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
                           "wrapper will be removed after OMMBV v1.0.0."]),
                  DeprecationWarning, stacklevel=2)

    return trans.ecef_to_geocentric(x, y, z, ref_height=ref_height)


def geodetic_to_ecef(latitude, longitude, altitude):
    """Convert WGS84 geodetic coordinates into ECEF.

    .. deprecated:: 1.0.0
       Function moved to `OMMBV.trans.geodetic_to_ecef`, this wrapper will
       be removed after v1.0.0.

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
                           "wrapper will be removed after OMMBV v1.0.0."]),
                  DeprecationWarning, stacklevel=2)

    return trans.geodetic_to_ecef(latitude, longitude, altitude)


def ecef_to_geodetic(*args, **kwargs):
    """Convert ECEF into Geodetic WGS84 coordinates.

    .. deprecated:: 1.0.0
       Function moved to `OMMBV.trans.geodetic_to_ecef`, this wrapper will
       be removed after v1.0.0.

    """

    warnings.warn("".join(["Function moved to `OMMBV.trans`, deprecated ",
                           "wrapper will be removed after OMMBV v1.0.0."]),
                  DeprecationWarning, stacklevel=2)

    return trans.ecef_to_geodetic(*args, **kwargs)


def python_ecef_to_geodetic(x, y, z, method='closed'):
    """Convert ECEF into Geodetic WGS84 coordinates using Python code.

    .. deprecated:: 1.0.0
       Function moved to `OMMBV.trans.geodetic_to_ecef`, this wrapper will
       be removed after v1.0.0.

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
                           "wrapper will be removed after OMMBV v1.0.0."]),
                  DeprecationWarning, stacklevel=2)

    if method is None:
        warnings.warn("".join(["`method` must be a string value in ",
                               "v1.0.0+. Setting to function default."]),
                      DeprecationWarning, stacklevel=2)
        method = 'closed'

    return trans.python_ecef_to_geodetic(x, y, z, method=method)


def enu_to_ecef_vector(east, north, up, glat, glong):
    """Convert vector from East, North, Up components to ECEF.

    .. deprecated:: 1.0.0
       Function moved to `OMMBV.vector.enu_to_ecef`, this wrapper will
       be removed after v1.0.0.

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
    glat : float or array_like
        Geodetic or geocentric latitude (degrees)
    glong : float or array_like
        Geodetic or geocentric longitude (degrees)

    Returns
    -------
    x, y, z : np.array
        Vector components along ECEF x, y, and z directions

    """

    warnings.warn("".join(["Function moved to `OMMBV.vector`, deprecated ",
                           "wrapper will be removed after OMMBV v1.0.0."]),
                  DeprecationWarning, stacklevel=2)

    return vector.enu_to_ecef(east, north, up, glat, glong)


def ecef_to_enu_vector(x, y, z, glat, glong):
    """Convert vector from ECEF X,Y,Z components to East, North, Up.

    .. deprecated:: 1.0.0
       Function moved to `OMMBV.vector.ecef_to_enu`, this wrapper will
       be removed after v1.0.0.

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
    glat : float or array_like
        Geodetic or geocentric latitude (degrees)
    glong : float or array_like
        Geodetic or geocentric longitude (degrees)

    Returns
    -------
    east, north, up : np.array
        Vector components along east, north, and up directions

    """

    warnings.warn("".join(["Function moved to `OMMBV.vector`, deprecated ",
                           "wrapper will be removed after OMMBV v1.0.0."]),
                  DeprecationWarning, stacklevel=2)

    return vector.ecef_to_enu(x, y, z, glat, glong)


def project_ECEF_vector_onto_basis(x, y, z, xx, xy, xz, yx, yy, yz, zx, zy, zz):
    """Project vector onto different basis.

    .. deprecated:: 1.0.0
       Function moved to `OMMBV.vector.project_onto_basis`, this wrapper will
       be removed after v1.0.0.

    Parameters
    ----------
    x : float or array-like
        X component of vector
    y : float or array-like
        Y component of vector
    z : float or array-like
        Z component of vector
    xx, yx, zx : float or array-like
        X component of the x, y, z unit vector of new basis in original basis
    xy, yy, zy : float or array-like
        Y component of the x, y, z unit vector of new basis in original basis
    xz, yz, zz : float or array-like
        Z component of the x, y, z unit vector of new basis in original basis

    Returns
    -------
    x, y, z
        Vector projected onto new basis

    """

    warnings.warn("".join(["Function moved to `OMMBV.vector`, deprecated ",
                           "wrapper will be removed after OMMBV v1.0.0."]),
                  DeprecationWarning, stacklevel=2)

    return vector.project_onto_basis(x, y, z, xx, xy, xz, yx, yy, yz,
                                     zx, zy, zz)


def normalize_vector(x, y, z):
    """Normalize vector to produce a unit vector.

    .. deprecated:: 1.0.0
       Function moved to `OMMBV.vector.normalize`, this wrapper will
       be removed after v1.0.0.

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

    warnings.warn("".join(["Function moved to `OMMBV.vector`, deprecated ",
                           "wrapper will be removed after OMMBV v1.0.0."]),
                  DeprecationWarning, stacklevel=2)

    return vector.normalize(x, y, z)


def cross_product(x1, y1, z1, x2, y2, z2):
    """Cross product of two vectors, v1 x v2.

    .. deprecated:: 1.0.0
       Function moved to `OMMBV.vector.cross_product`, this wrapper will
       be removed after v1.0.0.

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

    warnings.warn("".join(["Function moved to `OMMBV.vector`, deprecated ",
                           "wrapper will be removed after OMMBV v1.0.0."]),
                  DeprecationWarning, stacklevel=2)

    return vector.cross_product(x1, y1, z1, x2, y2, z2)
