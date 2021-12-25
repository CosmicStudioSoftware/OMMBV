"""Supporting routines for vector operations used within OMMBV."""

import numpy as np


def enu_to_ecef(east, north, up, glat, glong):
    """Convert vector from East, North, Up components to ECEF.

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

    # Convert lat and lon in degrees to radians
    rlat = np.radians(glat)
    rlon = np.radians(glong)

    x = -east * np.sin(rlon) - north * np.cos(rlon) * np.sin(rlat)\
        + up * np.cos(rlon) * np.cos(rlat)
    y = east * np.cos(rlon) - north * np.sin(rlon) * np.sin(rlat)\
        + up * np.sin(rlon) * np.cos(rlat)
    z = north * np.cos(rlat) + up * np.sin(rlat)

    return x, y, z


def ecef_to_enu(x, y, z, glat, glong):
    """Convert vector from ECEF X,Y,Z components to East, North, Up.

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

    # Convert lat and lon in degrees to radians
    rlat = np.radians(glat)
    rlon = np.radians(glong)

    east = -x * np.sin(rlon) + y * np.cos(rlon)
    north = -x * np.cos(rlon) * np.sin(rlat) - y * np.sin(rlon) * np.sin(rlat)\
            + z * np.cos(rlat)
    up = x * np.cos(rlon) * np.cos(rlat) + y * np.sin(rlon) * np.cos(rlat)\
         + z * np.sin(rlat)

    return east, north, up


def project_onto_basis(x, y, z, xx, xy, xz, yx, yy, yz, zx, zy, zz):
    """Project vector onto different basis.

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

    out_x = x * xx + y * xy + z * xz
    out_y = x * yx + y * yy + z * yz
    out_z = x * zx + y * zy + z * zz

    return out_x, out_y, out_z


def normalize(x, y, z):
    """Normalize vector to produce a unit vector.

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
    x = x / mag
    y = y / mag
    z = z / mag

    return x, y, z


def cross_product(x1, y1, z1, x2, y2, z2):
    """Cross product of two vectors, v1 x v2.

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
    x = y1 * z2 - y2 * z1
    y = z1 * x2 - x1 * z2
    z = x1 * y2 - y1 * x2

    return x, y, z
