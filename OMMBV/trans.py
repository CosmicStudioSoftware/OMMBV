"""Supporting routines for coordinate transformations used by OMMBV."""

import numpy as np
import warnings

# Parameters used to define Earth ellipsoid, WGS84 parameters
earth_a = 6378.1370
earth_b = 6356.75231424518

# Standard geocentric Earth radius, average radius of Earth in km.
earth_geo_radius = 6371.

_stored_funcs = {}

try:
    from OMMBV import fortran_coords
    ecef_to_geodetic = fortran_coords.ecef_to_geodetic

except (AttributeError, NameError, ModuleNotFoundError, ImportError):
    warnings.warn("Fortran modules could not be imported.", ImportWarning)


def configure_geocentric_earth(test_mode=True):
    """Engage test configuration where Earth treated as geocentric.

    Parameters
    ----------
    test_mode : bool
        If True, test mode will be engaged and `geodetic` calls
        are replaced with `geocentric` calls. If False, the original
        functions are restored. Test mode persists until disabled.

    """

    global ecef_to_geodetic
    global python_ecef_to_geodetic
    global geodetic_to_ecef
    global _stored_funcs

    if test_mode:
        _stored_funcs = [ecef_to_geodetic, python_ecef_to_geodetic,
                         geodetic_to_ecef]
        ecef_to_geodetic = ecef_to_geocentric
        python_ecef_to_geodetic = ecef_to_geocentric
        geodetic_to_ecef = geocentric_to_ecef
    else:
        if len(_stored_funcs) > 0:
            ecef_to_geodetic = _stored_funcs[0]
            python_ecef_to_geodetic = _stored_funcs[1]
            geodetic_to_ecef = _stored_funcs[2]
            _stored_funcs = []

    return


def geocentric_to_ecef(latitude, longitude, altitude):
    """Convert geocentric coordinates into ECEF.

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
    r = earth_geo_radius + np.asarray(altitude)
    x = r * np.cos(np.deg2rad(latitude)) * np.cos(np.deg2rad(longitude))
    y = r * np.cos(np.deg2rad(latitude)) * np.sin(np.deg2rad(longitude))
    z = r * np.sin(np.deg2rad(latitude))

    return x, y, z


def ecef_to_geocentric(x, y, z, ref_height=earth_geo_radius):
    """Convert ECEF into geocentric coordinates.

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
        (default=earth_geo_radius)

    Returns
    -------
    latitude, longitude, altitude : np.array
        Locations in latitude (degrees), longitude (degrees), and
        altitude above `reference_height` in km.

    """
    # Deal with float or array-like inputs
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)

    # Radial distance
    r = np.sqrt(x**2 + y**2 + z**2)

    # Geocentric parameters
    colatitude = np.rad2deg(np.arccos(z / r))
    longitude = np.rad2deg(np.arctan2(y, x))
    latitude = 90. - colatitude

    return latitude, longitude, r - ref_height


def geodetic_to_ecef(latitude, longitude, altitude):
    """Convert WGS84 geodetic coordinates into ECEF.

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
    # Convert to radians for calculations
    latitude = np.deg2rad(latitude)
    longitude = np.deg2rad(longitude)

    # Get local ellipsoid normal radius
    ellip = np.sqrt(1. - earth_b**2 / earth_a**2)
    r_n = earth_a / np.sqrt(1. - ellip**2 * np.sin(latitude)**2)

    # Calculate ECEF locations
    x = (r_n + altitude) * np.cos(latitude) * np.cos(longitude)
    y = (r_n + altitude) * np.cos(latitude) * np.sin(longitude)
    z = (r_n * (1. - ellip**2) + altitude) * np.sin(latitude)

    return x, y, z


def python_ecef_to_geodetic(x, y, z, method='closed'):
    """Convert ECEF into Geodetic WGS84 coordinates using Python code.

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
        (http://www.oc.nps.edu/oc2902w/coord/coordcvt.pdf).
        (default = 'closed')

    Returns
    -------
    latitude, longitude, altitude : np.array
        Locations in latitude (degrees), longitude (degrees), and
        altitude above WGS84 (km)

    """
    # Quick notes on ECEF to Geodetic transformations
    # (http://danceswithcode.net/engineeringnotes/geodetic_to_ecef/
    # geodetic_to_ecef.html)

    # Ellipticity of Earth
    ellip = np.sqrt(1. - earth_b**2 / earth_a**2)

    # First eccentricity squared
    e2 = ellip**2  # 6.6943799901377997E-3

    longitude = np.arctan2(y, x)

    # Cylindrical radius
    p = np.sqrt(x**2 + y**2)

    # Closed form solution (link broken now... :( )
    # ht://www.epsg.org/Portals/0/373-07-2.pdf, page 96 section 2.2.1
    if method == 'closed':
        e_prime = np.sqrt((earth_a**2 - earth_b**2) / earth_b**2)
        theta = np.arctan2(z * earth_a, p * earth_b)
        latitude = np.arctan2(z + e_prime**2 * earth_b * np.sin(theta)**3,
                              p - e2 * earth_a * np.cos(theta)**3)
        r_n = earth_a / np.sqrt(1. - e2 * np.sin(latitude)**2)
        h = p / np.cos(latitude) - r_n
    # Another closed form possibility
    # http://ir.lib.ncku.edu.tw/bitstream/987654321/39750/1/3011200501001.pdf

    # Iterative method
    # http://www.oc.nps.edu/oc2902w/coord/coordcvt.pdf
    elif method == 'iterative':
        latitude = np.arctan2(p, z)
        r_n = earth_a / np.sqrt(1. - e2 * np.sin(latitude)**2)

        for i in np.arange(6):
            r_n = earth_a / np.sqrt(1. - e2 * np.sin(latitude)**2)
            h = p / np.cos(latitude) - r_n
            latitude = np.arctan(z / (p * (1. - e2 * (r_n / (r_n + h)))))

        # Final ellipsoidal height update
        h = p / np.cos(latitude) - r_n

    return np.rad2deg(latitude), np.rad2deg(longitude), h
