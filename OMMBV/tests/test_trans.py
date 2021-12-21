"""Unit tests for `OMMBV.trans`."""

import numpy as np

import OMMBV
from OMMBV import igrf
from OMMBV import sources
import OMMBV.trans
from OMMBV.tests.test_core import gen_data_fixed_alt


# ############## TRANSFORMATIONS ############## #


def assert_difference_tol(data, data2, tol=1.E-5):
    """Assert absolute difference is less than `tol`

    Parameters
    ----------
    data, data2 : np.array
      Data to be differenced.
    tol : float
        Tolerance allowed between data

    """

    diff = np.abs(data2 - data)
    assert np.all(diff < tol)

    return


class TestTransformations(object):

    def setup(self):
        """Setup test environment before each function."""

        # Locations to perform tests at
        self.lats, self.longs, self.alts = gen_data_fixed_alt(550.)
        return

    def teardown(self):
        """Clean up test environment after each function."""

        del self.lats, self.longs, self.alts
        return

    def test_geocentric_to_ecef_to_geocentric(self):
        """Test Geocentric and ECEF transformations, round trip equality"""

        ecf_x, ecf_y, ecf_z = OMMBV.trans.geocentric_to_ecef(self.lats,
                                                             self.longs,
                                                             self.alts)

        lat, elong, alt = OMMBV.trans.ecef_to_geocentric(ecf_x, ecf_y, ecf_z)

        idx, = np.where(elong < 0)
        elong[idx] += 360.

        assert_difference_tol(lat, self.lats)
        assert_difference_tol(elong, self.longs)
        assert_difference_tol(alt, self.alts)

        return

    def test_geocentric_to_ecef_floats_and_list(self):
        """Geocentric and ECEF transformations different input types"""

        ecf_x, ecf_y, ecf_z = OMMBV.trans.geocentric_to_ecef(self.lats[:1],
                                                             self.longs[:1],
                                                             self.alts[:1])

        ecf_x2, ecf_y2, ecf_z2 = OMMBV.trans.geocentric_to_ecef(self.lats[0],
                                                                self.longs[0],
                                                                self.alts[0])
        assert_difference_tol(ecf_x, ecf_x2)
        assert_difference_tol(ecf_y, ecf_y2)
        assert_difference_tol(ecf_z, ecf_z2)

        return

    def test_ecef_to_geocentric_floats_and_list(self):
        """Geocentric and ECEF transformations with different input types"""

        # Generate ECEF starting locations
        secf_x, secf_y, secf_z = OMMBV.trans.geocentric_to_ecef(self.lats[:1],
                                                                self.longs[:1],
                                                                self.alts[:1])
        # Convert to geo with list input
        ecf_x, ecf_y, ecf_z = OMMBV.trans.ecef_to_geocentric(secf_x[:1],
                                                             secf_y[:1],
                                                             secf_z[:1])

        # Convert to geo with float input
        ecf_x2, ecf_y2, ecf_z2 = OMMBV.trans.geocentric_to_ecef(ecf_x[0],
                                                                ecf_y[0],
                                                                ecf_z[0])
        assert_difference_tol(secf_x, ecf_x2)
        assert_difference_tol(secf_y, ecf_y2)
        assert_difference_tol(secf_z, ecf_z2)

        return

    def test_geodetic_to_ecef_floats_and_list(self):
        """Geodetic and ECEF transformations different input types"""

        ecf_x, ecf_y, ecf_z = OMMBV.trans.geodetic_to_ecef(self.lats[:1],
                                                           self.longs[:1],
                                                           self.alts[:1])

        ecf_x2, ecf_y2, ecf_z2 = OMMBV.trans.geodetic_to_ecef(self.lats[0],
                                                              self.longs[0],
                                                              self.alts[0])

        assert_difference_tol(ecf_x, ecf_x2)
        assert_difference_tol(ecf_y, ecf_y2)
        assert_difference_tol(ecf_z, ecf_z2)

        return

    def test_ecef_to_geodetic_floats_and_list(self):
        """Geodetic and ECEF transformations different input types"""

        # Generate ECEF starting locations
        secf_x, secf_y, secf_z = OMMBV.trans.geodetic_to_ecef(self.lats[:1],
                                                              self.longs[:1],
                                                              self.alts[:1])
        # Convert to geo with list input
        ecf_x, ecf_y, ecf_z = OMMBV.trans.ecef_to_geodetic(secf_x[:1],
                                                           secf_y[:1],
                                                           secf_z[:1])

        # Convert to geo with float input
        ecf_x2, ecf_y2, ecf_z2 = OMMBV.trans.geodetic_to_ecef(ecf_x[0],
                                                              ecf_y[0],
                                                              ecf_z[0])

        assert_difference_tol(secf_x, ecf_x2)
        assert_difference_tol(secf_y, ecf_y2)
        assert_difference_tol(secf_z, ecf_z2)

        return

    def test_geodetic_to_ecef_to_geodetic(self):
        """Test geodetic to ECEF and back, both fortran and python"""

        ecf_x, ecf_y, ecf_z = OMMBV.trans.geodetic_to_ecef(self.lats,
                                                           self.longs,
                                                           self.alts)

        lat, elong, alt = OMMBV.trans.ecef_to_geodetic(ecf_x, ecf_y, ecf_z)

        lat2, elong2, alt2 = OMMBV.trans.python_ecef_to_geodetic(ecf_x, ecf_y,
                                                                 ecf_z)

        idx, = np.where(elong < 0)
        elong[idx] += 360.

        idx, = np.where(elong2 < 0)
        elong2[idx] += 360.

        assert_difference_tol(lat, self.lats)
        assert_difference_tol(elong, self.longs)
        assert_difference_tol(alt, self.alts)

        assert_difference_tol(lat2, self.lats)
        assert_difference_tol(elong2, self.longs)
        assert_difference_tol(alt2, self.alts)

        return

    def test_geodetic_to_ecef_to_geodetic_via_different_methods(self):
        """Multiple techniques for geodetic to ECEF to geodetic"""

        ecf_x, ecf_y, ecf_z = OMMBV.trans.geodetic_to_ecef(self.lats,
                                                           self.longs,
                                                           self.alts)
        methods = ['closed', 'iterative']
        for method in methods:
            lat, elong, alt = OMMBV.trans.python_ecef_to_geodetic(ecf_x, ecf_y,
                                                                  ecf_z,
                                                                  method=method)

            idx, = np.where(elong < 0)
            elong[idx] += 360.

            assert_difference_tol(lat, self.lats)
            assert_difference_tol(elong, self.longs)
            assert_difference_tol(alt, self.alts)

        return

    def test_geodetic_to_ecef_to_geocentric_to_ecef_to_geodetic(self):
        """Test geodetic to ecef and geocentric transformations"""

        ecf_x, ecf_y, ecf_z = OMMBV.trans.geodetic_to_ecef(self.lats,
                                                           self.longs,
                                                           self.alts)

        geo_lat, geo_long, geo_alt = OMMBV.trans.ecef_to_geocentric(ecf_x,
                                                                    ecf_y,
                                                                    ecf_z)

        ecfs_x, ecfs_y, ecfs_z = OMMBV.trans.geocentric_to_ecef(geo_lat,
                                                                geo_long,
                                                                geo_alt)

        lat, elong, alt = OMMBV.ecef_to_geodetic(ecfs_x, ecfs_y, ecfs_z)

        idx, = np.where(elong < 0)
        elong[idx] += 360.

        assert_difference_tol(lat, self.lats)
        assert_difference_tol(elong, self.longs)
        assert_difference_tol(alt, self.alts)

        assert_difference_tol(ecf_x, ecfs_x)
        assert_difference_tol(ecf_y, ecfs_y)
        assert_difference_tol(ecf_z, ecfs_z)

        return

    def test_igrf_ecef_to_geodetic_back_to_ecef(self):
        """Test IGRF_ECEF - Geodetic - and Back"""

        ecf_x, ecf_y, ecf_z = OMMBV.trans.geodetic_to_ecef(self.lats,
                                                           self.longs,
                                                           self.alts)
        for (ecef_x, ecef_y, ecef_z,
             geo_lat, geo_lon, geo_alt) in zip(ecf_x, ecf_y, ecf_z,
                                               self.lats, self.longs,
                                               self.alts):

            pos = np.array([ecef_x, ecef_y, ecef_z])

            lat, elong, alt = sources.ecef_to_geodetic(pos)
            lat = np.rad2deg(lat)
            elong = np.rad2deg(elong)
            if (elong < 0):
                elong = elong + 360.

            assert_difference_tol(lat, geo_lat)
            assert_difference_tol(elong, geo_lon)
            assert_difference_tol(alt, geo_alt)

        return

    def test_igrf_ecef_to_geographic_with_colatitude(self):
        """Test IGRF_ECEF - Geographic"""

        ecf_x, ecf_y, ecf_z = OMMBV.trans.geodetic_to_ecef(self.lats,
                                                           self.longs,
                                                           self.alts)

        for (ecef_x, ecef_y, ecef_z,
             geo_lat, geo_lon, geo_alt) in zip(ecf_x, ecf_y, ecf_z,
                                               self.lats, self.longs,
                                               self.alts):

            pos = np.array([ecef_x, ecef_y, ecef_z])

            colat, lon, r = sources.ecef_to_colat_long_r(pos)

            # Results are returned in radians
            lat = 90. - np.rad2deg(colat)
            lon = np.rad2deg(lon)

            lat2, lon2, h2 = OMMBV.trans.ecef_to_geocentric(*pos, ref_height=0)

            assert_difference_tol(r, h2)
            assert_difference_tol(lat, lat2)
            assert_difference_tol(lon, lon2)

        return
