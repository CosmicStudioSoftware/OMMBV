import numpy as np
import pytest

import OMMBV
from OMMBV import igrf
from OMMBV.tests.test_core import gen_data_fixed_alt


class TestVector(object):
    """Unit tests for `OMMBV.vector`."""

    def setup(self):
        """Setup test environment before each function."""

        # Locations to perform tests at
        self.lats, self.longs, self.alts = gen_data_fixed_alt(550.)

        # self.start_date = dt.datetime(2000, 1, 1)
        # self.end_date = dt.datetime(2020, 12, 31)
        # self.dates = pysat.utils.time.create_date_range(self.start_date,
        #                                                 self.end_date)
        return

    def teardown(self):
        """Clean up test environment after each function."""

        del self.lats, self.longs, self.alts
        # del self.start_date, self.end_date, self.dates
        return

    @pytest.mark.parametrize("input,output",
                             [([0., 1., 0., 0., 0.], [1.0, 0., 0.]),
                              ([1., 0., 0., 0., 0.], [0., 0., 1.]),
                              ([0., 0., 1., 0., 0.], [0., 1., 0.]),
                              ([1., 0., 0., 0., 90.], [-1., 0., 0.]),
                              ([0., 1., 0., 0., 90.], [0., 0., 1.]),
                              ([0., 0., 1., 0., 90.], [0., 1., 0.]),
                              ([0., 1., 0., 0., 180.], [-1., 0., 0.]),
                              ([1., 0., 0., 0., 180.], [0., 0., -1.]),
                              ([0., 0., 1., 0., 180.], [0., 1., 0.]),
                              ([0., 1., 0., 45., 0.], [1., 0., 0.]),
                              ([1., 0., 0., 45., 0.], [0., -np.cos(np.pi / 4),
                                                       np.cos(np.pi / 4)]),
                              ([0., 0., 1., 45., 0.], [0., np.cos(np.pi / 4),
                                                       np.cos(np.pi / 4)])])
    def test_basic_ecef_to_enu_rotations(self, input, output):
        """Test ECEF to ENU Vector Rotations"""

        ve, vn, vu = OMMBV.vector.ecef_to_enu(*input)

        np.testing.assert_allclose(ve, output[0], atol=1E-9)
        np.testing.assert_allclose(vn, output[1], atol=1E-9)
        np.testing.assert_allclose(vu, output[2], atol=1E-9)
        return

    @pytest.mark.parametrize("input,output", [([1., 0., 0., 0., 0.],
                                               [0., 1., 0.]),
                                              ([0., 0., 1., 0., 0.],
                                               [1., 0., 0.]),
                                              ([0., 1., 0., 0., 0.],
                                               [0., 0., 1.]),
                                              ([1., 0., 0., 0., 90.],
                                               [-1., 0., 0.]),
                                              ([0., 0., 1., 0., 90.],
                                               [0., 1., 0.]),
                                              ([0., 1., 0., 0., 90.],
                                               [0., 0., 1.]),
                                              ([1., 0., 0., 0., 180.],
                                               [0., -1., 0.]),
                                              ([0., 0., 1., 0., 180.],
                                               [-1., 0., 0.]),
                                              ([0., 1., 0., 0., 180.],
                                               [0., 0., 1.])])
    def test_basic_enu_to_ecef_rotations(self, input, output):
        """Test ENU to ECEF rotations"""

        vx, vy, vz = OMMBV.vector.enu_to_ecef(*input)

        np.testing.assert_allclose(vx, output[0], atol=1E-9)
        np.testing.assert_allclose(vy, output[1], atol=1E-9)
        np.testing.assert_allclose(vz, output[2], atol=1E-9)

        return

    def test_ecef_to_enu_back_to_ecef(self):
        """Test ECEF-ENU-ECEF"""
        vx = 0.9
        vy = 0.1
        vz = np.sqrt(1. - vx ** 2 + vy ** 2)

        for lat, lon, alt in zip(self.lats, self.longs, self.alts):
            vxx, vyy, vzz = OMMBV.vector.ecef_to_enu(vx, vy, vz, lat,
                                                     lon)
            vxx, vyy, vzz = OMMBV.vector.enu_to_ecef(vxx, vyy, vzz, lat,
                                                     lon)
            np.testing.assert_allclose(vx, vxx, atol=1E-9)
            np.testing.assert_allclose(vy, vyy, atol=1E-9)
            np.testing.assert_allclose(vz, vzz, atol=1E-9)

        return

    def test_enu_to_ecef_back_to_enu(self):
        """Test ENU-ECEF-ENU"""
        vx = 0.9
        vy = 0.1
        vz = np.sqrt(1. - vx ** 2 + vy ** 2)

        for lat, lon, alt in zip(self.lats, self.longs, self.alts):
            vxx, vyy, vzz = OMMBV.vector.enu_to_ecef(vx, vy, vz, lat,
                                                     lon)
            vxx, vyy, vzz = OMMBV.vector.ecef_to_enu(vxx, vyy, vzz, lat,
                                                     lon)
            np.testing.assert_allclose(vx, vxx, atol=1E-9)
            np.testing.assert_allclose(vy, vyy, atol=1E-9)
            np.testing.assert_allclose(vz, vzz, atol=1E-9)

        return

    def test_igrf_end_to_ecef_back_to_end(self):
        """Check consistency ENU-ECEF and IGRF implementation"""

        vx = 0.9
        vy = 0.1
        vz = np.sqrt(1. - vx ** 2 + vy ** 2)
        vz = -vz

        for lat, lon, alt in zip(self.lats, self.longs, self.alts):
            # Input here is co-latitude, not latitude.
            # Inputs to fortran are in radians.
            vxx, vyy, vzz = igrf.end_vector_to_ecef(vx, vy, vz,
                                                    np.deg2rad(90. - lat),
                                                    np.deg2rad(lon))
            vx2, vy2, vz2 = OMMBV.vector.enu_to_ecef(vx, vy, -vz, lat,
                                                     lon)

            np.testing.assert_allclose(vxx, vx2, atol=1E-9)
            np.testing.assert_allclose(vyy, vy2, atol=1E-9)
            np.testing.assert_allclose(vzz, vz2, atol=1E-9)

            vxx, vyy, vzz = OMMBV.vector.ecef_to_enu(vxx, vyy, vzz, lat,
                                                     lon)
            # Convert upward component back to down
            vzz = -vzz

            # Compare original inputs to outputs
            np.testing.assert_allclose(vx, vxx, atol=1E-9)
            np.testing.assert_allclose(vy, vyy, atol=1E-9)
            np.testing.assert_allclose(vz, vzz, atol=1E-9)

        return
