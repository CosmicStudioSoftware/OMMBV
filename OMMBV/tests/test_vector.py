"""Unit tests for `OMMBV.vector`."""

import numpy as np
import pytest

import OMMBV
from OMMBV import sources
from OMMBV.tests.test_core import gen_data_fixed_alt


class TestVector(object):
    """Unit tests for `OMMBV.vector`."""

    def setup(self):
        """Setup test environment before each function."""

        # Locations to perform tests at
        self.lats, self.longs, self.alts = gen_data_fixed_alt(550.)
        return

    def teardown(self):
        """Clean up test environment after each function."""

        del self.lats, self.longs, self.alts
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
        """Test ECEF to ENU Vector Rotations."""

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
        """Test ENU to ECEF rotations."""

        vx, vy, vz = OMMBV.vector.enu_to_ecef(*input)

        np.testing.assert_allclose(vx, output[0], atol=1E-9)
        np.testing.assert_allclose(vy, output[1], atol=1E-9)
        np.testing.assert_allclose(vz, output[2], atol=1E-9)

        return

    def test_ecef_to_enu_back_to_ecef(self):
        """Test ECEF-ENU-ECEF."""
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
        """Test ENU-ECEF-ENU."""
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
        """Check consistency ENU-ECEF and IGRF implementation."""

        vx = 0.9
        vy = 0.1
        vz = np.sqrt(1. - vx ** 2 + vy ** 2)
        vz = -vz

        for lat, lon, alt in zip(self.lats, self.longs, self.alts):
            # Input here is co-latitude, not latitude.
            # Inputs to fortran are in radians.
            vxx, vyy, vzz = sources.end_vector_to_ecef(vx, vy, vz,
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

    @pytest.mark.parametrize("input", [[22., 36., 42.],
                                       [np.array([1., 2., 3.]),
                                        np.array([4., 5., 6.]),
                                        np.array([7., 8., 9.])]])
    def test_normalize(self, input):
        """Test `OMMBV.vector.normalize` normalizes."""

        x, y, z = OMMBV.vector.normalize(*input)

        # Ensure unit magnitude
        unit_mag = np.sqrt(x**2 + y**2 + z**2)
        np.testing.assert_allclose(unit_mag, 1, atol=1E-9)

        # Check against each component
        mag = np.sqrt(input[0]**2 + input[1]**2 + input[2]**2)
        np.testing.assert_allclose(input[0], mag * x, atol=1E-9)
        np.testing.assert_allclose(input[1], mag * y, atol=1E-9)
        np.testing.assert_allclose(input[2], mag * z, atol=1E-9)

        return

    @pytest.mark.parametrize("input1,input2,output",([[1., 0., 0.],
                                                      [0., 1., 0.],
                                                      [0., 0., 1.]],
                                                     [[np.ones(3), np.zeros(3),
                                                       np.zeros(3)],
                                                      [np.zeros(3), np.ones(3),
                                                       np.zeros(3)],
                                                      [np.zeros(3), np.zeros(3),
                                                       np.ones(3)]]))
    def test_cross_product(self, input1, input2, output):
        """Test `OMMBV.vector.cross_product`."""

        x, y, z = OMMBV.vector.cross_product(*input1, *input2)

        np.testing.assert_allclose(output[0], x, atol=1E-9)
        np.testing.assert_allclose(output[1], y, atol=1E-9)
        np.testing.assert_allclose(output[2], z, atol=1E-9)

        return

    @pytest.mark.parametrize("input_vec,input_basis,output", [([1., 0., 0],
                                                               [0., 1., 0.,
                                                               -1., 0., 0.,
                                                                0., 0., 1.],
                                                               [0., -1., 0.])])
    def test_project_onto_basis(self, input_vec, input_basis, output):
        """Test `OMMBV.vector.project_onto_basis`."""

        x, y, z = OMMBV.vector.project_onto_basis(*input_vec, *input_basis)

        np.testing.assert_allclose(output[0], x, atol=1E-9)
        np.testing.assert_allclose(output[1], y, atol=1E-9)
        np.testing.assert_allclose(output[2], z, atol=1E-9)

        return
