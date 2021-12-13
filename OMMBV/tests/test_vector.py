from nose.tools import assert_almost_equals as asseq
import numpy as np

import OMMBV
from OMMBV import igrf
from OMMBV.tests.test_core import gen_data_fixed_alt


class TestVector():

    def __init__(self):

        # Locations to perform tests at
        self.lats, self.longs, self.alts = gen_data_fixed_alt(550.)

        return

    def test_basic_ecef_to_enu_rotations(self):
        """Test ECEF to ENU Vector Rotations"""
        # vector pointing along ecef y at 0, 0 is east
        ve, vn, vu = OMMBV.vector.ecef_to_enu_vector(0., 1., 0., 0., 0.)
        asseq(ve, 1.0, 9)
        asseq(vn, 0, 9)
        asseq(vu, 0, 9)

        # vector pointing along ecef x at 0, 0 is up
        ve, vn, vu = OMMBV.vector.ecef_to_enu_vector(1., 0., 0., 0., 0.)
        asseq(ve, 0.0, 9)
        asseq(vn, 0.0, 9)
        asseq(vu, 1.0, 9)

        # vector pointing along ecef z at 0, 0 is north
        ve, vn, vu = OMMBV.vector.ecef_to_enu_vector(0., 0., 1., 0., 0.)
        asseq(ve, 0.0, 9)
        asseq(vn, 1.0, 9)
        asseq(vu, 0.0, 9)

        # vector pointing along ecef x at 0, 90 long is west
        ve, vn, vu = OMMBV.vector.ecef_to_enu_vector(1., 0., 0., 0., 90.)
        asseq(ve, -1.0, 9)
        asseq(vn, 0.0, 9)
        asseq(vu, 0.0, 9)

        # vector pointing along ecef y at 0, 90 long is up
        ve, vn, vu = OMMBV.vector.ecef_to_enu_vector(0., 1., 0., 0., 90.)
        asseq(ve, 0.0, 9)
        asseq(vn, 0.0, 9)
        asseq(vu, 1.0, 9)

        # vector pointing along ecef z at 0, 90 long is north
        ve, vn, vu = OMMBV.vector.ecef_to_enu_vector(0., 0., 1., 0., 90.)
        asseq(ve, 0.0, 9)
        asseq(vn, 1.0, 9)
        asseq(vu, 0.0, 9)

        # vector pointing along ecef y at 0, 180 is west
        ve, vn, vu = OMMBV.vector.ecef_to_enu_vector(0., 1., 0., 0., 180.)
        asseq(ve, -1.0, 9)
        asseq(vn, 0.0, 9)
        asseq(vu, 0.0, 9)

        # vector pointing along ecef x at 0, 180 is down
        ve, vn, vu = OMMBV.vector.ecef_to_enu_vector(1., 0., 0., 0., 180.)
        asseq(ve, 0.0, 9)
        asseq(vn, 0.0, 9)
        asseq(vu, -1.0, 9)

        # vector pointing along ecef z at 0, 0 is north
        ve, vn, vu = OMMBV.vector.ecef_to_enu_vector(0., 0., 1., 0., 180.)
        asseq(ve, 0.0, 9)
        asseq(vn, 1.0, 9)
        asseq(vu, 0.0, 9)

        ve, vn, vu = OMMBV.vector.ecef_to_enu_vector(0., 1., 0., 45., 0.)
        asseq(ve, 1.0, 9)
        asseq(vn, 0, 9)
        asseq(vu, 0, 9)

        # vector pointing along ecef x at 0, 0 is south/up
        ve, vn, vu = OMMBV.vector.ecef_to_enu_vector(1., 0., 0., 45., 0.)
        asseq(ve, 0.0, 9)
        asseq(vn, -np.cos(np.pi / 4), 9)
        asseq(vu, np.cos(np.pi / 4), 9)

        # vector pointing along ecef z at 45, 0 is north/up
        ve, vn, vu = OMMBV.vector.ecef_to_enu_vector(0., 0., 1., 45., 0.)
        asseq(ve, 0.0, 9)
        asseq(vn, np.cos(np.pi / 4), 9)
        asseq(vu, np.cos(np.pi / 4), 9)

        return

    def test_basic_enu_to_ecef_rotations(self):
        """Test ENU to ECEF rotations"""

        # vector pointing east at 0, 0 is along y
        vx, vy, vz = OMMBV.vector.enu_to_ecef_vector(1., 0., 0., 0., 0.)
        asseq(vx, 0.0, 9)
        asseq(vy, 1.0, 9)
        asseq(vz, 0.0, 9)

        # vector pointing up at 0, 0 is along x
        vx, vy, vz = OMMBV.vector.enu_to_ecef_vector(0., 0., 1., 0., 0.)
        asseq(vx, 1.0, 9)
        asseq(vy, 0.0, 9)
        asseq(vz, 0.0, 9)

        # vector pointing north at 0, 0 is along z
        vx, vy, vz = OMMBV.vector.enu_to_ecef_vector(0., 1., 0., 0., 0.)
        asseq(vx, 0.0, 9)
        asseq(vy, 0.0, 9)
        asseq(vz, 1.0, 9)

        # east vector at 0, 90 long points along -x
        vx, vy, vz = OMMBV.vector.enu_to_ecef_vector(1., 0., 0., 0., 90.)
        asseq(vx, -1.0, 9)
        asseq(vy, 0.0, 9)
        asseq(vz, 0.0, 9)

        # vector pointing up at 0, 90 is along y
        vx, vy, vz = OMMBV.vector.enu_to_ecef_vector(0., 0., 1., 0., 90.)
        asseq(vx, 0.0, 9)
        asseq(vy, 1.0, 9)
        asseq(vz, 0.0, 9)

        # vector pointing north at 0, 90 is along z
        vx, vy, vz = OMMBV.vector.enu_to_ecef_vector(0., 1., 0., 0., 90.)
        asseq(vx, 0.0, 9)
        asseq(vy, 0.0, 9)
        asseq(vz, 1.0, 9)

        # vector pointing east at 0, 0 is along y
        vx, vy, vz = OMMBV.vector.enu_to_ecef_vector(1., 0., 0., 0., 180.)
        asseq(vx, 0.0, 9)
        asseq(vy, -1.0, 9)
        asseq(vz, 0.0, 9)

        # vector pointing up at 0, 180 is along -x
        vx, vy, vz = OMMBV.vector.enu_to_ecef_vector(0., 0., 1., 0., 180.)
        asseq(vx, -1.0, 9)
        asseq(vy, 0.0, 9)
        asseq(vz, 0.0, 9)

        # vector pointing north at 0, 180 is along z
        vx, vy, vz = OMMBV.vector.enu_to_ecef_vector(0., 1., 0., 0., 180.)
        asseq(vx, 0.0, 9)
        asseq(vy, 0.0, 9)
        asseq(vz, 1.0, 9)

        return

    def test_ecef_to_enu_back_to_ecef(self):
        """Test ECEF-ENU-ECEF"""
        vx = 0.9
        vy = 0.1
        vz = np.sqrt(1. - vx ** 2 + vy ** 2)

        for lat, lon, alt in zip(self.lats, self.longs, self.alts):
            vxx, vyy, vzz = OMMBV.vector.ecef_to_enu_vector(vx, vy, vz, lat,
                                                            lon)
            vxx, vyy, vzz = OMMBV.vector.enu_to_ecef_vector(vxx, vyy, vzz, lat,
                                                            lon)
            asseq(vx, vxx, 9)
            asseq(vy, vyy, 9)
            asseq(vz, vzz, 9)

        return

    def test_enu_to_ecef_back_to_enu(self):
        """Test ENU-ECEF-ENU"""
        vx = 0.9
        vy = 0.1
        vz = np.sqrt(1. - vx ** 2 + vy ** 2)

        for lat, lon, alt in zip(self.lats, self.longs, self.alts):
            vxx, vyy, vzz = OMMBV.vector.enu_to_ecef_vector(vx, vy, vz, lat,
                                                            lon)
            vxx, vyy, vzz = OMMBV.vector.ecef_to_enu_vector(vxx, vyy, vzz, lat,
                                                            lon)
            asseq(vx, vxx, 9)
            asseq(vy, vyy, 9)
            asseq(vz, vzz, 9)

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
            vx2, vy2, vz2 = OMMBV.vector.enu_to_ecef_vector(vx, vy, -vz, lat,
                                                            lon)

            asseq(vxx, vx2, 9)
            asseq(vyy, vy2, 9)
            asseq(vzz, vz2, 9)

            vxx, vyy, vzz = OMMBV.vector.ecef_to_enu_vector(vxx, vyy, vzz, lat,
                                                            lon)
            # Convert upward component back to down
            vzz = -vzz
            # Compare original inputs to outputs
            asseq(vx, vxx, 9)
            asseq(vy, vyy, 9)
            asseq(vz, vzz, 9)

        return
