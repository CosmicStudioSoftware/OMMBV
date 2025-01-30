"""Test OMMBV.satellite functions"""
import datetime as dt
import numpy as np
import pandas as pds

import pysat

import OMMBV


class TestSatellite(object):

    def setup(self):
        """Setup test environment before each function."""

        self.inst = pysat.Instrument('pysat', 'testing', num_samples=32)
        return

    def teardown(self):
        """Clean up test environment after each function."""
        del self.inst
        return

    def test_application_add_unit_vectors(self):
        """Check application of unit_vectors to satellite data"""
        self.inst.load(2010, 365)
        self.inst['altitude'] = 550.
        OMMBV.satellite.add_mag_drift_unit_vectors_ecef(self.inst)
        items = ['unit_zon_ecef_x', 'unit_zon_ecef_y', 'unit_zon_ecef_z',
                 'unit_fa_ecef_x', 'unit_fa_ecef_y', 'unit_fa_ecef_z',
                 'unit_mer_ecef_x', 'unit_mer_ecef_y', 'unit_mer_ecef_z']
        for item in items:
            assert item in self.inst.data

        return

    def test_application_add_mag_drifts(self):
        """Check application of unit vectors to drift measurements"""

        self.inst.load(2010, 365)
        self.inst['altitude'] = 550.

        # Create false orientation signal
        self.inst['sc_xhat_x'] = 1.
        self.inst['sc_xhat_y'] = 0.
        self.inst['sc_xhat_z'] = 0.
        self.inst['sc_yhat_x'] = 0.
        self.inst['sc_yhat_y'] = 1.
        self.inst['sc_yhat_z'] = 0.
        self.inst['sc_zhat_x'] = 0.
        self.inst['sc_zhat_y'] = 0.
        self.inst['sc_zhat_z'] = 1.

        # Add vectors and test that vectors were added.
        OMMBV.satellite.add_mag_drift_unit_vectors(self.inst)
        items = ['unit_zon_x', 'unit_zon_y', 'unit_zon_z',
                 'unit_fa_x', 'unit_fa_y', 'unit_fa_z',
                 'unit_mer_x', 'unit_mer_y', 'unit_mer_z']
        for item in items:
            assert item in self.inst.data

        # Check adding drifts now.
        self.inst['iv_x'] = 150.
        self.inst['iv_y'] = 50.
        self.inst['iv_z'] = -50.
        OMMBV.satellite.add_mag_drifts(self.inst)
        items = ['iv_zon', 'iv_fa', 'iv_mer']
        for item in items:
            assert item in self.inst.data

        # Check scaling to footpoints and equator
        self.inst['equ_mer_drifts_scalar'] = 1.
        self.inst['equ_zon_drifts_scalar'] = 1.
        self.inst['north_footpoint_mer_drifts_scalar'] = 1.
        self.inst['north_footpoint_zon_drifts_scalar'] = 1.
        self.inst['south_footpoint_mer_drifts_scalar'] = 1.
        self.inst['south_footpoint_zon_drifts_scalar'] = 1.
        OMMBV.satellite.add_footpoint_and_equatorial_drifts(self.inst)
        items = ['equ_mer_drifts_scalar', 'equ_zon_drifts_scalar',
                 'north_footpoint_mer_drifts_scalar',
                 'north_footpoint_zon_drifts_scalar',
                 'south_footpoint_mer_drifts_scalar',
                 'south_footpoint_zon_drifts_scalar']
        for item in items:
            assert item in self.inst.data

        return

    def test_unit_vector_properties(self):
        """Test basic vector properties along field lines."""

        # Start with a set of locations
        p_long = np.arange(0., 360., 12.)
        p_alt = 0 * p_long + 550.
        p_lats = [5., 10., 15., 20., 25., 30.]

        truthiness = []
        for i, p_lat in enumerate(p_lats):

            date = dt.datetime(2000, 1, 1)
            ecef_x, ecef_y, ecef_z = OMMBV.trans.geocentric_to_ecef(p_lat,
                                                                    p_long,
                                                                    p_alt)

            for j, (x, y, z) in enumerate(zip(ecef_x, ecef_y, ecef_z)):
                # Perform field line traces
                trace_n = OMMBV.trace.field_line_trace(np.array([x, y, z]), date, 1.,
                                                       0., step_size=.5,
                                                       max_steps=1.E6)
                trace_s = OMMBV.trace.field_line_trace(np.array([x, y, z]), date, -1.,
                                                       0., step_size=.5,
                                                       max_steps=1.E6)

                # Combine together, S/C position is first for both
                # Reverse first array and join so plotting makes sense
                trace = np.vstack((trace_n[::-1], trace_s))
                trace = pds.DataFrame(trace, columns=['x', 'y', 'z'])

                # clear stored data
                self.inst.data = pds.DataFrame()
                # downselect, reduce number of points
                trace = trace.loc[::1000, :]

                # compute magnetic field vectors
                # need to provide alt, latitude, and longitude in geodetic coords
                (latitude,
                 longitude,
                 altitude) = OMMBV.trans.ecef_to_geodetic(trace['x'],
                                                          trace['y'],
                                                          trace['z'])

                self.inst[:, 'latitude'] = latitude
                self.inst[:, 'longitude'] = longitude
                self.inst[:, 'altitude'] = altitude

                # Store values for plotting locations for vectors
                self.inst[:, 'x'] = trace['x'].values
                self.inst[:, 'y'] = trace['y'].values
                self.inst[:, 'z'] = trace['z'].values
                idx, = np.where(self.inst['altitude'] > 250.)
                self.inst.data = self.inst[idx, :]

                # also need to provide transformation from ECEF to S/C
                # going to leave that a null transformation so we can plot in ECF
                (self.inst[:, 'sc_xhat_x'],
                 self.inst[:, 'sc_xhat_y'],
                 self.inst[:, 'sc_xhat_z']) = 1., 0., 0.
                (self.inst[:, 'sc_yhat_x'],
                 self.inst[:, 'sc_yhat_y'],
                 self.inst[:, 'sc_yhat_z']) = 0., 1., 0.
                (self.inst[:, 'sc_zhat_x'],
                 self.inst[:, 'sc_zhat_y'],
                 self.inst[:, 'sc_zhat_z']) = 0., 0., 1.
                self.inst.data.index = pysat.utils.time.create_date_range(dt.datetime(2000, 1, 1),
                                                                          dt.datetime(2000, 1, 1) +
                                                                          pds.DateOffset(
                                                                              seconds=len(self.inst.data) - 1),
                                                                          freq='S')
                OMMBV.satellite.add_mag_drift_unit_vectors(self.inst)

                # Check that vectors norm to 1
                assert np.all(np.sqrt(self.inst['unit_zon_x'] ** 2 +
                                      self.inst['unit_zon_y'] ** 2 +
                                      self.inst['unit_zon_z'] ** 2) > 0.999999)
                assert np.all(np.sqrt(self.inst['unit_fa_x'] ** 2 +
                                      self.inst['unit_fa_y'] ** 2 +
                                      self.inst['unit_fa_z'] ** 2) > 0.999999)
                assert np.all(np.sqrt(self.inst['unit_mer_x'] ** 2 +
                                      self.inst['unit_mer_y'] ** 2 +
                                      self.inst['unit_mer_z'] ** 2) > 0.999999)

                # Confirm vectors are mutually orthogonal
                dot1 = self.inst['unit_zon_x'] * self.inst['unit_fa_x'] \
                       + self.inst['unit_zon_y'] * self.inst['unit_fa_y'] \
                       + self.inst['unit_zon_z'] * self.inst['unit_fa_z']
                dot2 = self.inst['unit_zon_x'] * self.inst['unit_mer_x']\
                       + self.inst['unit_zon_y'] * self.inst['unit_mer_y'] \
                       + self.inst['unit_zon_z'] * self.inst['unit_mer_z']
                dot3 = self.inst['unit_fa_x'] * self.inst['unit_mer_x']\
                       + self.inst['unit_fa_y'] * self.inst['unit_mer_y']\
                       + self.inst['unit_fa_z'] * self.inst['unit_mer_z']
                assert np.all(np.abs(dot1) < 1.E-6)
                assert np.all(np.abs(dot2) < 1.E-6)
                assert np.all(np.abs(dot3) < 1.E-6)

                # ensure that zonal vector is generally eastward
                ones = np.ones(len(self.inst.data.index))
                zeros = np.zeros(len(self.inst.data.index))
                ex, ey, ez = OMMBV.vector.enu_to_ecef(ones, zeros, zeros,
                                                      self.inst['latitude'],
                                                      self.inst['longitude'])
                nx, ny, nz = OMMBV.vector.enu_to_ecef(zeros, ones, zeros,
                                                      self.inst['latitude'],
                                                      self.inst['longitude'])
                ux, uy, uz = OMMBV.vector.enu_to_ecef(zeros, zeros, ones,
                                                      self.inst['latitude'],
                                                      self.inst['longitude'])

                dot1 = self.inst['unit_zon_x'] * ex\
                       + self.inst['unit_zon_y'] * ey\
                       + self.inst['unit_zon_z'] * ez
                assert np.all(dot1 > 0.)

                dot1 = self.inst['unit_fa_x'] * nx\
                       + self.inst['unit_fa_y'] * ny\
                       + self.inst['unit_fa_z'] * nz
                assert np.all(dot1 > 0.)

                dot1 = self.inst['unit_mer_x'] * ux\
                       + self.inst['unit_mer_y'] * uy\
                       + self.inst['unit_mer_z'] * uz
                assert np.all(dot1 > 0.)

        assert np.all(truthiness)

        return
