"""Unit tests for `OMMBV.trace`."""

import datetime as dt
import functools
import numpy as np
import pytest

from OMMBV.tests.test_core import gen_data_fixed_alt
import OMMBV.trans
import OMMBV.trace as trace


class TestTracing(object):
    """Test `OMMBV.trace` functions."""

    def setup(self):
        """Setup test environment before each function."""

        self.date = dt.datetime(2020, 1, 1)

        # Locations to perform tests at
        self.lats, self.longs, self.alts = gen_data_fixed_alt(550.)

        (self.ecef_xs,
         self.ecef_ys,
         self.ecef_zs) = OMMBV.trans.geodetic_to_ecef(self.lats, self.longs,
                                                      self.alts)
        return

    def teardown(self):
        """Clean up test environment after each function."""

        del self.lats, self.longs, self.alts
        return

    @pytest.mark.parametrize('ref_height', (120., 240.))
    def test_full_field_trace_termination_altitude(self, ref_height):
        """Ensure `full_field_trace` stops at terminating altitude."""

        for ecef_x, ecef_y, ecef_z in zip(self.ecef_xs, self.ecef_ys,
                                          self.ecef_zs):
            # Perform full field line trace
            trace = OMMBV.trace.full_field_line([ecef_x, ecef_y, ecef_z],
                                                self.date, ref_height)

            # Convert to geodetic locations
            gx, gy, gz = OMMBV.trans.ecef_to_geodetic(trace[:, 0],
                                                      trace[:, 1],
                                                      trace[:, 2])

            # Test that terminating altitude is close to target altitude
            assert np.all(gz >= ref_height - 1.E-3)

        return

    def test_bad_steps_inputs_full_field_trace(self):
        """Test for expected failure with bad input."""

        with pytest.raises(ValueError) as err:
            OMMBV.trace.full_field_line([self.ecef_xs[0], self.ecef_ys[0],
                                         self.ecef_zs[0]], self.date,
                                        120., max_steps=1000,
                                        steps=np.arange(10))

        err_msg = 'Length of steps must be `max_steps`+1.'
        assert str(err.value.args[0]).find(err_msg) >= 0

        return

    def test_nan_input_full_field_trace(self):
        """Test for nan output for nan input."""

        trace = OMMBV.trace.full_field_line([np.nan, np.nan, np.nan], self.date,
                                            120.)

        assert np.all(np.isnan(trace))
        assert len(trace) == 1

        return

    def test_trace_along_magnetic_field(self):
        """Test traced field line along magnetic field."""

        for ecef_x, ecef_y, ecef_z in zip(self.ecef_xs, self.ecef_ys,
                                          self.ecef_zs):
            # Perform full field line trace
            trace = OMMBV.trace.full_field_line([ecef_x, ecef_y, ecef_z],
                                                self.date, 120.,
                                                min_check_flag=True)

            # Construct ECEF unit vector connecting each point of field line
            # trace to the next.
            tx, ty, tz = OMMBV.vector.normalize(np.diff(trace[:, 0]),
                                                np.diff(trace[:, 1]),
                                                np.diff(trace[:, 2]))

            # Downselect positions where movement along field line is positive
            diff_mag = np.sqrt(np.diff(trace[:, 0])**2
                               + np.diff(trace[:, 1])**2
                               + np.diff(trace[:, 2])**2)
            idx, = np.where(diff_mag > 1.)

            # Magnetic field unit vector
            dates = [self.date]*len(tx[idx])
            bx, by, bz, bm = OMMBV.trace.magnetic_vector(trace[:-1, 0][idx],
                                                         trace[:-1, 1][idx],
                                                         trace[:-1, 2][idx],
                                                         dates,
                                                         normalize=True)

            np.testing.assert_allclose(tx[idx], bx, atol=1.E-1)
            np.testing.assert_allclose(ty[idx], by, atol=1.E-1)
            np.testing.assert_allclose(tz[idx], bz, atol=1.E-1)

        return

    def test_field_line_trace_max_recursion(self):
        """Test recursion limit code in `field_line_trace`."""

        # Configure a pure dipole field, spherical earth. Pick a
        # location very close to Southern magnetic pole, a very small
        # step size, and then try to integrate towards northern hemisphere.
        # It will not make it.
        step_fcn = functools.partial(OMMBV.sources.gen_step, 0, 1)
        out = trace.field_line_trace([1., 0., -7000.],
                                     self.date, 1, 10., step_size=1.E-5,
                                     step_fcn=step_fcn)

        assert len(out) == 1
        assert len(out[0]) == 3
        assert np.all(np.isnan(out[0]))

        return

    @pytest.mark.parametrize('height', (120., 240.))
    def test_footpoint_target_altitude(self, height):
        """Test `footpoint_location_info` terminating altitude."""

        dates = [self.date] * len(self.alts)
        north, south = trace.footpoint_location_info(self.lats, self.longs,
                                                     self.alts, dates,
                                                     height=height,
                                                     return_geodetic=True)
        # Latitude check
        assert np.all(np.abs(north[:, 0]) <= 90.)
        assert np.all(np.abs(south[:, 0]) <= 90.)

        # Longitude check
        assert np.all((np.abs(north[:, 1]) <= 360.)
                      & (np.abs(north[:, 1]) >= 0.))
        assert np.all((np.abs(south[:, 1]) <= 360.)
                      & (np.abs(south[:, 1]) >= 0.))

        # Altitude check
        np.testing.assert_allclose(north[:, 2], height, atol=1.E-3)
        np.testing.assert_allclose(south[:, 2], height, atol=1.E-3)

        return

    def test_failsafe_geographic_pole_apex_location(self):
        """Confirm np.nan output when `apex_location_info` fails at pole."""
        out = OMMBV.trace.apex_location_info([90.], [0.], [10.],
                                             [self.date], return_geodetic=True)
        assert np.all(np.isnan(out))
        return

    def test_failsafe_low_altitude_apex_location(self):
        """Confirm np.nan when `apex_location_info` fails at low altitude."""
        out = OMMBV.trace.apex_location_info([90.], [0.], [0.],
                                             [self.date], return_geodetic=True,
                                             validate_input=True)
        assert np.all(np.isnan(out))
        return
