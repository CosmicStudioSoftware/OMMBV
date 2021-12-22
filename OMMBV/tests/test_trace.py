"""Unit tests for `OMMBV.trace`."""

import datetime as dt
import numpy as np
import pytest

from OMMBV.tests.test_core import gen_data_fixed_alt
import OMMBV.trans


class TestTracing(object):

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