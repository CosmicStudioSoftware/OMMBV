"""Unit tests for `OMMBV.trace.apex_location_info`."""

import datetime as dt
import itertools
import os

import numpy as np
import pandas as pds
import pytest

import pysat

import OMMBV


class TestMaxApexHeight(object):

    def test_apex_heights(self):
        """Check meridional vector along max in apex height gradient"""
        date = dt.datetime(2010, 1, 1)

        delta = 1.

        ecef_x, ecef_y, ecef_z = OMMBV.trans.geodetic_to_ecef([0.], [320.],
                                                              [550.])
        # Get basis vectors
        (zx, zy, zz, _, _, _, mx, my, mz
         ) = OMMBV.calculate_mag_drift_unit_vectors_ecef(ecef_x, ecef_y, ecef_z,
                                                         [date],
                                                         ecef_input=True)

        # Get apex height for step along meridional directions,
        # then around that direction
        (_, _, _, _, _, nominal_max
         ) = OMMBV.trace.apex_location_info(ecef_x + delta * mx,
                                            ecef_y + delta * my,
                                            ecef_z + delta * mz,
                                            [date], ecef_input=True,
                                            return_geodetic=True)

        steps = (np.arange(101) - 50.) * delta / 10000.
        output_max = []
        for step in steps:
            del_x = delta * mx + step * zx
            del_y = delta * my + step * zy
            del_z = delta * mz + step * zz
            del_x, del_y, del_z = OMMBV.vector.normalize(del_x, del_y, del_z)
            (_, _, _, _, _, loop_h
             ) = OMMBV.trace.apex_location_info(ecef_x + del_x, ecef_y + del_y,
                                                ecef_z + del_z, [date],
                                                ecef_input=True,
                                                return_geodetic=True)
            output_max.append(loop_h)

        # Make sure meridional direction is correct
        assert np.all(np.abs(np.max(output_max) - nominal_max) < 1.E-4)
        return


class TestApexAccuracy(object):

    def setup(self):
        """Setup test environment before each function."""

        self.inst = pysat.Instrument('pysat', 'testing', num_samples=100)
        self.inst.yr = 2010.
        self.inst.doy = 1.

        self.date = dt.datetime(2000, 1, 1)
        return

    def teardown(self):
        """Clean up test environment after each function."""

        del self.inst
        return

    @pytest.mark.parametrize("param,vals", [('fine_step_size', [1.E-5, 5.E-6]),
                                            ('fine_max_steps', [5., 10.]),
                                            ('step_size', [100., 50.]),
                                            ('max_steps', [100., 200.])])
    def test_apex_info_accuracy(self, param, vals):
        """Test accuracy of `apex_location_info` for `fine_step_size`."""

        lats, longs, alts = gen_trace_data_fixed_alt(550.)

        out = []
        for val in vals:
            kwargs = {param: val}
            (x, y, z, _, _, apex_height
             ) = OMMBV.trace.apex_location_info(lats, longs, alts,
                                                [self.date]*len(lats),
                                                return_geodetic=True,
                                                **kwargs)
            out.append(pds.DataFrame({'x': x, 'y': y, 'z': z,
                                      'h': apex_height}))

        pt1 = out[0]
        pt2 = out[1]

        for var in pt1.columns:
            np.testing.assert_allclose(pt2[var], pt1[var], rtol=1.E-4)

        np.testing.assert_allclose(pt1['h'], pt2['h'], rtol=1.E-9)

        return


def gen_trace_data_fixed_alt(alt, step_long=80., step_lat=25.):
    """Generate grid data between -50 and 50 degrees latitude.

    Parameters
    ----------
    alt : float
        Fixed altitude to use over longitude latitude grid
    step_long : float (80. degrees)
        Step size used when generating longitudes
    step_lat : float (25. degrees)
        Step size used when generating latitudes

    Returns
    -------
    np.array, np.array, np.array
        Lats, longs, and altitudes.


    """
    # generate test data set
    long_dim = np.arange(0., 361., step_long)
    lat_dim = np.arange(-50., 51., step_lat)

    alt_dim = alt
    locs = np.array(list(itertools.product(long_dim, lat_dim)))
    # pull out lats and longs
    lats = locs[:, 1]
    longs = locs[:, 0]
    alts = longs*0 + alt_dim
    return lats, longs, alts