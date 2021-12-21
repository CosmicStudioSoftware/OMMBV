"""Unit tests for OMMBV core vector basis functions."""

import datetime as dt
import functools
import itertools
import numpy as np
import pandas as pds
import pytest

import OMMBV
import OMMBV.trace
import OMMBV.trans
import OMMBV.vector

import pysat


def mstp(field, surf):
    """Return dict with appropriate `mag_fcn` and `step_fcn`.

    Parameters
    ----------
    field : int
        0 - Dipole,
        1 - Dipole + Linear Quadrupole, +Z,
        2 - Dipole + Linear Quadrupole, +X
        3 - Dipole + Normal Quadrupole, Equator
        4 - Dipole + Normal Quadrupole, Equator
    surf : int
        0 - Geodetic Eath
        1 - Geocentric Earth

    Returns
    -------
    dict
        `mag_fcn` and `step_fcn` keys with functions.

    """

    out = {'mag_fcn': functools.partial(OMMBV.sources.poles, field),
           'step_fcn': functools.partial(OMMBV.sources.gen_step, field, surf)}

    return out


# Methods to generate data sets used by testing routines.
def gen_data_fixed_alt(alt, min_lat=-90., max_lat=90., step_lat=5.,
                       step_long=20.):
    """Generate grid data between `min_lat` and `max_lat` degrees latitude.

    Parameters
    ----------
    alt : float
        Fixed altitude to use over longitude latitude grid

    Returns
    -------
    np.array, np.array, np.array
        Lats, longs, and altitudes.

    Note
    ----
      Maximum latitude is 89.999 degrees

    """
    # Generate test data set
    long_dim = np.arange(0., 361., step_long)
    lat_dim = np.arange(min_lat, max_lat + step_lat, step_lat)

    idx, = np.where(lat_dim >= 90.)
    lat_dim[idx] = 89.999
    idx, = np.where(lat_dim <= -90.)
    lat_dim[idx] = -89.999

    alt_dim = alt
    locs = np.array(list(itertools.product(long_dim, lat_dim)))

    # Pull out lats and longs
    lats = locs[:, 1]
    longs = locs[:, 0]
    alts = longs * 0 + alt_dim
    return lats, longs, alts


def gen_plot_grid_fixed_alt(alt):
    """Generate dimensional data between -50 and 50 degrees latitude.

    Parameters
    ----------
    alt : float
        Fixed altitude to use over longitude latitude grid

    Returns
    -------
    np.array, np.array, np.array
        Lats, longs, and altitudes.

    Note
    ----
    Output is different than routines above.

    """
    # generate test data set
    long_dim = np.arange(0., 360., 1. * 80)
    lat_dim = np.arange(-50., 50.1, 0.25 * 80)

    alt_dim = np.array([alt])
    return lat_dim, long_dim, alt_dim


class TestUnitVectors(object):

    def setup(self):
        """Setup test environment before each function."""
        self.inst = pysat.Instrument('pysat', 'testing')
        self.inst.yr = 2010.
        self.inst.doy = 1.

        self.date = dt.datetime(2000, 1, 1)

        self.map_labels = ['d_zon_x', 'd_zon_y', 'd_zon_z',
                           'd_mer_x', 'd_mer_y', 'd_mer_z',
                           'e_zon_x', 'e_zon_y', 'e_zon_z',
                           'e_mer_x', 'e_mer_y', 'e_mer_z',
                           'd_fa_x', 'd_fa_y', 'd_fa_z',
                           'e_fa_x', 'e_fa_y', 'e_fa_z']

        self.lats, self.longs, self.alts = gen_plot_grid_fixed_alt(550.)
        self.alts = [self.alts[0]] * len(self.longs)

        return

    def teardown(self):
        """Clean up test environment after each function."""

        del self.inst, self.date, self.map_labels
        return

    def test_vectors_at_pole(self):
        """Ensure np.nan returned at magnetic pole for scalars and vectors."""
        out = OMMBV.calculate_mag_drift_unit_vectors_ecef([80.97], [250.],
                                                          [550.],
                                                          [self.date],
                                                          full_output=True,
                                                          pole_tol=1.E-4)
        # Confirm vectors are np.nan
        assert np.all(np.isnan(out[0:3]))
        assert np.all(np.isnan(out[6:9]))

        for item in self.map_labels:
            assert np.isnan(out[-1][item])

        return

    @pytest.mark.parametrize("".join(["param,param_vals,unit_tol,apex_tol,"
                                      "map_tol,orth_tol"]),
                             [('step_size', [1., 0.5], 1.E-4, 1.E-4, 1.E-4,
                               2.E-5),
                              ('dstep_size', [1., 0.5], 1.E-4, 1.E-4, 1.E-4,
                               2.E-5),
                              ('tol', [1.E-5], 1.E-5, 1.E-4, 1.E-5, 1.E-5),
                              ('tol_zonal_apex', [1.E-5], 1.E-4, 1.E-5, 1.E-5,
                               1.E-5),
                              ('dipole_spherical', [mstp(0, 1)], 1.E-4, 1.E-4,
                               1.E-4, 2.E-5),
                              ('dipole_geodetic', [mstp(0, 0)], 1.E-4, 1.E-4,
                               1.E-4, 2.E-5),
                              ('dipole_quad_z', [mstp(1, 0)], 1.E-4, 1.E-4,
                               1.E-4, 2.E-5),
                              ('dipole_quad_x', [mstp(2, 0)], 1.E-4, 1.E-4,
                               1.E-4, 2.E-5),
                              ('dipole_quad_norm', [mstp(3, 0)], 1.E-4, 1.E-4,
                               1.E-4, 2.E-5),
                              ('dipole_oct_norm',  [mstp(4, 0)], 1.E-4, 1.E-4,
                               1.E-4, 2.E-5)])
    def test_basis_performance(self, param, param_vals, unit_tol, apex_tol,
                               map_tol, orth_tol):
        """Test performance of vector basis.

        Parameters
        ----------
        param : str
            Keyword argument label to be assigned.
        param_vals : list or dict
            If a list, then `param` is iteratively assigned a value
            from `param_vals`. If the length is 2, the
            difference in outputs for each `param_val` is taken.
            If a dict, then the dict is passed along as keyword
            arguments and `param` is ignored.
        unit_tol : float
            Tolerance requirement for unit vectors
        apex_tol : float
            Tolerance requirement for the apex zonal gradient
        map_tol : float
            Tolerance requirement for mapping values, D and E vectors.
        orth_tol : float
            Tolerance requirement for orthogonality, determined as
            normalized difference between the D and D' vectors.

        """
        cmduv = OMMBV.calculate_mag_drift_unit_vectors_ecef

        if param == 'dipole_spherical':
            OMMBV.trans.configure_geocentric_earth(True)
        else:
            OMMBV.trans.configure_geocentric_earth(False)

        dvl = {'d_zon2_x': 'd_zon_x', 'd_zon2_y': 'd_zon_y',
               'd_zon2_z': 'd_zon_z', 'd_mer2_x': 'd_mer_x',
               'd_mer2_y': 'd_mer_y',
               'd_mer2_z': 'd_mer_z'}
        unit_vector_labels = ['zx', 'zy', 'zz',
                              'fx', 'fy', 'fz',
                              'mx', 'my', 'mz']

        out = []
        dates = [self.date] * len(self.longs)
        for lat in self.lats:
            lats = [lat] * len(self.longs)
            for val in param_vals:
                if isinstance(val, dict):
                    kwargs = val
                else:
                    kwargs = {param: val}

                (zx, zy, zz,
                 fx, fy, fz,
                 mx, my, mz,
                 infod) = cmduv(lats, self.longs, self.alts, dates,
                                full_output=True, include_debug=True,
                                **kwargs)
                ddict = {'zx': zx, 'zy': zy, 'zz': zz,
                         'fx': fx, 'fy': fy, 'fz': fz,
                         'mx': mx, 'my': my, 'mz': mz}

                # Check internally calculated achieved tolerance
                np.testing.assert_array_less(infod['diff_zonal_vec'],
                                             infod['diff_zonal_vec'] * 0
                                             + unit_tol)
                np.testing.assert_array_less(infod['diff_mer_vec'],
                                             infod['diff_mer_vec'] * 0
                                             + unit_tol)

                # Check internally calculated apex height gradient
                np.testing.assert_array_less(infod['grad_zonal_apex'],
                                             infod['grad_zonal_apex'] * 0
                                             + apex_tol)
                
                # Ensure generated basis is the same for both D vector
                # calculations.
                for key in dvl.keys():
                    # Message if the comparison fails
                    estr = ''.join(['Failing for latitude ', str(lat)])

                    # Need a different check for values close to zero or not.
                    idx, = np.where(np.abs(infod[key]) >= 1.E-10)
                    np.testing.assert_allclose(infod[key][idx],
                                               infod[dvl[key]][idx],
                                               rtol=orth_tol, err_msg=estr)
                    idx, = np.where(np.abs(infod[key]) < 1.E-10)
                    np.testing.assert_allclose(infod[key][idx],
                                               infod[dvl[key]][idx],
                                               atol=orth_tol, err_msg=estr)

                # Collect D, E vector data
                for key in self.map_labels:
                    ddict[key] = infod[key]
                out.append(pds.DataFrame(ddict))

            if len(param_vals) > 1:
                pt1 = out[0]
                pt2 = out[1]

                # Check unit vectors
                for var in unit_vector_labels:
                    np.testing.assert_allclose(pt2[var], pt1[var],
                                               atol=unit_tol)
                # Check D, E vectors
                for var in self.map_labels:
                    np.testing.assert_allclose(pt2[var], pt1[var],
                                               atol=map_tol)

        return

    def test_simple_geomagnetic_basis_interface(self):
        """Ensure simple geomagnetic basis interface runs"""

        for i, p_lat in enumerate(self.lats):
            out_d = OMMBV.calculate_geomagnetic_basis([p_lat] * len(self.longs),
                                                      self.longs, self.alts,
                                                      [self.date]
                                                      * len(self.longs))

        assert True
        return

    @pytest.mark.parametrize('direction', ['zonal', 'meridional'])
    def test_step_along_mag_unit_vector_sensitivity(self, direction, tol=1.E-5):
        """Characterize apex location uncertainty of neighboring field lines.

        Parameters
        ----------
        direction : str
            Step along 'meridional' or 'zonal' directions.

        """

        # Create memory for method locations from method output
        x = np.zeros((len(self.lats), len(self.longs)))
        y = x.copy()
        z = x.copy()
        h = x.copy()

        # Second set of outputs
        x2 = np.zeros((len(self.lats), len(self.longs)))
        y2 = x2.copy()
        z2 = x2.copy()
        h2 = x.copy()

        dates = [self.date] * len(self.longs)

        for i, p_lat in enumerate(self.lats):
            (in_x, in_y, in_z
             ) = OMMBV.trans.geodetic_to_ecef([p_lat] * len(self.longs),
                                              self.longs, self.alts)

            (x[i, :], y[i, :], z[i, :]
             ) = OMMBV.step_along_mag_unit_vector(in_x, in_y, in_z,
                                                  dates,
                                                  direction=direction,
                                                  num_steps=2,
                                                  step_size=1. / 2.)
            # Second run
            (x2[i, :], y2[i, :], z2[i, :]
             ) = OMMBV.step_along_mag_unit_vector(in_x, in_y, in_z,
                                                  dates,
                                                  direction=direction,
                                                  num_steps=1,
                                                  step_size=1.)

        for i, p_lat in enumerate(self.lats):
            # Convert all locations to geodetic coordinates
            tlat, tlon, talt = OMMBV.trans.ecef_to_geodetic(x[i, :], y[i, :],
                                                            z[i, :])

            # Get apex location
            (x[i, :], y[i, :], z[i, :], _, _, h[i, :]
             ) = OMMBV.trace.apex_location_info(tlat, tlon, talt, dates,
                                                return_geodetic=True)

            # Repeat process for second set of positions.
            tlat, tlon, talt = OMMBV.trans.ecef_to_geodetic(x2[i, :], y2[i, :],
                                                            z2[i, :])
            (x2[i, :], y2[i, :], z2[i, :], _, _,h2[i, :]
             ) = OMMBV.trace.apex_location_info(tlat, tlon, talt, dates,
                                                return_geodetic=True)

        # Test within expected tolerance.
        for item1, item2 in zip([x, y, z, h], [x2, y2, z2, h2]):
            idx, idy,  = np.where(np.abs(item1) >= 1.E-10)
            np.testing.assert_allclose(item1[idx, idy], item2[idx, idy],
                                       rtol=tol)
            idx, idy, = np.where(np.abs(item1) < 1.E-10)
            np.testing.assert_allclose(item1[idx, idy], item2[idx, idy],
                                       atol=tol)

        return

    @pytest.mark.parametrize("".join(["kwargs"]), [mstp(0, 0), {}])
    def test_geomag_efield_scalars(self, kwargs):
        """Test electric field and drift mapping values."""

        data = {}
        for i, p_lat in enumerate(self.lats):
            templ = [p_lat] * len(self.longs)
            tempd = [self.date] * len(self.longs)
            scalars = OMMBV.scalars_for_mapping_ion_drifts(templ,
                                                           self.longs,
                                                           self.alts,
                                                           tempd,
                                                           **kwargs)
            for scalar in scalars:
                if scalar not in data:
                    data[scalar] = np.full((len(self.lats), len(self.longs)),
                                           np.nan)

                data[scalar][i, :] = scalars[scalar]

        assert len(scalars.keys()) == 12

        for scalar in scalars:
            assert np.all(np.isfinite(scalars[scalar]))

        return
