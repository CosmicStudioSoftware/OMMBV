"""Unit tests for OMMBV core vector basis functions."""

import datetime as dt
import functools
import itertools
import numpy as np
import os
import pandas as pds
import pytest

import OMMBV as OMMBV
import OMMBV.trace
import OMMBV.trans
import OMMBV.vector

import pysat


# Methods to generate data sets used by testing routines.
def gen_data_fixed_alt(alt):
    """Generate grid data between -90 and 90 degrees latitude, almost.

    Parameters
    ----------
    alt : float
        Fixed altitude to use over longitude latitude grid

    Returns
    -------
    np.array, np.array, np.array
        Lats, longs, and altitudes.

    Notes
    -----
      Maximum latitude is 89.999 degrees

    """
    # generate test data set
    on_travis = os.environ.get('ONTRAVIS') == 'True'
    if on_travis:
        # reduced resolution on the test server
        long_dim = np.arange(0., 361., 180.)
        lat_dim = np.arange(-90., 91., 50.)
    else:
        long_dim = np.arange(0., 361., 20.)
        lat_dim = np.arange(-90., 91., 5.)

    idx, = np.where(lat_dim == 90.)
    lat_dim[idx] = 89.999
    idx, = np.where(lat_dim == -90.)
    lat_dim[idx] = -89.999

    alt_dim = alt
    locs = np.array(list(itertools.product(long_dim, lat_dim)))

    # Pull out lats and longs
    lats = locs[:, 1]
    longs = locs[:, 0]
    alts = longs*0 + alt_dim
    return lats, longs, alts


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
    on_travis = os.environ.get('ONTRAVIS') == 'True'
    if on_travis:
        # reduced resolution on the test server
        long_dim = np.arange(0., 361., 180.)
        lat_dim = np.arange(-50., 51., 50.)
    else:
        long_dim = np.arange(0., 361., step_long)
        lat_dim = np.arange(-50., 51., step_lat)

    alt_dim = alt
    locs = np.array(list(itertools.product(long_dim, lat_dim)))
    # pull out lats and longs
    lats = locs[:, 1]
    longs = locs[:, 0]
    alts = longs*0 + alt_dim
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
    import os
    # generate test data set
    on_travis = os.environ.get('ONTRAVIS') == 'True'
    if on_travis:
        # reduced resolution on the test server
        long_dim = np.arange(0., 360., 180.)
        lat_dim = np.arange(-50., 51., 50.)
    else:
        long_dim = np.arange(0., 360., 1. * 4)
        lat_dim = np.arange(-50., 50.1, 0.25 * 4)

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

    @pytest.mark.parametrize("param,vals", [('step_size', [1., 0.5], 1.E-4,
                                             1.E-4, 1.E-5),
                                            ('dstep_size', [1., 0.5], 1.E-4,
                                             1.E-4, 1.E-5),
                                            ('tol', [1.E-5], 1.E-5, 1.E-4,
                                             1.E-5),
                                            ('tol_zonal_apex', [1.E-5],
                                             1.E-4, 1.E-5, 1.E-5)
                                            ])
    def test_basis_sensitivity(self, param, param_vals, unit_tol, apex_tol,
                               map_tol, orth_tol):
        """Test sensitivity of vector basis as param varied."""
        p_lats, p_longs, p_alts = gen_plot_grid_fixed_alt(550.)
        # data returned are the locations along each direction
        # the full range of points obtained by iterating over all
        # recasting alts into a more convenient form for later calculation
        p_alts = [p_alts[0]]*len(p_longs)

        cmduv = OMMBV.calculate_mag_drift_unit_vectors_ecef

        dvl = {'d_zon2_x': 'd_zon_x', 'd_zon2_y': 'd_zon_y',
               'd_zon2_z': 'd_zon_z', 'd_mer2_x': 'd_mer_x',
               'd_mer2_y': 'd_mer_y',
               'd_mer2_z': 'd_mer_z'}
        unit_vector_labels = ['zx', 'zy', 'zz',
                              'fx', 'fy', 'fz',
                              'mx', 'my', 'mz']

        out = []
        dates = [self.date]*len(p_longs)
        for lat in p_lats:
            lats = [lat] * len(p_longs)
            for val in param_vals:
                kwargs = {param: val}

                (zx, zy, zz,
                 fx, fy, fz,
                 mx, my, mz,
                 infod) = cmduv(lats, p_longs, p_alts, dates,
                                full_output=True, include_debug=True,
                                **kwargs)
                ddict = {'zx': zx, 'zy': zy, 'zz': zz,
                         'fx': fx, 'fy': fy, 'fz': fz,
                         'mx': mx, 'my': my, 'mz': mz}

                # Check internally calculated achieved tolerance
                assert infod['diff_zonal_vec'] < unit_tol
                assert infod['diff_mer_vec'] < unit_tol

                # Check internally calculated apex height gradient
                assert infod['grad_zonal_apex'] < apex_tol
                
                # Ensure generated basis is the same for both covariant
                # and contra-variant forms.
                for key in dvl.keys():
                    np.testing.assert_allclose(ddict[key], ddict[dvl[key]],
                                               rtol=orth_tol)

                # Collect D, E vector data
                for key in self.map_labels:
                    ddict[key] = infod[key]
                out.append(pds.DataFrame(ddict))

            if len(param_vals) > 1:
                pt1 = out[0]
                pt2 = out[1]

                # Check unit vectors
                for var in unit_vector_labels:
                    assert np.testing.assert_allclose(pt2[var], pt1[var],
                                                      atol=unit_tol)
                # Check D, E vectors
                for var in self.map_labels:
                    assert np.testing.assert_allclose(pt2[var], pt1[var],
                                                      rtol=map_tol)

        return


    # # Edge steps currently not code effective. Skipping.
    # def test_unit_vector_component_plots_edge_steps(self):
    #     """Check precision D of vectors as edge_steps increased"""
    #     import matplotlib.pyplot as plt
    #
    #     p_lats, p_longs, p_alts = gen_plot_grid_fixed_alt(550.)
    #     # data returned are the locations along each direction
    #     # the full range of points obtained by iterating over all
    #     # recasting alts into a more convenient form for later calculation
    #     p_alts = [p_alts[0]]*len(p_longs)
    #
    #     d_zvx = np.zeros((len(p_lats), len(p_longs) + 1))
    #     d_zvy = d_zvx.copy()
    #     d_zvz = d_zvx.copy()
    #     d2_zvx = np.zeros((len(p_lats), len(p_longs) + 1))
    #     d2_zvy = d_zvx.copy()
    #     d2_zvz = d_zvx.copy()
    #     d_mx = d_zvx.copy()
    #     d_my = d_zvx.copy()
    #     d_mz = d_zvx.copy()
    #     d_fax = d_zvx.copy()
    #     d_fay = d_zvx.copy()
    #     d_faz = d_zvx.copy()
    #     d2_mx = d_zvx.copy()
    #     d2_my = d_zvx.copy()
    #     d2_mz = d_zvx.copy()
    #
    #     date = dt.datetime(2000, 1, 1)
    #     # set up multi
    #     if self.dc is not None:
    #         targets = itertools.cycle(dc.ids)
    #         pending = []
    #         for i, p_lat in enumerate(p_lats):
    #             # iterate through target cyclicly and run commands
    #             print(i, p_lat)
    #             dview.targets = next(targets)
    #             pending.append(
    #                 dview.apply_async(OMMBV.calculate_mag_drift_unit_vectors_ecef,
    #                                   [p_lat]*len(p_longs), p_longs,
    #                                   p_alts, [date]*len(p_longs), full_output=True,
    #                                   include_debug=True, edge_steps=2))
    #         for i, p_lat in enumerate(p_lats):
    #             print('collecting ', i, p_lat)
    #             # collect output
    #             tzx, tzy, tzz, tbx, tby, tbz, tmx, tmy, tmz, infod = pending.pop(0).get()
    #
    #             # collect outputs on E and D vectors
    #             dzx, dzy, dzz = infod['d_zon_x'], infod['d_zon_y'], infod['d_zon_z']
    #             dfx, dfy, dfz = infod['d_fa_x'], infod['d_fa_y'], infod['d_fa_z']
    #             dmx, dmy, dmz = infod['d_mer_x'], infod['d_mer_y'], infod['d_mer_z']
    #             d_zvx[i, :-1], d_zvy[i, :-1], d_zvz[i, :-1] = OMMBV.ecef_to_enu(dzx, dzy, dzz,
    #                                                                                    [p_lat]*len(p_longs),
    #                                                                                    p_longs)
    #             dzx, dzy, dzz = infod['d_zon2_x'], infod['d_zon2_y'], infod['d_zon2_z']
    #             d2_zvx[i, :-1], d2_zvy[i, :-1], d2_zvz[i, :-1] = OMMBV.ecef_to_enu(dzx, dzy, dzz,
    #                                                                                       [p_lat]*len(p_longs),
    #                                                                                       p_longs)
    #             d_fax[i, :-1], d_fay[i, :-1], d_faz[i, :-1] = OMMBV.ecef_to_enu(dfx, dfy, dfz,
    #                                                                                    [p_lat]*len(p_longs),
    #                                                                                    p_longs)
    #             d_mx[i, :-1], d_my[i, :-1], d_mz[i, :-1] = OMMBV.ecef_to_enu(dmx, dmy, dmz,
    #                                                                                 [p_lat]*len(p_longs),
    #                                                                                 p_longs)
    #             dmx, dmy, dmz = infod['d_mer2_x'], infod['d_mer2_y'], infod['d_mer2_z']
    #             d2_mx[i, :-1], d2_my[i, :-1], d2_mz[i, :-1] = OMMBV.ecef_to_enu(dmx, dmy, dmz,
    #                                                                                    [p_lat]*len(p_longs),
    #                                                                                    p_longs)
    #
    #
    #     else:
    #         for i, p_lat in enumerate(p_lats):
    #             print (i, p_lat)
    #             tzx, tzy, tzz, tbx, tby, tbz, tmx, tmy, tmz, infod = OMMBV.calculate_mag_drift_unit_vectors_ecef(
    #                                                                                     [p_lat]*len(p_longs), p_longs,
    #                                                                                     p_alts, [date]*len(p_longs),
    #                                                                                     full_output=True,
    #                                                                                     include_debug=True,
    #                                                                                     edge_steps=2)
    #
    #             # collect outputs on E and D vectors
    #             dzx, dzy, dzz = infod['d_zon_x'], infod['d_zon_y'], infod['d_zon_z']
    #             dfx, dfy, dfz = infod['d_fa_x'], infod['d_fa_y'], infod['d_fa_z']
    #             dmx, dmy, dmz = infod['d_mer_x'], infod['d_mer_y'], infod['d_mer_z']
    #             d_zvx[i, :-1], d_zvy[i, :-1], d_zvz[i, :-1] = OMMBV.ecef_to_enu(dzx, dzy, dzz,
    #                                                                                    [p_lat]*len(p_longs), p_longs)
    #             dzx, dzy, dzz = infod['d_zon2_x'], infod['d_zon2_y'], infod['d_zon2_z']
    #             d2_zvx[i, :-1], d2_zvy[i, :-1], d2_zvz[i, :-1] = OMMBV.ecef_to_enu(dzx, dzy, dzz,
    #                                                                                       [p_lat]*len(p_longs),
    #                                                                                       p_longs)
    #             d_fax[i, :-1], d_fay[i, :-1], d_faz[i, :-1] = OMMBV.ecef_to_enu(dfx, dfy, dfz,
    #                                                                                    [p_lat]*len(p_longs), p_longs)
    #             d_mx[i, :-1], d_my[i, :-1], d_mz[i, :-1] = OMMBV.ecef_to_enu(dmx, dmy, dmz,
    #                                                                                 [p_lat]*len(p_longs), p_longs)
    #             dmx, dmy, dmz = infod['d_mer2_x'], infod['d_mer2_y'], infod['d_mer2_z']
    #             d2_mx[i, :-1], d2_my[i, :-1], d2_mz[i, :-1] = OMMBV.ecef_to_enu(dmx, dmy, dmz,
    #                                                                                    [p_lat]*len(p_longs), p_longs)
    #
    #     # account for periodicity
    #
    #     d_zvx[:, -1] = d_zvx[:, 0]
    #     d_zvy[:, -1] = d_zvy[:, 0]
    #     d_zvz[:, -1] = d_zvz[:, 0]
    #     d2_zvx[:, -1] = d2_zvx[:, 0]
    #     d2_zvy[:, -1] = d2_zvy[:, 0]
    #     d2_zvz[:, -1] = d2_zvz[:, 0]
    #     d_fax[:, -1] = d_fax[:, 0]
    #     d_fay[:, -1] = d_fay[:, 0]
    #     d_faz[:, -1] = d_faz[:, 0]
    #     d_mx[:, -1] = d_mx[:, 0]
    #     d_my[:, -1] = d_my[:, 0]
    #     d_mz[:, -1] = d_mz[:, 0]
    #     d2_mx[:, -1] = d2_mx[:, 0]
    #     d2_my[:, -1] = d2_my[:, 0]
    #     d2_mz[:, -1] = d2_mz[:, 0]
    #
    #     ytickarr = np.array([0, 0.25, 0.5, 0.75, 1])*(len(p_lats) - 1)
    #     xtickarr = np.array([0, 0.2, 0.4, 0.6, 0.8, 1])*len(p_longs)
    #
    #     try:
    #         fig = plt.figure()
    #         dmag = np.sqrt(d_mx ** 2 + d_my ** 2 + d_mz ** 2)
    #         dmag2 = np.sqrt(d2_mx ** 2 + d2_my ** 2 + d2_mz ** 2)
    #         plt.imshow(np.log10(np.abs(dmag - dmag2) / dmag), origin='lower')
    #         plt.colorbar()
    #         plt.yticks(ytickarr, ['-50', '-25', '0', '25', '50'])
    #         plt.xticks(xtickarr, ['0', '72', '144', '216', '288', '360'])
    #         plt.title('Log D Meridional Vector Normalized Difference')
    #         plt.xlabel('Geodetic Longitude (Degrees)')
    #         plt.ylabel('Geodetic Latitude (Degrees)')
    #         plt.tight_layout()
    #         plt.savefig('d_diff_mer_norm_edgesteps.pdf')
    #         plt.close()
    #
    #         fig = plt.figure()
    #         dmag = np.sqrt(d_zvx ** 2 + d_zvy ** 2 + d_zvz ** 2)
    #         dmag2 = np.sqrt(d2_zvx ** 2 + d2_zvy ** 2 + d2_zvz ** 2)
    #         plt.imshow(np.log10(np.abs(dmag2 - dmag) / dmag), origin='lower')
    #         plt.colorbar()
    #         plt.yticks(ytickarr, ['-50', '-25', '0', '25', '50'])
    #         plt.xticks(xtickarr, ['0', '72', '144', '216', '288', '360'])
    #         plt.title('Log D Zonal Vector Normalized Difference')
    #         plt.xlabel('Geodetic Longitude (Degrees)')
    #         plt.ylabel('Geodetic Latitude (Degrees)')
    #         plt.tight_layout()
    #         plt.savefig('d_diff_zon_norm_edegesteps.pdf')
    #         plt.close()
    #     except:
    #         pass

    def test_simple_geomagnetic_basis_interface(self):
        """Ensure simple geomagnetic basis interface runs"""

        p_lats, p_longs, p_alts = gen_plot_grid_fixed_alt(550.)
        # data returned are the locations along each direction
        # the full range of points obtained by iterating over all
        # recasting alts into a more convenient form for later calculation
        p_alts = [p_alts[0]] * len(p_longs)
        date = dt.datetime(2000, 1, 1)

        if self.dc is not None:
            targets = itertools.cycle(dc.ids)
            pending = []
            for i, p_lat in enumerate(p_lats):
                # iterate through target cyclicly and run commands
                print(i, p_lat)
                dview.targets = next(targets)
                pending.append(
                    dview.apply_async(OMMBV.calculate_geomagnetic_basis,
                                      [p_lat]*len(p_longs), p_longs,
                                      p_alts, [date]*len(p_longs)))

            for i, p_lat in enumerate(p_lats):
                print ('collecting ', i, p_lat)
                # collect output from first run
                out_d = pending.pop(0).get()
        else:
            for i, p_lat in enumerate(p_lats):
                print (i, p_lat)
                out_d = OMMBV.calculate_geomagnetic_basis([p_lat]*len(p_longs), p_longs,
                                                          p_alts, [date]*len(p_longs))

    def test_unit_vector_component_stepsize_sensitivity_plots(self):
        """Produce spatial plots of unit vector output sensitivity at the default step_size"""
        import matplotlib.pyplot as plt

        p_lats, p_longs, p_alts = gen_plot_grid_fixed_alt(550.)
        # data returned are the locations along each direction
        # the full range of points obtained by iterating over all
        # recasting alts into a more convenient form for later calculation
        p_alts = [p_alts[0]]*len(p_longs)

        # zonal vector components
        # +1 on length of longitude array supports repeating first element
        # shows nice periodicity on the plots
        zvx = np.zeros((len(p_lats), len(p_longs) + 1))
        zvy = zvx.copy()
        zvz = zvx.copy()
        #  meridional vector components
        mx = zvx.copy()
        my = zvx.copy()
        mz = zvx.copy()
        # field aligned, along B
        bx = zvx.copy()
        by = zvx.copy()
        bz = zvx.copy()
        date = dt.datetime(2000, 1, 1)
        # set up multi
        if self.dc is not None:
            targets = itertools.cycle(dc.ids)
            pending = []
            for i, p_lat in enumerate(p_lats):
                # iterate through target cyclicly and run commands
                print(i, p_lat)
                dview.targets = next(targets)
                pending.append(
                    dview.apply_async(OMMBV.calculate_mag_drift_unit_vectors_ecef,
                                      [p_lat]*len(p_longs), p_longs,
                                      p_alts, [date]*len(p_longs),
                                      step_size=1.))
                pending.append(
                    dview.apply_async(OMMBV.calculate_mag_drift_unit_vectors_ecef,
                                      [p_lat]*len(p_longs), p_longs,
                                      p_alts, [date]*len(p_longs),
                                      step_size=2.))

            for i, p_lat in enumerate(p_lats):
                print('collecting ', i, p_lat)
                # collect output from first run
                tzx, tzy, tzz, tbx, tby, tbz, tmx, tmy, tmz = pending.pop(0).get()
                zvx[i, :-1], zvy[i, :-1], zvz[i, :-1] = OMMBV.vector.ecef_to_enu(tzx, tzy, tzz,
                                                                                 [p_lat] * len(p_longs),
                                                                                 p_longs)
                bx[i, :-1], by[i, :-1], bz[i, :-1] = OMMBV.vector.ecef_to_enu(tbx, tby, tbz,
                                                                              [p_lat] * len(p_longs),
                                                                              p_longs)
                mx[i, :-1], my[i, :-1], mz[i, :-1] = OMMBV.vector.ecef_to_enu(tmx, tmy, tmz,
                                                                              [p_lat] * len(p_longs),
                                                                              p_longs)
                # collect output from second run
                tzx, tzy, tzz, tbx, tby, tbz, tmx, tmy, tmz = pending.pop(0).get()
                _a, _b, _c = OMMBV.vector.ecef_to_enu(tzx, tzy, tzz, [p_lat] * len(p_longs), p_longs)
                # take difference with first run
                zvx[i, :-1] = (zvx[i, :-1] - _a)  # /zvx[i,:-1]
                zvy[i, :-1] = (zvy[i, :-1] - _b)  # /zvy[i,:-1]
                zvz[i, :-1] = (zvz[i, :-1] - _c)  # /zvz[i,:-1]

                _a, _b, _c = OMMBV.vector.ecef_to_enu(tbx, tby, tbz, [p_lat] * len(p_longs), p_longs)
                # take difference with first run
                bx[i, :-1] = (bx[i, :-1] - _a)  # /bx[i,:-1]
                by[i, :-1] = (by[i, :-1] - _b)  # /by[i,:-1]
                bz[i, :-1] = (bz[i, :-1] - _c)  # /bz[i,:-1]

                _a, _b, _c = OMMBV.vector.ecef_to_enu(tmx, tmy, tmz, [p_lat] * len(p_longs), p_longs)
                # take difference with first run
                mx[i, :-1] = (mx[i, :-1] - _a)  # /mx[i,:-1]
                my[i, :-1] = (my[i, :-1] - _b)  # /my[i,:-1]
                mz[i, :-1] = (mz[i, :-1] - _c)  # /mz[i,:-1]

        else:
            for i, p_lat in enumerate(p_lats):
                print(i, p_lat)
                tzx, tzy, tzz, tbx, tby, tbz, tmx, tmy, tmz = OMMBV.calculate_mag_drift_unit_vectors_ecef(
                                                                        [p_lat]*len(p_longs), p_longs,
                                                                        p_alts, [date]*len(p_longs),
                                                                        step_size=1.)
                zvx[i, :-1], zvy[i, :-1], zvz[i, :-1] = OMMBV.vector.ecef_to_enu(tzx, tzy, tzz,
                                                                                 [p_lat] * len(p_longs),
                                                                                 p_longs)
                bx[i, :-1], by[i, :-1], bz[i, :-1] = OMMBV.vector.ecef_to_enu(tbx, tby, tbz,
                                                                              [p_lat] * len(p_longs),
                                                                              p_longs)
                mx[i, :-1], my[i, :-1], mz[i, :-1] = OMMBV.vector.ecef_to_enu(tmx, tmy, tmz,
                                                                              [p_lat] * len(p_longs),
                                                                              p_longs)

                # second run
                tzx, tzy, tzz, tbx, tby, tbz, tmx, tmy, tmz = OMMBV.calculate_mag_drift_unit_vectors_ecef(
                    [p_lat]*len(p_longs), p_longs,
                    p_alts, [date]*len(p_longs),
                    step_size=2.)
                _a, _b, _c = OMMBV.vector.ecef_to_enu(tzx, tzy, tzz, [p_lat] * len(p_longs), p_longs)
                # take difference with first run
                zvx[i, :-1] = (zvx[i, :-1] - _a)  # /zvx[i,:-1]
                zvy[i, :-1] = (zvy[i, :-1] - _b)  # /zvy[i,:-1]
                zvz[i, :-1] = (zvz[i, :-1] - _c)  # /zvz[i,:-1]

                _a, _b, _c = OMMBV.vector.ecef_to_enu(tbx, tby, tbz, [p_lat] * len(p_longs), p_longs)
                # take difference with first run
                bx[i, :-1] = (bx[i, :-1] - _a)  # /bx[i,:-1]
                by[i, :-1] = (by[i, :-1] - _b)  # /by[i,:-1]
                bz[i, :-1] = (bz[i, :-1] - _c)  # /bz[i,:-1]

                _a, _b, _c = OMMBV.vector.ecef_to_enu(tmx, tmy, tmz, [p_lat] * len(p_longs), p_longs)
                # take difference with first run
                mx[i, :-1] = (mx[i, :-1] - _a)  # /mx[i,:-1]
                my[i, :-1] = (my[i, :-1] - _b)  # /my[i,:-1]
                mz[i, :-1] = (mz[i, :-1] - _c)  # /mz[i,:-1]

        # account for periodicity
        zvx[:, -1] = zvx[:, 0]
        zvy[:, -1] = zvy[:, 0]
        zvz[:, -1] = zvz[:, 0]
        bx[:, -1] = bx[:, 0]
        by[:, -1] = by[:, 0]
        bz[:, -1] = bz[:, 0]
        mx[:, -1] = mx[:, 0]
        my[:, -1] = my[:, 0]
        mz[:, -1] = mz[:, 0]

        # feedbback on locations with highest error
        idx = np.argmax(mz)
        idx, idy = np.unravel_index(idx, np.shape(mz))
        print('****** ****** ******')
        print('maxixum location lat, long', p_lats[idx], p_longs[idy])

        ytickarr = np.array([0, 0.25, 0.5, 0.75, 1])*(len(p_lats) - 1)
        xtickarr = np.array([0, 0.2, 0.4, 0.6, 0.8, 1])*len(p_longs)

        try:
            fig = plt.figure()
            plt.imshow(np.log10(np.abs(zvx)), origin='lower')
            plt.colorbar()
            plt.yticks(ytickarr, ['-50', '-25', '0', '25', '50'])
            plt.xticks(xtickarr, ['0', '72', '144', '216', '288', '360'])
            plt.title('Log Zonal Unit Vector Difference - Eastward')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.tight_layout()
            plt.savefig('zonal_east_diff.pdf')
            plt.close()

            fig = plt.figure()
            plt.imshow(np.log10(np.abs(zvy)), origin='lower')
            plt.colorbar()
            plt.yticks(ytickarr, ['-50', '-25', '0', '25', '50'])
            plt.xticks(xtickarr, ['0', '72', '144', '216', '288', '360'])
            plt.title('Log Zonal Unit Vector Difference - Northward')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.tight_layout()
            plt.savefig('zonal_north_diff.pdf')
            plt.close()

            fig = plt.figure()
            plt.imshow(np.log10(np.abs(zvz)), origin='lower')
            plt.colorbar()
            plt.yticks(ytickarr, ['-50', '-25', '0', '25', '50'])
            plt.xticks(xtickarr, ['0', '72', '144', '216', '288', '360'])
            plt.title('Log Zonal Unit Vector Difference - Upward')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.tight_layout()
            plt.savefig('zonal_up_diff.pdf')
            plt.close()

            fig = plt.figure()
            plt.imshow(np.log10(np.abs(bx)), origin='lower')
            plt.colorbar()
            plt.yticks(ytickarr, ['-50', '-25', '0', '25', '50'])
            plt.xticks(xtickarr, ['0', '72', '144', '216', '288', '360'])
            plt.title('Log Field Aligned Unit Vector Difference - Eastward')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.tight_layout()
            plt.savefig('fa_east_diff.pdf')
            plt.close()

            fig = plt.figure()
            plt.imshow(np.log10(np.abs(by)), origin='lower')
            plt.colorbar()
            plt.yticks(ytickarr, ['-50', '-25', '0', '25', '50'])
            plt.xticks(xtickarr, ['0', '72', '144', '216', '288', '360'])
            plt.title('Log Field Aligned Unit Vector Difference - Northward')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.tight_layout()
            plt.savefig('fa_north_diff.pdf')
            plt.close()

            fig = plt.figure()
            plt.imshow(np.log10(np.abs(bz)), origin='lower')
            plt.colorbar()
            plt.yticks(ytickarr, ['-50', '-25', '0', '25', '50'])
            plt.xticks(xtickarr, ['0', '72', '144', '216', '288', '360'])
            plt.title('Log Field Aligned Unit Vector Difference - Upward')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.tight_layout()
            plt.savefig('fa_up_diff.pdf')
            plt.close()

            fig = plt.figure()
            plt.imshow(np.log10(np.abs(mx)), origin='lower')
            plt.colorbar()
            plt.yticks(ytickarr, ['-50', '-25', '0', '25', '50'])
            plt.xticks(xtickarr, ['0', '72', '144', '216', '288', '360'])
            plt.title('Log Meridional Unit Vector Difference - Eastward')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.tight_layout()
            plt.savefig('mer_east_diff.pdf')
            plt.close()

            fig = plt.figure()
            plt.imshow(np.log10(np.abs(my)), origin='lower')
            plt.colorbar()
            plt.yticks(ytickarr, ['-50', '-25', '0', '25', '50'])
            plt.xticks(xtickarr, ['0', '72', '144', '216', '288', '360'])
            plt.title('Log Meridional Unit Vector Difference - Northward')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.tight_layout()
            plt.savefig('mer_north_diff.pdf')
            plt.close()

            fig = plt.figure()
            plt.imshow(np.log10(np.abs(mz)), origin='lower')
            plt.colorbar()
            plt.yticks(ytickarr, ['-50', '-25', '0', '25', '50'])
            plt.xticks(xtickarr, ['0', '72', '144', '216', '288', '360'])
            plt.title('Log Meridional Unit Vector Difference - Upward')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.tight_layout()
            plt.savefig('mer_up_diff.pdf')
            plt.close()

            # calculate mean and standard deviation and then plot those
            plt.figure()
            # plt.errorbar(p_longs, np.log10(np.nanmedian(np.abs(zvx[:, :-1]), axis=0)),
            #              yerr=np.abs(np.log10(np.nanstd(zvx[:, :-1], axis=0))), label='East')
            # plt.errorbar(p_longs, np.log10(np.nanmedian(np.abs(zvy[:, :-1]), axis=0)),
            #              yerr=np.abs(np.log10(np.nanstd(zvy[:, :-1], axis=0))), label='North')
            # plt.errorbar(p_longs, np.log10(np.nanmedian(np.abs(zvz[:, :-1]), axis=0)),
            #              yerr=np.abs(np.log10(np.nanstd(zvz[:, :-1], axis=0))), label='Up')
            plt.plot(p_longs, np.log10(np.nanmedian(np.abs(zvx[:, :-1]), axis=0)),
                     label='East')
            plt.plot(p_longs, np.log10(np.nanmedian(np.abs(zvy[:, :-1]), axis=0)),
                     label='North')
            plt.plot(p_longs, np.log10(np.nanmedian(np.abs(zvz[:, :-1]), axis=0)),
                     label='Up')

            plt.xlabel('Longitude (Degrees)')
            plt.ylabel('Log Change in Zonal Vector')
            plt.title("Sensitivity of Zonal Unit Vector")
            plt.legend()
            plt.tight_layout()
            plt.tight_layout()
            plt.savefig('zonal_diff_v_longitude.pdf')
            plt.close()

            # calculate mean and standard deviation and then plot those
            plt.figure()
            # plt.errorbar(p_longs, np.log10(np.nanmedian(np.abs(mx[:, :-1]), axis=0)),
            #              yerr=np.abs(np.log10(np.nanstd(mx[:, :-1], axis=0))), label='East')
            # plt.errorbar(p_longs, np.log10(np.nanmedian(np.abs(my[:, :-1]), axis=0)),
            #              yerr=np.abs(np.log10(np.nanstd(my[:, :-1], axis=0))), label='North')
            # plt.errorbar(p_longs, np.log10(np.nanmedian(np.abs(mz[:, :-1]), axis=0)),
            #              yerr=np.abs(np.log10(np.nanstd(mz[:, :-1], axis=0))), label='Up')
            plt.plot(p_longs, np.log10(np.nanmedian(np.abs(mx[:, :-1]), axis=0)),
                     label='East')
            plt.plot(p_longs, np.log10(np.nanmedian(np.abs(my[:, :-1]), axis=0)),
                     label='North')
            plt.plot(p_longs, np.log10(np.nanmedian(np.abs(mz[:, :-1]), axis=0)),
                     label='Up')
            plt.xlabel('Longitude (Degrees)')
            plt.ylabel('Log Change in Meridional Vector')
            plt.title("Sensitivity of Meridional Unit Vector")
            plt.legend()
            plt.tight_layout()
            plt.tight_layout()
            plt.savefig('mer_diff_v_longitude.pdf')
            plt.close()

        except:
            print('Skipping plots due to error.')

    # Should be moved to test_heritage
    def step_along_mag_unit_vector_sensitivity_plots(self, direction=None):
        """Characterize the uncertainty associated with obtaining the apex location of neighboring field lines"""
        import matplotlib.pyplot as plt

        p_lats, p_longs, p_alts = gen_plot_grid_fixed_alt(550.)
        # data returned are the locations along each direction
        # the full range of points obtained by iterating over all
        # recasting alts into a more convenient form for later calculation
        p_alts = [p_alts[0]]*len(p_longs)

        # create memory for method
        # locations from method output, in ECEF
        # want positions with one setting on method under test, then another
        # +1 on length of longitude array supports repeating first element
        # shows nice periodicity on the plots
        x = np.zeros((len(p_lats), len(p_longs) + 1))
        y = x.copy();
        z = x.copy();
        h = x.copy()
        # second set of outputs
        x2 = np.zeros((len(p_lats), len(p_longs) + 1))
        y2 = x2.copy();
        z2 = x2.copy();
        h2 = x.copy()

        date = dt.datetime(2000, 1, 1)
        dates = [date]*len(p_longs)
        # set up multi
        if self.dc is not None:
            targets = itertools.cycle(dc.ids)
            pending = []
            for i, p_lat in enumerate(p_lats):
                print (i, p_lat)
                # iterate through target cyclicly and run commands

                dview.targets = next(targets)
                # inputs are ECEF locations
                in_x, in_y, in_z = OMMBV.trans.geodetic_to_ecef([p_lat] * len(p_longs), p_longs, p_alts)
                pending.append(dview.apply_async(OMMBV.step_along_mag_unit_vector,
                                                 in_x, in_y, in_z, dates,
                                                 direction=direction,
                                                 num_steps=2, step_size=25. / 2.))
                pending.append(dview.apply_async(OMMBV.step_along_mag_unit_vector,
                                                 in_x, in_y, in_z, dates,
                                                 direction=direction,
                                                 num_steps=1, step_size=25. / 1.))
            for i, p_lat in enumerate(p_lats):
                print ('collecting ', i, p_lat)
                # collect output from first run
                x[i, :-1], y[i, :-1], z[i, :-1] = pending.pop(0).get()
                # collect output from second run
                x2[i, :-1], y2[i, :-1], z2[i, :-1] = pending.pop(0).get()
            # trace each location to its apex
            # this provides an increase in the spatial difference that results
            # from innacurate movement between field lines from step_along_mag_unit_vector
            for i, p_lat in enumerate(p_lats):
                dview.targets = next(targets)
                # convert all locations to geodetic coordinates
                tlat, tlon, talt = OMMBV.ecef_to_geodetic(x[i, :-1], y[i, :-1], z[i, :-1])
                pending.append(dview.apply_async(OMMBV.trace.apex_location_info,
                                                 tlat, tlon, talt, dates,
                                                 return_geodetic=True))
                # convert all locations to geodetic coordinates
                tlat, tlon, talt = OMMBV.ecef_to_geodetic(x2[i, :-1], y2[i, :-1], z2[i, :-1])
                pending.append(dview.apply_async(OMMBV.trace.apex_location_info,
                                                 tlat, tlon, talt, dates,
                                                 return_geodetic=True))
            for i, p_lat in enumerate(p_lats):
                x[i, :-1], y[i, :-1], z[i, :-1], _, _, h[i, :-1] = pending.pop(0).get()
                x2[i, :-1], y2[i, :-1], z2[i, :-1], _, _, h2[i, :-1] = pending.pop(0).get()
            normx = x.copy()
            normy = y.copy()
            normz = z.copy()
            normh = h.copy()
            # take difference in locations
            x = x - x2
            y = y - y2
            z = z - z2
            h = h - h2

        else:
            for i, p_lat in enumerate(p_lats):
                in_x, in_y, in_z = OMMBV.trans.geodetic_to_ecef([p_lat] * len(p_longs), p_longs, p_alts)

                x[i, :-1], y[i, :-1], z[i, :-1] = OMMBV.step_along_mag_unit_vector(in_x, in_y, in_z, dates,
                                                                                  direction=direction,
                                                                                  num_steps=2, step_size=25. / 2.)
                # second run
                x2[i, :-1], y2[i, :-1], z2[i, :-1] = OMMBV.step_along_mag_unit_vector(in_x, in_y, in_z, dates,
                                                                                     direction=direction,
                                                                                     num_steps=1, step_size=25. / 1.)

            for i, p_lat in enumerate(p_lats):
                # convert all locations to geodetic coordinates
                tlat, tlon, talt = OMMBV.ecef_to_geodetic(x[i, :-1], y[i, :-1], z[i, :-1])
                x[i, :-1], y[i, :-1], z[i, :-1], _, _, h[i, :-1] = OMMBV.trace.apex_location_info(tlat, tlon, talt, dates,
                                                                                                  return_geodetic=True)
                # convert all locations to geodetic coordinates
                tlat, tlon, talt = OMMBV.ecef_to_geodetic(x2[i, :-1], y2[i, :-1], z2[i, :-1])
                x2[i, :-1], y2[i, :-1], z2[i, :-1], _, _, h2[i, :-1] = OMMBV.trace.apex_location_info(tlat, tlon, talt, dates,
                                                                                                      return_geodetic=True)
            # take difference in locations
            normx = x.copy()
            normy = y.copy()
            normz = z.copy()
            normh = np.abs(h)

            x = x - x2
            y = y - y2
            z = z - z2
            h = h - h2

        # account for periodicity
        x[:, -1] = x[:, 0]
        y[:, -1] = y[:, 0]
        z[:, -1] = z[:, 0]
        h[:, -1] = h[:, 0]
        normx[:, -1] = normx[:, 0]
        normy[:, -1] = normy[:, 0]
        normz[:, -1] = normz[:, 0]
        normh[:, -1] = normh[:, 0]
        # plot tick locations and labels
        ytickarr = np.array([0, 0.25, 0.5, 0.75, 1])*(len(p_lats) - 1)
        xtickarr = np.array([0, 0.2, 0.4, 0.6, 0.8, 1])*len(p_longs)

        try:
            fig = plt.figure()
            plt.imshow(np.log10(np.abs(x)), origin='lower')
            plt.colorbar()
            plt.yticks(ytickarr, ['-50', '-25', '0', '25', '50'])
            plt.xticks(xtickarr, ['0', '72', '144', '216', '288', '360'])
            plt.title('Log Difference in Apex Position (X - km) After Stepping')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.tight_layout()
            plt.savefig(direction + '_step_diff_apex_height_x.pdf')
            plt.close()

            fig = plt.figure()
            plt.imshow(np.log10(np.abs(y)), origin='lower')
            plt.colorbar()
            plt.yticks(ytickarr, ['-50', '-25', '0', '25', '50'])
            plt.xticks(xtickarr, ['0', '72', '144', '216', '288', '360'])
            plt.title('Log Difference in Apex Position (Y - km) After Stepping')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.tight_layout()
            plt.savefig(direction + '_step_diff_apex_height_y.pdf')
            plt.close()

            fig = plt.figure()
            plt.imshow(np.log10(np.abs(z)), origin='lower')
            plt.colorbar()
            plt.yticks(ytickarr, ['-50', '-25', '0', '25', '50'])
            plt.xticks(xtickarr, ['0', '72', '144', '216', '288', '360'])
            plt.title('Log Difference in Apex Position (Z - km) After Stepping')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.tight_layout()
            plt.savefig(direction + '_step_diff_apex_height_z.pdf')
            plt.close()

            fig = plt.figure()
            plt.imshow(np.log10(np.sqrt(x**2 + y**2 + z**2)), origin='lower')
            plt.colorbar()
            plt.yticks(ytickarr, ['-50', '-25', '0', '25', '50'])
            plt.xticks(xtickarr, ['0', '72', '144', '216', '288', '360'])
            plt.title('Log Difference in Apex Position After Stepping')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.tight_layout()
            plt.savefig(direction + '_step_diff_apex_height_r.pdf')
            plt.close()

            # calculate mean and standard deviation and then plot those
            fig = plt.figure()
            yerrx = np.nanstd(np.log10(x[:, :-1]), axis=0)
            yerry = np.nanstd(np.log10(y[:, :-1]), axis=0)
            yerrz = np.nanstd(np.log10(z[:, :-1]), axis=0)
            vals = np.log10(np.nanmedian(np.abs(x[:, :-1]), axis=0))
            # plt.errorbar(p_longs, vals,
            #              yerr=yerrx - vals, label='x')
            plt.plot(p_longs, vals, label='x')
            vals = np.log10(np.nanmedian(np.abs(y[:, :-1]), axis=0))
            # plt.errorbar(p_longs, vals,
            #              yerr=yerry - vals, label='y')
            plt.plot(p_longs, vals, label='y')
            vals = np.log10(np.nanmedian(np.abs(z[:, :-1]), axis=0))
            # plt.errorbar(p_longs, vals,
            #              yerr=yerrz - vals, label='z')
            plt.plot(p_longs, vals, label='z')
            plt.xlabel('Longitude (Degrees)')
            plt.ylabel('Change in ECEF (km)')
            plt.title('Log Median Difference in Apex Position')
            plt.legend()
            plt.tight_layout()
            plt.tight_layout()
            plt.savefig(direction + '_step_diff_v_longitude.pdf')
            plt.close()

            fig = plt.figure()
            plt.imshow(np.log10(np.abs(h / normh)), origin='lower')
            plt.colorbar()
            plt.yticks(ytickarr, ['-50', '-25', '0', '25', '50'])
            plt.xticks(xtickarr, ['0', '72', '144', '216', '288', '360'])
            plt.title('Log Normalized Difference in Apex Height (h) After Stepping')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.tight_layout()
            plt.savefig(direction + '_normal_step_diff_apex_height_h.pdf')
            plt.close()
        except:
            pass

    def test_step_sensitivity(self):
        f = functools.partial(self.step_along_mag_unit_vector_sensitivity_plots, direction='zonal')
        yield (f,)
        f = functools.partial(self.step_along_mag_unit_vector_sensitivity_plots, direction='meridional')
        yield (f,)

    def test_geomag_efield_scalars_plots(self):
        """Produce summary plots of the electric field and drift mapping values """
        import matplotlib.pyplot as plt

        p_lats, p_longs, p_alts = gen_plot_grid_fixed_alt(550.)
        # data returned are the locations along each direction
        # the full range of points obtained by iterating over all
        # recasting alts into a more convenient form for later calculation
        p_alts = [p_alts[0]]*len(p_longs)

        north_zonal = np.zeros((len(p_lats), len(p_longs) + 1))
        north_mer = north_zonal.copy()
        south_zonal = north_zonal.copy()
        south_mer = north_zonal.copy()
        eq_zonal = north_zonal.copy()
        eq_mer = north_zonal.copy()

        north_zonald = np.zeros((len(p_lats), len(p_longs) + 1))
        north_merd = north_zonal.copy()
        south_zonald = north_zonal.copy()
        south_merd = north_zonal.copy()
        eq_zonald = north_zonal.copy()
        eq_merd = north_zonal.copy()

        date = dt.datetime(2000, 1, 1)
        # set up multi
        if self.dc is not None:
            targets = itertools.cycle(dc.ids)
            pending = []
            for i, p_lat in enumerate(p_lats):
                # iterate through target cyclicly and run commands
                print(i, p_lat)
                dview.targets = next(targets)
                pending.append(dview.apply_async(OMMBV.scalars_for_mapping_ion_drifts,
                                                 [p_lat]*len(p_longs), p_longs,
                                                 p_alts, [date]*len(p_longs)))
            for i, p_lat in enumerate(p_lats):
                print('collecting ', i, p_lat)
                # collect output
                scalars = pending.pop(0).get()
                north_zonal[i, :-1] = scalars['north_mer_fields_scalar']
                north_mer[i, :-1] = scalars['north_zon_fields_scalar']
                south_zonal[i, :-1] = scalars['south_mer_fields_scalar']
                south_mer[i, :-1] = scalars['south_zon_fields_scalar']
                eq_zonal[i, :-1] = scalars['equator_mer_fields_scalar']
                eq_mer[i, :-1] = scalars['equator_zon_fields_scalar']

                north_zonald[i, :-1] = scalars['north_zon_drifts_scalar']
                north_merd[i, :-1] = scalars['north_mer_drifts_scalar']
                south_zonald[i, :-1] = scalars['south_zon_drifts_scalar']
                south_merd[i, :-1] = scalars['south_mer_drifts_scalar']
                eq_zonald[i, :-1] = scalars['equator_zon_drifts_scalar']
                eq_merd[i, :-1] = scalars['equator_mer_drifts_scalar']
        else:
            for i, p_lat in enumerate(p_lats):
                print(i, p_lat)
                scalars = OMMBV.scalars_for_mapping_ion_drifts([p_lat]*len(p_longs),
                                                               p_longs,
                                                               p_alts,
                                                               [date]*len(p_longs))
                north_zonal[i, :-1] = scalars['north_mer_fields_scalar']
                north_mer[i, :-1] = scalars['north_zon_fields_scalar']
                south_zonal[i, :-1] = scalars['south_mer_fields_scalar']
                south_mer[i, :-1] = scalars['south_zon_fields_scalar']
                eq_zonal[i, :-1] = scalars['equator_mer_fields_scalar']
                eq_mer[i, :-1] = scalars['equator_zon_fields_scalar']

                north_zonald[i, :-1] = scalars['north_zon_drifts_scalar']
                north_merd[i, :-1] = scalars['north_mer_drifts_scalar']
                south_zonald[i, :-1] = scalars['south_zon_drifts_scalar']
                south_merd[i, :-1] = scalars['south_mer_drifts_scalar']
                eq_zonald[i, :-1] = scalars['equator_zon_drifts_scalar']
                eq_merd[i, :-1] = scalars['equator_mer_drifts_scalar']
        # account for periodicity
        north_zonal[:, -1] = north_zonal[:, 0]
        north_mer[:, -1] = north_mer[:, 0]
        south_zonal[:, -1] = south_zonal[:, 0]
        south_mer[:, -1] = south_mer[:, 0]
        eq_zonal[:, -1] = eq_zonal[:, 0]
        eq_mer[:, -1] = eq_mer[:, 0]
        north_zonald[:, -1] = north_zonald[:, 0]
        north_merd[:, -1] = north_merd[:, 0]
        south_zonald[:, -1] = south_zonald[:, 0]
        south_merd[:, -1] = south_merd[:, 0]
        eq_zonald[:, -1] = eq_zonald[:, 0]
        eq_merd[:, -1] = eq_merd[:, 0]

        xtickvals = ['-25', '-12.5', '0', '12.5', '25']
        xtickarr = np.array([0, 0.25, 0.5, 0.75, 1])*(len(p_lats) - 1)
        ytickarr = np.array([0, 0.2, 0.4, 0.6, 0.8, 1])*len(p_longs)

        try:
            fig = plt.figure()
            plt.imshow(eq_zonal, origin='lower')  # , vmin=0, vmax=2)
            plt.colorbar()
            plt.yticks(xtickarr, xtickvals)
            plt.xticks(ytickarr, ['0', '72', '144', '216', '288', '360'])
            plt.title('Meridional Electric Field Mapping to Magnetic Equator')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.tight_layout()
            plt.savefig('eq_mer_field.pdf')
            plt.close()

            fig = plt.figure()
            plt.imshow(eq_mer, origin='lower')  # , vmin=0, vmax=1.)
            plt.colorbar()
            plt.yticks(xtickarr, ['-50', '-25', '0', '25', '50'])
            plt.xticks(ytickarr, ['0', '72', '144', '216', '288', '360'])
            plt.title('Zonal Electric Field Mapping to Magnetic Equator')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.tight_layout()
            plt.savefig('eq_zon_field.pdf')
            plt.close()

            fig = plt.figure()
            plt.imshow(north_zonal, origin='lower')  # , vmin=0, vmax=2)
            plt.colorbar()
            plt.yticks(xtickarr, ['-50', '-25', '0', '25', '50'])
            plt.xticks(ytickarr, ['0', '72', '144', '216', '288', '360'])
            plt.title('Meridional Electric Field Mapping to Northern Footpoint')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.tight_layout()
            plt.savefig('north_mer_field.pdf')
            plt.close()

            fig = plt.figure()
            plt.imshow(north_mer, origin='lower')  # , vmin=0, vmax=2)
            plt.colorbar()
            plt.yticks(xtickarr, ['-50', '-25', '0', '25', '50'])
            plt.xticks(ytickarr, ['0', '72', '144', '216', '288', '360'])
            plt.title('Zonal Electric Field Mapping to Northern Footpoint')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.tight_layout()
            plt.savefig('north_zon_field.pdf')
            plt.close()

            fig = plt.figure()
            plt.imshow(south_zonal, origin='lower')  # , vmin=0, vmax=2)
            plt.colorbar()
            plt.yticks(xtickarr, ['-50', '-25', '0', '25', '50'])
            plt.xticks(ytickarr, ['0', '72', '144', '216', '288', '360'])
            plt.title('Meridional Electric Field Mapping to Southern Footpoint')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.tight_layout()
            plt.savefig('south_mer_field.pdf')
            plt.close()

            fig = plt.figure()
            plt.imshow(south_mer, origin='lower')  # , vmin=0, vmax=2)
            plt.colorbar()
            plt.yticks(xtickarr, ['-50', '-25', '0', '25', '50'])
            plt.xticks(ytickarr, ['0', '72', '144', '216', '288', '360'])
            plt.title('Zonal Electric Field Mapping to Southern Footpoint')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.tight_layout()
            plt.savefig('south_zon_field.pdf')
            plt.close()

            fig = plt.figure()
            plt.imshow(np.log10(eq_zonald), origin='lower')  # , vmin=0, vmax=2)
            plt.colorbar()
            plt.yticks(xtickarr, xtickvals)
            plt.xticks(ytickarr, ['0', '72', '144', '216', '288', '360'])
            plt.title('Log Zonal Ion Drift Mapping to Magnetic Equator')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.tight_layout()
            plt.savefig('eq_zonal_drift.pdf')
            plt.close()

            fig = plt.figure()
            plt.imshow(np.log10(eq_merd), origin='lower')  # , vmin=0, vmax=1.)
            plt.colorbar()
            plt.yticks(xtickarr, xtickvals)
            plt.xticks(ytickarr, ['0', '72', '144', '216', '288', '360'])
            plt.title('Log Meridional Ion Drift Mapping to Magnetic Equator')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.tight_layout()
            plt.savefig('eq_mer_drift.pdf')
            plt.close()

            fig = plt.figure()
            plt.imshow(north_zonald, origin='lower')  # , vmin=0, vmax=2)
            plt.colorbar()
            plt.yticks(xtickarr, xtickvals)
            plt.xticks(ytickarr, ['0', '72', '144', '216', '288', '360'])
            plt.title('Zonal Ion Drift Mapping to Northern Footpoint')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.tight_layout()
            plt.savefig('north_zonal_drift.pdf')
            plt.close()

            fig = plt.figure()
            plt.imshow(north_merd, origin='lower')  # , vmin=0, vmax=2)
            plt.colorbar()
            plt.yticks(xtickarr, xtickvals)
            plt.xticks(ytickarr, ['0', '72', '144', '216', '288', '360'])
            plt.title('Meridional Ion Drift Mapping to Northern Footpoint')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.tight_layout()
            plt.savefig('north_mer_drift.pdf')
            plt.close()

            fig = plt.figure()
            plt.imshow(south_zonald, origin='lower')  # , vmin=0, vmax=2)
            plt.colorbar()
            plt.yticks(xtickarr, xtickvals)
            plt.xticks(ytickarr, ['0', '72', '144', '216', '288', '360'])
            plt.title('Zonal Ion Drift Mapping to Southern Footpoint')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.tight_layout()
            plt.savefig('south_zonal_drift.pdf')
            plt.close()

            fig = plt.figure()
            plt.imshow(south_merd, origin='lower')  # , vmin=0, vmax=2)
            plt.colorbar()
            plt.yticks(xtickarr, xtickvals)
            plt.xticks(ytickarr, ['0', '72', '144', '216', '288', '360'])
            plt.title('Meridional Ion Drift Mapping to Southern Footpoint')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.tight_layout()
            plt.savefig('south_mer_drift.pdf')
            plt.close()

        except:
            pass

        return
