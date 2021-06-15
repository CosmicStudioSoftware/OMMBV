from nose.tools import assert_almost_equals as asseq

import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pds

import OMMBV
from OMMBV import igrf
import pysat

from OMMBV.tests.test_core import gen_data_fixed_alt, gen_trace_data_fixed_alt
from OMMBV.tests.test_core import gen_plot_grid_fixed_alt
from OMMBV.tests.test_core import dview, dc

# results from omniweb calculator
omni_list = [[550., 20.00, 0.00, 29.77, 359.31, -9.04, 3.09],
             [550., 20.00, 7.50, 29.50, 7.19, -8.06, 9.54],
             [550., 20.00, 15.00, 28.34, 15.01, -7.51, 16.20],
             [550., 20.00, 22.50, 27.61, 22.68, -7.75, 23.10],
             [550., 20.00, 30.00, 27.36, 30.27, -8.85, 30.34],
             [550., 20.00, 37.50, 27.10, 37.79, -10.13, 38.00],
             [550., 20.00, 45.00, 26.89, 45.24, -11.24, 46.09],
             [550., 20.00, 52.50, 26.77, 52.65, -11.80, 54.31],
             [550., 20.00, 60.00, 26.75, 60.05, -11.77, 62.31],
             [550., 20.00, 67.50, 26.80, 67.49, -11.34, 69.94],
             [550., 20.00, 75.00, 26.89, 74.95, -10.80, 77.22],
             [550., 20.00, 82.50, 26.97, 82.45, -10.37, 84.27],
             [550., 20.00, 90.00, 27.01, 89.94, -10.20, 91.22],
             [550., 20.00, 97.50, 27.01, 97.42, -10.31, 98.19],
             [550., 20.00, 105.00, 26.99, 104.86, -10.64, 105.27],
             [550., 20.00, 112.50, 27.02, 112.26, -11.00, 112.49],
             [550., 20.00, 120.00, 27.11, 119.65, -11.25, 119.77],
             [550., 20.00, 127.50, 27.28, 127.08, -11.35, 126.99],
             [550., 20.00, 135.00, 27.50, 134.59, -11.41, 134.03],
             [550., 20.00, 142.50, 27.69, 142.23, -11.63, 140.83],
             [550., 20.00, 150.00, 27.76, 149.98, -12.20, 147.44],
             [550., 20.00, 157.50, 27.66, 157.81, -13.19, 153.90],
             [550., 20.00, 165.00, 27.37, 165.64, -14.60, 160.32],
             [550., 20.00, 172.50, 26.96, 173.39, -16.33, 166.77],
             [550., 20.00, 180.00, 26.50, 181.04, -18.23, 173.34],
             [550., 20.00, 187.50, 26.10, 188.60, -20.15, 180.05],
             [550., 20.00, 195.00, 25.78, 196.11, -22.00, 186.89],
             [550., 20.00, 202.50, 25.53, 203.62, -23.77, 193.77],
             [550., 20.00, 210.00, 25.31, 211.12, -25.52, 200.61],
             [550., 20.00, 217.50, 25.09, 218.62, -27.37, 207.40],
             [550., 20.00, 225.00, 24.87, 226.09, -29.37, 214.17],
             [550., 20.00, 232.50, 24.64, 233.52, -31.56, 220.97],
             [550., 20.00, 240.00, 24.42, 240.93, -33.92, 227.85],
             [550., 20.00, 247.50, 24.19, 248.29, -36.49, 234.86],
             [550., 20.00, 255.00, 23.98, 255.62, -39.28, 242.16],
             [550., 20.00, 262.50, 23.80, 262.90, -42.28, 250.04],
             [550., 20.00, 270.00, 23.66, 270.13, -45.35, 259.07],
             [550., 20.00, 277.50, 23.61, 277.33, -48.05, 270.08],
             [550., 20.00, 285.00, 23.65, 284.50, -49.36, 283.83],
             [550., 20.00, 292.50, 23.81, 291.67, -47.58, 299.53],
             [550., 20.00, 300.00, 24.10, 298.85, -41.86, 313.55],
             [550., 20.00, 307.50, 24.55, 306.06, -34.06, 323.42],
             [550., 20.00, 315.00, 25.14, 313.34, -26.36, 330.13],
             [550., 20.00, 322.50, 25.87, 320.73, -19.73, 335.31],
             [550., 20.00, 330.00, 26.63, 328.27, -14.56, 340.08],
             [550., 20.00, 337.50, 28.33, 335.63, -12.03, 345.55],
             [550., 20.00, 345.00, 29.45, 343.37, -10.82, 351.18],
             [550., 20.00, 352.50, 30.17, 351.27, -9.90, 356.93]]
omni = pds.DataFrame(omni_list, columns=['p_alt', 'p_lat', 'p_long', 'n_lat', 'n_long', 's_lat', 's_long'])


class TestTracing():

    def __init__(self):
        # placeholder for data management features
        self.inst = pysat.Instrument('pysat', 'testing')
        self.inst.yr = 2010.
        self.inst.doy = 1.
        self.dview = dview
        self.dc = dc
        return

    def test_field_line_tracing_against_vitmo(self):
        """Compare model to http://omniweb.gsfc.nasa.gov/vitmo/cgm_vitmo.html"""

        # convert position to ECEF
        ecf_x, ecf_y, ecf_z = OMMBV.geocentric_to_ecef(omni['p_lat'],
                                                       omni['p_long'],
                                                       omni['p_alt'])
        trace_n = []
        trace_s = []
        date = datetime.datetime(2000, 1, 1)
        for x, y, z in zip(ecf_x, ecf_y, ecf_z):
            # trace north and south, take last points
            trace_n.append(
                OMMBV.field_line_trace(np.array([x, y, z]), date, 1., 0.,
                                      step_size=0.5, max_steps=1.E6)[-1, :])
            trace_s.append(
                OMMBV.field_line_trace(np.array([x, y, z]), date, -1., 0.,
                                      step_size=0.5, max_steps=1.E6)[-1, :])
        trace_n = pds.DataFrame(trace_n, columns=['x', 'y', 'z'])
        trace_n['lat'], trace_n['long'], trace_n['altitude'] = OMMBV.ecef_to_geocentric(trace_n['x'],
                                                                                        trace_n['y'],
                                                                                        trace_n['z'])
        trace_s = pds.DataFrame(trace_s, columns=['x', 'y', 'z'])
        trace_s['lat'], trace_s['long'], trace_s['altitude'] = OMMBV.ecef_to_geocentric(trace_s['x'],
                                                                                        trace_s['y'],
                                                                                        trace_s['z'])

        # ensure longitudes are all 0-360
        # idx, = np.where(omni['n_long'] < 0)
        omni.loc[omni['n_long'] < 0, 'n_long'] += 360.
        # idx, = np.where(omni['s_long'] < 0)
        omni.loc[omni['s_long'] < 0, 's_long'] += 360.

        # idx, = np.where(trace_n['long'] < 0)
        trace_n.loc[trace_n['long'] < 0, 'long'] += 360.
        # idx, = np.where(trace_s['long'] < 0)
        trace_s.loc[trace_s['long'] < 0, 'long'] += 360.

        # compute difference between OMNI and local calculation
        # there is a difference near 0 longitude, ignore this area
        diff_n_lat = (omni['n_lat'] - trace_n['lat'])[4:-4]
        diff_n_lon = (omni['n_long'] - trace_n['long'])[4:-4]
        diff_s_lat = (omni['s_lat'] - trace_s['lat'])[4:-4]
        diff_s_lon = (omni['s_long'] - trace_s['long'])[4:-4]

        try:
            f = plt.figure()
            plt.plot(omni['n_long'], omni['n_lat'], 'r.', label='omni')
            plt.plot(omni['s_long'], omni['s_lat'], 'r.', label='_omni')
            plt.plot(trace_n['long'], trace_n['lat'], 'b.', label='UTD')
            plt.plot(trace_s['long'], trace_s['lat'], 'b.', label='_UTD')
            plt.title('Comparison of Magnetic Footpoints for Field Lines through 20 Lat, 550 km')
            plt.xlabel('Geographic Longitude')
            plt.ylabel('Geographic Latitude')
            plt.legend(loc=0)
            plt.xlim((0, 360.))
            plt.savefig('magnetic_footpoint_comparison.pdf')
            print('Saving magnetic_footpoint_comparison.pdf')
        except:
            pass

        # better than 0.5 km accuracy expected for settings above
        assert np.all(np.nanstd(diff_n_lat) < .5)
        assert np.all(np.nanstd(diff_n_lon) < .5)
        assert np.all(np.nanstd(diff_s_lat) < .5)
        assert np.all(np.nanstd(diff_s_lon) < .5)

    def test_tracing_accuracy(self):
        """Establish performance of field-line tracing as function of step_size"""
        lats, longs, alts = gen_trace_data_fixed_alt(550.)
        ecf_x, ecf_y, ecf_z = OMMBV.geodetic_to_ecef(lats,
                                                    longs,
                                                    alts)
        # step size to be tried, move by inverse powers of 2 (half step size)
        steps_goal = np.arange(13)
        steps_goal = 1000. / 2 ** steps_goal
        # max number of steps (fixed)
        max_steps_goal = steps_goal * 0 + 1E6

        date = datetime.datetime(2000, 1, 1)
        dx = []
        dy = []
        dz = []

        # set up multi
        if self.dc is not None:
            # if False:
            import itertools
            targets = itertools.cycle(dc.ids)
            pending = []
            for x, y, z in zip(ecf_x, ecf_y, ecf_z):
                for steps, max_steps in zip(steps_goal, max_steps_goal):
                    # iterate through target cyclicly and run commands
                    dview.targets = next(targets)
                    pending.append(dview.apply_async(OMMBV.field_line_trace,
                                                     np.array([x, y, z]), date, 1., 0.,
                                                     step_size=steps,
                                                     max_steps=max_steps))
                # for x, y, z in zip(ecf_x, ecf_y, ecf_z):
                out = []
                for steps, max_steps in zip(steps_goal, max_steps_goal):
                    # collect output
                    trace_n = pending.pop(0).get()
                    pt = trace_n[-1, :]
                    out.append(pt)

                final_pt = pds.DataFrame(out, columns=['x', 'y', 'z'])
                dx.append(np.abs(final_pt.loc[1:, 'x'].values - final_pt.loc[:, 'x'].values[:-1]))
                dy.append(np.abs(final_pt.loc[1:, 'y'].values - final_pt.loc[:, 'y'].values[:-1]))
                dz.append(np.abs(final_pt.loc[1:, 'z'].values - final_pt.loc[:, 'z'].values[:-1]))
        else:
            for x, y, z in zip(ecf_x, ecf_y, ecf_z):
                out = []
                for steps, max_steps in zip(steps_goal, max_steps_goal):
                    trace_n = OMMBV.field_line_trace(np.array([x, y, z]), date, 1., 0.,
                                                    step_size=steps,
                                                    max_steps=max_steps)
                    pt = trace_n[-1, :]
                    out.append(pt)

                final_pt = pds.DataFrame(out, columns=['x', 'y', 'z'])
                dx.append(np.abs(final_pt.loc[1:, 'x'].values - final_pt.loc[:, 'x'].values[:-1]))
                dy.append(np.abs(final_pt.loc[1:, 'y'].values - final_pt.loc[:, 'y'].values[:-1]))
                dz.append(np.abs(final_pt.loc[1:, 'z'].values - final_pt.loc[:, 'z'].values[:-1]))

        dx = pds.DataFrame(dx)
        dy = pds.DataFrame(dy)
        dz = pds.DataFrame(dz)

        try:
            plt.figure()
            yerrx = np.nanstd(np.log10(dx), axis=0)
            yerry = np.nanstd(np.log10(dy), axis=0)
            yerrz = np.nanstd(np.log10(dz), axis=0)
            plt.errorbar(np.log10(steps_goal[1:]), np.log10(dx.mean(axis=0)),
                         yerr=yerrx, label='x')
            plt.errorbar(np.log10(steps_goal[1:]), np.log10(dy.mean(axis=0)),
                         yerr=yerry, label='y')
            plt.errorbar(np.log10(steps_goal[1:]), np.log10(dz.mean(axis=0)),
                         yerr=yerrz, label='z')
            plt.xlabel('Log Step Size (km)')
            plt.ylabel('Change in Foot Point Position (km)')
            plt.title("Change in Final ECEF Position")
            plt.legend()
            plt.tight_layout()
            plt.savefig('Footpoint_position_vs_step_size.pdf')
            plt.close()
        except:
            pass

    def test_tracing_accuracy_w_recursion(self):
        """Establish performance field-line vs max_steps"""
        lats, longs, alts = gen_trace_data_fixed_alt(550.)
        ecf_x, ecf_y, ecf_z = OMMBV.geodetic_to_ecef(lats,
                                                    longs,
                                                    alts)
        # step size to be tried
        steps_goal = np.array([5.] * 13)
        # max number of steps (fixed)
        max_steps_goal = np.arange(13)
        max_steps_goal = 100000. / 2 ** max_steps_goal

        date = datetime.datetime(2000, 1, 1)
        dx = []
        dy = []
        dz = []
        for x, y, z in zip(ecf_x, ecf_y, ecf_z):
            out = []
            for steps, max_steps in zip(steps_goal, max_steps_goal):
                trace_n = OMMBV.field_line_trace(np.array([x, y, z]), date, 1., 0.,
                                                step_size=steps,
                                                max_steps=max_steps)
                pt = trace_n[-1, :]
                out.append(pt)

            final_pt = pds.DataFrame(out, columns=['x', 'y', 'z'])
            dx.append(np.abs(final_pt.loc[1:, 'x'].values - final_pt.loc[:, 'x'].values[:-1]))
            dy.append(np.abs(final_pt.loc[1:, 'y'].values - final_pt.loc[:, 'y'].values[:-1]))
            dz.append(np.abs(final_pt.loc[1:, 'z'].values - final_pt.loc[:, 'z'].values[:-1]))
        dx = pds.DataFrame(dx)
        dy = pds.DataFrame(dy)
        dz = pds.DataFrame(dz)

        try:
            plt.figure()
            yerrx = np.nanstd(np.log10(dx), axis=0)
            yerry = np.nanstd(np.log10(dy), axis=0)
            yerrz = np.nanstd(np.log10(dz), axis=0)
            plt.errorbar(np.log10(max_steps_goal[1:]), np.log10(dx.mean(axis=0)),
                         yerr=yerrx,
                         label='x')
            plt.errorbar(np.log10(max_steps_goal[1:]), np.log10(dy.mean(axis=0)),
                         yerr=yerry,
                         label='y')
            plt.errorbar(np.log10(max_steps_goal[1:]), np.log10(dz.mean(axis=0)),
                         yerr=yerrz,
                         label='z')
            plt.xlabel('Log Number of Steps per Run')
            plt.ylabel('Change in Foot Point Position (km)')
            plt.title("Change in Final ECEF Position, Recursive Calls")
            plt.legend()
            plt.tight_layout()
            plt.ylabel('Log Change in Foot Point Position (km)')
            plt.savefig('Footpoint_position_vs_max_steps_recursion.pdf')
            plt.close()
        except:
            pass

    def test_tracing_accuracy_w_recursion_step_size(self):
        """Establish field-line tracing performance fixed max_steps"""
        lats, longs, alts = gen_trace_data_fixed_alt(550.)
        ecf_x, ecf_y, ecf_z = OMMBV.geodetic_to_ecef(lats,
                                                    longs,
                                                    alts)
        # step size to be tried
        steps_goal = np.arange(13)
        steps_goal = 500. / 2 ** steps_goal
        # max number of steps (fixed)
        max_steps_goal = np.array([10000.] * 13)

        date = datetime.datetime(2000, 1, 1)
        dx = []
        dy = []
        dz = []
        for x, y, z in zip(ecf_x, ecf_y, ecf_z):
            out = []
            for steps, max_steps in zip(steps_goal, max_steps_goal):
                trace_n = OMMBV.field_line_trace(np.array([x, y, z]), date, 1., 0.,
                                                step_size=steps,
                                                max_steps=max_steps)
                pt = trace_n[-1, :]
                out.append(pt)

            final_pt = pds.DataFrame(out, columns=['x', 'y', 'z'])
            dx.append(np.abs(final_pt.loc[1:, 'x'].values - final_pt.loc[:, 'x'].values[:-1]))
            dy.append(np.abs(final_pt.loc[1:, 'y'].values - final_pt.loc[:, 'y'].values[:-1]))
            dz.append(np.abs(final_pt.loc[1:, 'z'].values - final_pt.loc[:, 'z'].values[:-1]))
        dx = pds.DataFrame(dx)
        dy = pds.DataFrame(dy)
        dz = pds.DataFrame(dz)

        try:
            plt.figure()
            yerrx = np.nanstd(np.log10(dx), axis=0)
            yerry = np.nanstd(np.log10(dy), axis=0)
            yerrz = np.nanstd(np.log10(dz), axis=0)
            plt.errorbar(np.log10(steps_goal[:-1]), np.log10(dx.mean(axis=0)),
                         yerr=yerrx,
                         label='x')
            plt.errorbar(np.log10(steps_goal[:-1]), np.log10(dy.mean(axis=0)),
                         yerr=yerry,
                         label='y')
            plt.errorbar(np.log10(steps_goal[:-1]), np.log10(dz.mean(axis=0)),
                         yerr=yerrz,
                         label='z')
            plt.xlabel('Log Step Size (km)')
            plt.ylabel('Log Change in Foot Point Position (km)')
            plt.title("Change in Final ECEF Position, Recursive Calls")
            plt.legend()
            plt.tight_layout()
            plt.savefig('Footpoint_position_vs_step_size__recursion.pdf')
            plt.close()
        except:
            pass


############### TRANSFORMATIONS  ##################################

class TestTransformations():

    def __init__(self):
        # placeholder for data management features
        self.inst = pysat.Instrument('pysat', 'testing')
        self.inst.yr = 2010.
        self.inst.doy = 1.
        self.dview = dview
        self.dc = dc
        return

    def test_geodetic_to_ecef_to_geodetic(self):
        """Geodetic to ECEF and back"""
        lats, longs, alts = gen_data_fixed_alt(550.)
        ecf_x, ecf_y, ecf_z = OMMBV.geodetic_to_ecef(lats,
                                                    longs,
                                                    alts)
        lat, elong, alt = OMMBV.ecef_to_geodetic(ecf_x, ecf_y, ecf_z)
        lat2, elong2, alt2 = OMMBV.python_ecef_to_geodetic(ecf_x, ecf_y, ecf_z)

        idx, = np.where(elong < 0)
        elong[idx] += 360.

        idx, = np.where(elong2 < 0)
        elong2[idx] += 360.

        d_lat = lat - lats
        d_long = elong - longs
        d_alt = alt - alts

        assert np.all(np.abs(d_lat) < 1.E-5)
        assert np.all(np.abs(d_long) < 1.E-5)
        assert np.all(np.abs(d_alt) < 1.E-5)

        d_lat = lat2 - lats
        d_long = elong2 - longs
        d_alt = alt2 - alts

        assert np.all(np.abs(d_lat) < 1.E-5)
        assert np.all(np.abs(d_long) < 1.E-5)
        assert np.all(np.abs(d_alt) < 1.E-5)

    def test_geodetic_to_ecef_to_geodetic_via_different_methods(self):
        """Multiple techniques for geodetic to ECEF to geodetic"""
        lats, longs, alts = gen_data_fixed_alt(550.)
        ecf_x, ecf_y, ecf_z = OMMBV.geodetic_to_ecef(lats,
                                                     longs,
                                                     alts)
        methods = ['closed', 'iterative']
        for method in methods:
            lat, elong, alt = OMMBV.python_ecef_to_geodetic(ecf_x, ecf_y, ecf_z,
                                                            method=method)

            idx, = np.where(elong < 0)
            elong[idx] += 360.

            d_lat = lat - lats
            d_long = elong - longs
            d_alt = alt - alts

            assert np.all(np.abs(d_lat) < 1.E-5)
            assert np.all(np.abs(d_long) < 1.E-5)
            assert np.all(np.abs(d_alt) < 1.E-5)

    def test_geodetic_to_ecef_to_geocentric_to_ecef_to_geodetic(self):
        """geodetic to ecef and geocentric transformations"""
        lats, longs, alts = gen_data_fixed_alt(550.)
        ecf_x, ecf_y, ecf_z = OMMBV.geodetic_to_ecef(lats,
                                                    longs,
                                                    alts)
        geo_lat, geo_long, geo_alt = OMMBV.ecef_to_geocentric(ecf_x, ecf_y, ecf_z)

        ecfs_x, ecfs_y, ecfs_z = OMMBV.geocentric_to_ecef(geo_lat, geo_long, geo_alt)

        lat, elong, alt = OMMBV.ecef_to_geodetic(ecfs_x, ecfs_y, ecfs_z)

        idx, = np.where(elong < 0)
        elong[idx] += 360.

        d_lat = lat - lats
        d_long = elong - longs
        d_alt = alt - alts

        assert np.all(np.abs(d_lat) < 1.E-5)
        assert np.all(np.abs(d_long) < 1.E-5)
        assert np.all(np.abs(d_alt) < 1.E-5)
        assert np.all(np.abs(ecf_x - ecfs_x) < 1.E-5)
        assert np.all(np.abs(ecf_y - ecfs_y) < 1.E-5)
        assert np.all(np.abs(ecf_z - ecfs_z) < 1.E-5)

    def test_geocentric_to_ecef_to_geocentric(self):
        """Geocentric and ECEF transformations"""
        lats, longs, alts = gen_data_fixed_alt(550.)
        ecf_x, ecf_y, ecf_z = OMMBV.geocentric_to_ecef(lats,
                                                      longs,
                                                      alts)
        lat, elong, alt = OMMBV.ecef_to_geocentric(ecf_x, ecf_y, ecf_z)

        idx, = np.where(elong < 0)
        elong[idx] += 360.

        d_lat = lat - lats
        d_long = elong - longs
        d_alt = alt - alts

        assert np.all(np.abs(d_lat) < 1.E-5)
        assert np.all(np.abs(d_long) < 1.E-5)
        assert np.all(np.abs(d_alt) < 1.E-5)

    def test_ecef_geodetic_diff_plots(self):
        """Generate uncertainty plots of ECEF-Geodetic transformations"""
        import matplotlib.pyplot as plt
        import os
        # on_travis = os.environ.get('ONTRAVIS') == 'True'

        p_lats, p_longs, p_alts = gen_plot_grid_fixed_alt(550.)
        # data returned are the locations along each direction
        # the full range of points obtained by iterating over all
        # recasting alts into a more convenient form for later calculation
        p_alts = [p_alts[0]] * len(p_longs)
        # set the date
        date = datetime.datetime(2000, 1, 1)
        # memory for results
        apex_x = np.zeros((len(p_lats), len(p_longs) + 1))
        apex_y = np.zeros((len(p_lats), len(p_longs) + 1))
        apex_z = np.zeros((len(p_lats), len(p_longs) + 1))
        norm_alt = np.zeros((len(p_lats), len(p_longs) + 1))

        # single processor case
        for i, p_lat in enumerate(p_lats):
            print (i, p_lat)
            x, y, z = OMMBV.geodetic_to_ecef([p_lat] * len(p_longs), p_longs, p_alts)
            lat2, lon2, alt2 = OMMBV.ecef_to_geodetic(x, y, z)
            x2, y2, z2 = OMMBV.geodetic_to_ecef(lat2, lon2, alt2)
            apex_x[i, :-1] = np.abs(x2 - x)
            apex_y[i, :-1] = np.abs(y2 - y)
            apex_z[i, :-1] = np.abs(z2 - z)

        # account for periodicity
        apex_x[:, -1] = apex_x[:, 0]
        apex_y[:, -1] = apex_y[:, 0]
        apex_z[:, -1] = apex_z[:, 0]

        ytickarr = np.array([0, 0.25, 0.5, 0.75, 1]) * (len(p_lats) - 1)
        xtickarr = np.array([0, 0.2, 0.4, 0.6, 0.8, 1]) * len(p_longs)
        ytickvals = ['-50', '-25', '0', '25', '50']

        try:
            fig = plt.figure()
            plt.imshow(np.log10(apex_x), origin='lower')
            plt.colorbar()
            plt.yticks(ytickarr, ytickvals)
            plt.xticks(xtickarr, ['0', '72', '144', '216', '288', '360'])
            plt.title('Log ECEF-Geodetic Location Difference (ECEF-x km)')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.savefig('ecef_geodetic_diff_x.pdf')
            plt.close()

            fig = plt.figure()
            plt.imshow(np.log10(apex_y), origin='lower')
            plt.colorbar()
            plt.yticks(ytickarr, ytickvals)
            plt.xticks(xtickarr, ['0', '72', '144', '216', '288', '360'])
            plt.title('Log ECEF-Geodetic Location Difference (ECEF-y km)')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.savefig('ecef_geodetic_diff_y.pdf')
            plt.close()

            fig = plt.figure()
            plt.imshow(np.log10(apex_z), origin='lower')
            plt.colorbar()
            plt.yticks(ytickarr, ytickvals)
            plt.xticks(xtickarr, ['0', '72', '144', '216', '288', '360'])
            plt.title('Log ECEF-Geodetic Location Difference (ECEF-z km)')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.savefig('ecef_geodetic_diff_z.pdf')
            plt.close()

        except:
            pass

    def test_basic_ecef_to_enu_rotations(self):
        """Test ECEF to ENU Vector Rotations"""
        # test basic transformations first
        # vector pointing along ecef y at 0, 0 is east
        ve, vn, vu = OMMBV.ecef_to_enu_vector(0., 1., 0., 0., 0.)
        # print ('{:9f}, {:9f}, {:9f}'.format(ve, vn, vu))
        asseq(ve, 1.0, 9)
        asseq(vn, 0, 9)
        asseq(vu, 0, 9)
        # vector pointing along ecef x at 0, 0 is up
        ve, vn, vu = OMMBV.ecef_to_enu_vector(1., 0., 0., 0., 0.)
        asseq(ve, 0.0, 9)
        asseq(vn, 0.0, 9)
        asseq(vu, 1.0, 9)
        # vector pointing along ecef z at 0, 0 is north
        ve, vn, vu = OMMBV.ecef_to_enu_vector(0., 0., 1., 0., 0.)
        asseq(ve, 0.0, 9)
        asseq(vn, 1.0, 9)
        asseq(vu, 0.0, 9)

        # vector pointing along ecef x at 0, 90 long is west
        ve, vn, vu = OMMBV.ecef_to_enu_vector(1., 0., 0., 0., 90.)
        # print ('{:9f}, {:9f}, {:9f}'.format(ve, vn, vu))
        asseq(ve, -1.0, 9)
        asseq(vn, 0.0, 9)
        asseq(vu, 0.0, 9)
        # vector pointing along ecef y at 0, 90 long is up
        ve, vn, vu = OMMBV.ecef_to_enu_vector(0., 1., 0., 0., 90.)
        asseq(ve, 0.0, 9)
        asseq(vn, 0.0, 9)
        asseq(vu, 1.0, 9)
        # vector pointing along ecef z at 0, 90 long is north
        ve, vn, vu = OMMBV.ecef_to_enu_vector(0., 0., 1., 0., 90.)
        asseq(ve, 0.0, 9)
        asseq(vn, 1.0, 9)
        asseq(vu, 0.0, 9)

        # vector pointing along ecef y at 0, 180 is west
        ve, vn, vu = OMMBV.ecef_to_enu_vector(0., 1., 0., 0., 180.)
        asseq(ve, -1.0, 9)
        asseq(vn, 0.0, 9)
        asseq(vu, 0.0, 9)
        # vector pointing along ecef x at 0, 180 is down
        ve, vn, vu = OMMBV.ecef_to_enu_vector(1., 0., 0., 0., 180.)
        asseq(ve, 0.0, 9)
        asseq(vn, 0.0, 9)
        asseq(vu, -1.0, 9)
        # vector pointing along ecef z at 0, 0 is north
        ve, vn, vu = OMMBV.ecef_to_enu_vector(0., 0., 1., 0., 180.)
        asseq(ve, 0.0, 9)
        asseq(vn, 1.0, 9)
        asseq(vu, 0.0, 9)

        ve, vn, vu = OMMBV.ecef_to_enu_vector(0., 1., 0., 45., 0.)
        # print ('{:9f}, {:9f}, {:9f}'.format(ve, vn, vu))
        asseq(ve, 1.0, 9)
        asseq(vn, 0, 9)
        asseq(vu, 0, 9)
        # vector pointing along ecef x at 0, 0 is south/up
        ve, vn, vu = OMMBV.ecef_to_enu_vector(1., 0., 0., 45., 0.)
        asseq(ve, 0.0, 9)
        asseq(vn, -np.cos(np.pi / 4), 9)
        asseq(vu, np.cos(np.pi / 4), 9)
        # vector pointing along ecef z at 45, 0 is north/up
        ve, vn, vu = OMMBV.ecef_to_enu_vector(0., 0., 1., 45., 0.)
        asseq(ve, 0.0, 9)
        asseq(vn, np.cos(np.pi / 4), 9)
        asseq(vu, np.cos(np.pi / 4), 9)

    def test_basic_enu_to_ecef_rotations(self):
        """Test ENU to ECEF rotations"""
        # test basic transformations first
        # vector pointing east at 0, 0 is along y
        vx, vy, vz = OMMBV.enu_to_ecef_vector(1., 0., 0., 0., 0.)
        # print ('{:9f}, {:9f}, {:9f}'.format(vx, vy, vz))
        asseq(vx, 0.0, 9)
        asseq(vy, 1.0, 9)
        asseq(vz, 0.0, 9)
        # vector pointing up at 0, 0 is along x
        vx, vy, vz = OMMBV.enu_to_ecef_vector(0., 0., 1., 0., 0.)
        asseq(vx, 1.0, 9)
        asseq(vy, 0.0, 9)
        asseq(vz, 0.0, 9)
        # vector pointing north at 0, 0 is along z
        vx, vy, vz = OMMBV.enu_to_ecef_vector(0., 1., 0., 0., 0.)
        asseq(vx, 0.0, 9)
        asseq(vy, 0.0, 9)
        asseq(vz, 1.0, 9)

        # east vector at 0, 90 long points along -x
        vx, vy, vz = OMMBV.enu_to_ecef_vector(1., 0., 0., 0., 90.)
        # print ('{:9f}, {:9f}, {:9f}'.format(vx, vy, vz))
        asseq(vx, -1.0, 9)
        asseq(vy, 0.0, 9)
        asseq(vz, 0.0, 9)
        # vector pointing up at 0, 90 is along y
        vx, vy, vz = OMMBV.enu_to_ecef_vector(0., 0., 1., 0., 90.)
        asseq(vx, 0.0, 9)
        asseq(vy, 1.0, 9)
        asseq(vz, 0.0, 9)
        # vector pointing north at 0, 90 is along z
        vx, vy, vz = OMMBV.enu_to_ecef_vector(0., 1., 0., 0., 90.)
        asseq(vx, 0.0, 9)
        asseq(vy, 0.0, 9)
        asseq(vz, 1.0, 9)

        # vector pointing east at 0, 0 is along y
        vx, vy, vz = OMMBV.enu_to_ecef_vector(1., 0., 0., 0., 180.)
        # print ('{:9f}, {:9f}, {:9f}'.format(vx, vy, vz))
        asseq(vx, 0.0, 9)
        asseq(vy, -1.0, 9)
        asseq(vz, 0.0, 9)
        # vector pointing up at 0, 180 is along -x
        vx, vy, vz = OMMBV.enu_to_ecef_vector(0., 0., 1., 0., 180.)
        asseq(vx, -1.0, 9)
        asseq(vy, 0.0, 9)
        asseq(vz, 0.0, 9)
        # vector pointing north at 0, 180 is along z
        vx, vy, vz = OMMBV.enu_to_ecef_vector(0., 1., 0., 0., 180.)
        asseq(vx, 0.0, 9)
        asseq(vy, 0.0, 9)
        asseq(vz, 1.0, 9)

    def test_ecef_to_enu_back_to_ecef(self):
        """Test ECEF-ENU-ECEF"""
        vx = 0.9
        vy = 0.1
        vz = np.sqrt(1. - vx ** 2 + vy ** 2)
        lats, longs, alts = gen_data_fixed_alt(550.)
        for lat, lon, alt in zip(lats, longs, alts):
            vxx, vyy, vzz = OMMBV.ecef_to_enu_vector(vx, vy, vz, lat, lon)
            vxx, vyy, vzz = OMMBV.enu_to_ecef_vector(vxx, vyy, vzz, lat, lon)
            asseq(vx, vxx, 9)
            asseq(vy, vyy, 9)
            asseq(vz, vzz, 9)

    def test_enu_to_ecef_back_to_enu(self):
        """Test ENU-ECEF-ENU"""
        vx = 0.9
        vy = 0.1
        vz = np.sqrt(1. - vx ** 2 + vy ** 2)
        lats, longs, alts = gen_data_fixed_alt(550.)
        for lat, lon, alt in zip(lats, longs, alts):
            vxx, vyy, vzz = OMMBV.enu_to_ecef_vector(vx, vy, vz, lat, lon)
            vxx, vyy, vzz = OMMBV.ecef_to_enu_vector(vxx, vyy, vzz, lat, lon)
            asseq(vx, vxx, 9)
            asseq(vy, vyy, 9)
            asseq(vz, vzz, 9)

    def test_igrf_end_to_ecef_back_to_end(self):
        """Check consistency ENU-ECEF and IGRF implementation"""
        # import pdb
        vx = 0.9
        vy = 0.1
        vz = np.sqrt(1. - vx ** 2 + vy ** 2)
        vz = -vz
        lats, longs, alts = gen_data_fixed_alt(550.)
        for lat, lon, alt in zip(lats, longs, alts):
            # print(vx, vy, vz, lat, lon)
            # pdb.set_trace()
            # input here is co-latitude, not latitude
            # inputs to fortran are in radians
            vxx, vyy, vzz = igrf.end_vector_to_ecef(vx, vy, vz, np.deg2rad(90. - lat), np.deg2rad(lon))
            vx2, vy2, vz2 = OMMBV.enu_to_ecef_vector(vx, vy, -vz, lat, lon)
            # print ('end check ', vxx, vyy, vzz, vx2, vy2, vz2)
            asseq(vxx, vx2, 9)
            asseq(vyy, vy2, 9)
            asseq(vzz, vz2, 9)

            vxx, vyy, vzz = OMMBV.ecef_to_enu_vector(vxx, vyy, vzz, lat, lon)
            # convert upward component back to down
            vzz = -vzz
            # compare original inputs to outputs
            asseq(vx, vxx, 9)
            asseq(vy, vyy, 9)
            asseq(vz, vzz, 9)

    def test_igrf_ecef_to_geodetic_back_to_ecef(self):
        """Test IGRF_ECEF - Geodetic - and Back"""
        lats, longs, alts = gen_data_fixed_alt(550.)
        ecf_x, ecf_y, ecf_z = OMMBV.geodetic_to_ecef(lats,
                                                    longs,
                                                    alts)
        for ecef_x, ecef_y, ecef_z, geo_lat, geo_lon, geo_alt in zip(ecf_x, ecf_y,
                                                                     ecf_z, lats, longs, alts):
            pos = np.array([ecef_x, ecef_y, ecef_z])
            lat, elong, alt = igrf.ecef_to_geodetic(pos)
            lat = np.rad2deg(lat)
            elong = np.rad2deg(elong)
            if (elong < 0):
                elong = elong + 360.

            d_lat = lat - geo_lat
            d_long = elong - geo_lon
            d_alt = alt - geo_alt

            # print ('Word', ecef_x, ecef_y, ecef_z)
            # print (geo_lat, geo_lon, geo_alt)
            # print (lat, elong, alt)
            assert np.all(np.abs(d_lat) < 1.E-5)
            assert np.all(np.abs(d_long) < 1.E-5)
            assert np.all(np.abs(d_alt) < 1.E-5)

    def test_igrf_ecef_to_geographic_with_colatitude(self):
        """Test IGRF_ECEF - Geographic"""
        lats, longs, alts = gen_data_fixed_alt(550.)
        ecf_x, ecf_y, ecf_z = OMMBV.geodetic_to_ecef(lats,
                                                    longs,
                                                    alts)
        for ecef_x, ecef_y, ecef_z, geo_lat, geo_lon, geo_alt in zip(ecf_x, ecf_y,
                                                                     ecf_z, lats, longs, alts):
            pos = np.array([ecef_x, ecef_y, ecef_z])

            colat, lon, r = igrf.ecef_to_colat_long_r(pos)
            # results are returned in radians
            lat = 90. - np.rad2deg(colat)
            lon = np.rad2deg(lon)

            lat2, lon2, h2 = OMMBV.ecef_to_geocentric(*pos, ref_height=0)

            # print(lat, lon, r, lat2, lon2, h2)
            asseq(r, h2, 9)
            asseq(lat, lat2, 9)
            asseq(lon, lon2, 9)
