import datetime as dt
import numpy as np

import OMMBV
import pysat

from OMMBV.tests.test_core import gen_plot_grid_fixed_alt
from OMMBV.tests.test_core import dview, dc


class TestIntegratedMethods():

    def __init__(self):
        # placeholder for data management features
        self.inst = pysat.Instrument('pysat', 'testing')
        self.inst.yr = 2010.
        self.inst.doy = 1.
        self.dview = dview
        self.dc = dc

        return

    def test_integrated_unit_vector_component_plots(self):
        """Generate Field-Line Integrated Unit Vector Plots"""
        import matplotlib.pyplot as plt

        p_lats, p_longs, p_alts = gen_plot_grid_fixed_alt(550.)
        # data returned are the locations along each direction
        # the full range of points obtained by iterating over all
        # recasting alts into a more convenient form for later calculation
        p_alts = [p_alts[0]]*len(p_longs)
        zvx = np.zeros((len(p_lats), len(p_longs) + 1))
        zvy = zvx.copy()
        zvz = zvx.copy()
        mx = zvx.copy()
        my = zvx.copy()
        mz = zvx.copy()
        bx = zvx.copy()
        by = zvx.copy()
        bz = zvx.copy()
        date = dt.datetime(2000, 1, 1)
        # set up multi
        if self.dc is not None:
            import itertools
            targets = itertools.cycle(dc.ids)
            pending = []
            for i, p_lat in enumerate(p_lats):
                # iterate through target cyclicly and run commands
                print (i, p_lat)
                dview.targets = next(targets)
                pending.append(
                    dview.apply_async(OMMBV.calculate_integrated_mag_drift_unit_vectors_ecef, [p_lat]*len(p_longs),
                                      p_longs,
                                      p_alts, [date]*len(p_longs),
                                      steps=None, max_steps=10000, step_size=10.,
                                      ref_height=120.))
            for i, p_lat in enumerate(p_lats):
                print ('collecting ', i, p_lat)
                # collect output
                tzx, tzy, tzz, tbx, tby, tbz, tmx, tmy, tmz = pending.pop(0).get()
                zvx[i, :-1], zvy[i, :-1], zvz[i, :-1] = OMMBV.ecef_to_enu_vector(tzx, tzy, tzz,
                                                                                [p_lat]*len(p_longs),
                                                                                p_longs)
                bx[i, :-1], by[i, :-1], bz[i, :-1] = OMMBV.ecef_to_enu_vector(tbx, tby, tbz,
                                                                             [p_lat]*len(p_longs),
                                                                             p_longs)
                mx[i, :-1], my[i, :-1], mz[i, :-1] = OMMBV.ecef_to_enu_vector(tmx, tmy, tmz,
                                                                             [p_lat]*len(p_longs),
                                                                             p_longs)
        else:
            for i, p_lat in enumerate(p_lats):
                print (i, p_lat)
                tzx, tzy, tzz, tbx, tby, tbz, tmx, tmy, tmz = OMMBV.calculate_integrated_mag_drift_unit_vectors_ecef(
                    [p_lat]*len(p_longs), p_longs,
                    p_alts, [date]*len(p_longs),
                    steps=None, max_steps=10000, step_size=10.,
                    ref_height=120.)
                zvx[i, :-1], zvy[i, :-1], zvz[i, :-1] = OMMBV.ecef_to_enu_vector(tzx, tzy, tzz,
                                                                                [p_lat]*len(p_longs),
                                                                                p_longs)
                bx[i, :-1], by[i, :-1], bz[i, :-1] = OMMBV.ecef_to_enu_vector(tbx, tby, tbz,
                                                                             [p_lat]*len(p_longs),
                                                                             p_longs)
                mx[i, :-1], my[i, :-1], mz[i, :-1] = OMMBV.ecef_to_enu_vector(tmx, tmy, tmz,
                                                                             [p_lat]*len(p_longs),
                                                                             p_longs)

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

        ytickarr = np.array([0, 0.25, 0.5, 0.75, 1])*(len(p_lats) - 1)
        xtickarr = np.array([0, 0.2, 0.4, 0.6, 0.8, 1])*len(p_longs)

        try:
            fig = plt.figure()
            plt.imshow(zvx, origin='lower')
            plt.colorbar()
            plt.yticks(ytickarr, ['-50', '-25', '0', '25', '50'])
            plt.xticks(xtickarr, ['0', '72', '144', '216', '288', '360'])
            plt.title('Zonal Unit Vector - Eastward')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.savefig('int_zonal_east.pdf')
            plt.close()

            fig = plt.figure()
            plt.imshow(zvy, origin='lower')
            plt.colorbar()
            plt.yticks(ytickarr, ['-50', '-25', '0', '25', '50'])
            plt.xticks(xtickarr, ['0', '72', '144', '216', '288', '360'])
            plt.title('Zonal Unit Vector - Northward')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.savefig('int_zonal_north.pdf')
            plt.close()

            fig = plt.figure()
            plt.imshow(zvz, origin='lower')
            plt.colorbar()
            plt.yticks(ytickarr, ['-50', '-25', '0', '25', '50'])
            plt.xticks(xtickarr, ['0', '72', '144', '216', '288', '360'])
            plt.title('Zonal Unit Vector - Upward')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.savefig('int_zonal_up.pdf')
            plt.close()

            fig = plt.figure()
            plt.imshow(bx, origin='lower')
            plt.colorbar()
            plt.yticks(ytickarr, ['-50', '-25', '0', '25', '50'])
            plt.xticks(xtickarr, ['0', '72', '144', '216', '288', '360'])
            plt.title('Field Aligned Unit Vector - Eastward')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.savefig('int_fa_east.pdf')
            plt.close()

            fig = plt.figure()
            plt.imshow(by, origin='lower')
            plt.colorbar()
            plt.yticks(ytickarr, ['-50', '-25', '0', '25', '50'])
            plt.xticks(xtickarr, ['0', '72', '144', '216', '288', '360'])
            plt.title('Field Aligned Unit Vector - Northward')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.savefig('int_fa_north.pdf')
            plt.close()

            fig = plt.figure()
            plt.imshow(bz, origin='lower')
            plt.colorbar()
            plt.yticks(ytickarr, ['-50', '-25', '0', '25', '50'])
            plt.xticks(xtickarr, ['0', '72', '144', '216', '288', '360'])
            plt.title('Field Aligned Unit Vector - Upward')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.savefig('int_fa_up.pdf')
            plt.close()

            fig = plt.figure()
            plt.imshow(mx, origin='lower')
            plt.colorbar()
            plt.yticks(ytickarr, ['-50', '-25', '0', '25', '50'])
            plt.xticks(xtickarr, ['0', '72', '144', '216', '288', '360'])
            plt.title('Meridional Unit Vector - Eastward')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.savefig('int_mer_east.pdf')
            plt.close()

            fig = plt.figure()
            plt.imshow(my, origin='lower')
            plt.colorbar()
            plt.yticks(ytickarr, ['-50', '-25', '0', '25', '50'])
            plt.xticks(xtickarr, ['0', '72', '144', '216', '288', '360'])
            plt.title('Meridional Unit Vector - Northward')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.savefig('int_mer_north.pdf')
            plt.close()

            fig = plt.figure()
            plt.imshow(mz, origin='lower')
            plt.colorbar()
            plt.yticks(ytickarr, ['-50', '-25', '0', '25', '50'])
            plt.xticks(xtickarr, ['0', '72', '144', '216', '288', '360'])
            plt.title('Meridional Unit Vector - Upward')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.savefig('int_mer_up.pdf')
            plt.close()
        except:
            pass

    def closed_loop_footpoint_sensitivity_plots(self, direction, vector_direction):
        import matplotlib.pyplot as plt
        # from mpl_toolkits.mplot3d import Axes3D
        import os

        p_lats, p_longs, p_alts = gen_plot_grid_fixed_alt(550.)
        # data returned are the locations along each direction
        # the full range of points obtained by iterating over all
        # recasting alts into a more convenient form for later calculation
        p_alts = [p_alts[0]]*len(p_longs)

        # zonal vector components
        # +1 on length of longitude array supports repeating first element
        # shows nice periodicity on the plots
        zvx = np.zeros((len(p_lats), len(p_longs) + 1))
        #  meridional vecrtor components
        mx = zvx.copy();
        my = zvx.copy();
        mz = zvx.copy()
        date = dt.datetime(2000, 1, 1)
        # set up multi
        if self.dc is not None:
            import itertools
            targets = itertools.cycle(dc.ids)
            pending = []
            for i, p_lat in enumerate(p_lats):
                # iterate through target cyclicly and run commands
                print (i, p_lat)
                dview.targets = next(targets)
                pending.append(
                    dview.apply_async(OMMBV.closed_loop_edge_lengths_via_footpoint,
                                      [p_lat]*len(p_longs), p_longs,
                                      p_alts, [date]*len(p_longs),
                                      direction,
                                      vector_direction,
                                      edge_length=25.,
                                      edge_steps=5))
                pending.append(
                    dview.apply_async(OMMBV.closed_loop_edge_lengths_via_footpoint,
                                      [p_lat]*len(p_longs), p_longs,
                                      p_alts, [date]*len(p_longs),
                                      direction,
                                      vector_direction,
                                      edge_length=25.,
                                      edge_steps=10))
            for i, p_lat in enumerate(p_lats):
                print ('collecting ', i, p_lat)
                # collect output from first run
                mx[i, :-1], my[i, :-1], mz[i, :-1] = pending.pop(0).get()
                # collect output from second run
                _a, _b, _c = pending.pop(0).get()
                # take difference with first run
                mx[i, :-1] = (mx[i, :-1] - _a) / mx[i, :-1]
                my[i, :-1] = (my[i, :-1] - _b) / my[i, :-1]
                mz[i, :-1] = (mz[i, :-1] - _c) / mz[i, :-1]

        else:
            for i, p_lat in enumerate(p_lats):
                print (i, p_lat)
                mx[i, :-1], my[i, :-1], mz[i, :-1] = OMMBV.closed_loop_edge_lengths_via_footpoint([p_lat]*len(p_longs),
                                                                                                 p_longs,
                                                                                                 p_alts,
                                                                                                 [date]*len(p_longs),
                                                                                                 direction,
                                                                                                 vector_direction,
                                                                                                 edge_length=25.,
                                                                                                 edge_steps=5)

                # second run
                _a, _b, _c = OMMBV.closed_loop_edge_lengths_via_footpoint([p_lat]*len(p_longs), p_longs,
                                                                         p_alts, [date]*len(p_longs),
                                                                         direction,
                                                                         vector_direction,
                                                                         edge_length=25.,
                                                                         edge_steps=10)
                # take difference with first run
                mx[i, :-1] = (mx[i, :-1] - _a) / mx[i, :-1]
                my[i, :-1] = (my[i, :-1] - _b) / my[i, :-1]
                mz[i, :-1] = (mz[i, :-1] - _c) / mz[i, :-1]

        # account for periodicity
        mx[:, -1] = mx[:, 0]
        my[:, -1] = my[:, 0]
        mz[:, -1] = mz[:, 0]

        ytickarr = np.array([0, 0.25, 0.5, 0.75, 1])*(len(p_lats) - 1)
        xtickarr = np.array([0, 0.2, 0.4, 0.6, 0.8, 1])*len(p_longs)

        try:

            fig = plt.figure()
            plt.imshow(np.log10(np.abs(mx)), origin='lower')
            plt.colorbar()
            plt.yticks(ytickarr, ['-50', '-25', '0', '25', '50'])
            plt.xticks(xtickarr, ['0', '72', '144', '216', '288', '360'])
            plt.title('Log Closed Loop Edge Length Normalized Difference, Footpoint Path')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.savefig(direction + '_' + vector_direction + '_closed_loop_footpoint_edge_diff.pdf')
            plt.close()

            fig = plt.figure()
            plt.imshow(np.log10(np.abs(my)), origin='lower')
            plt.colorbar()
            plt.yticks(ytickarr, ['-50', '-25', '0', '25', '50'])
            plt.xticks(xtickarr, ['0', '72', '144', '216', '288', '360'])
            plt.title('Log Closed Loop Closest Approach Normalized Difference, Pos Footpoint Path')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.savefig(direction + '_' + vector_direction + '_closed_loop_footpoint_pos_diff.pdf')
            plt.close()

            fig = plt.figure()
            plt.imshow(np.log10(np.abs(mz)), origin='lower')
            plt.colorbar()
            plt.yticks(ytickarr, ['-50', '-25', '0', '25', '50'])
            plt.xticks(xtickarr, ['0', '72', '144', '216', '288', '360'])
            plt.title('Log Closed Loop Closest Approach Normalized Difference, Minus Footpoint Path')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.savefig(direction + '_' + vector_direction + '_closed_loop_footpoint_min_diff.pdf')
            plt.close()

            # calculate mean and standard deviation and then plot those
            plt.figure()
            yerrx = np.nanstd(np.log10(mx[:, :-1]), axis=0)
            yerry = np.nanstd(np.log10(my[:, :-1]), axis=0)
            yerrz = np.nanstd(np.log10(mz[:, :-1]), axis=0)
            plt.errorbar(p_longs, np.log10(np.nanmedian(np.abs(mx[:, :-1]), axis=0)),
                         yerr=yerrx, label='Edge')
            plt.errorbar(p_longs, np.log10(np.nanmedian(np.abs(my[:, :-1]), axis=0)),
                         yerr=yerry, label='Positive')
            plt.errorbar(p_longs, np.log10(np.nanmedian(np.abs(mz[:, :-1]), axis=0)),
                         yerr=yerrz, label='Minus')
            plt.xlabel('Longitude (Degrees)')
            plt.ylabel('Log Normalized Closed Loop Difference')
            plt.title("Sensitivity of Closed Loop Values")
            plt.legend()
            plt.tight_layout()
            plt.savefig(direction + '_' + vector_direction + '_length_diff_v_longitude.pdf')
            plt.close()

        except:
            pass

    # def test_closed_loop_footpoint_sensitivity_plots(self):
    #     f = functools.partial(self.closed_loop_footpoint_sensitivity_plots, 'north', 'meridional')
    #     yield(f,)
    #     f = functools.partial(self.closed_loop_footpoint_sensitivity_plots, 'south', 'meridional')
    #     yield(f,)
    #     f = functools.partial(self.closed_loop_footpoint_sensitivity_plots, 'north', 'zonal')
    #     yield(f,)
    #     f = functools.partial(self.closed_loop_footpoint_sensitivity_plots, 'south', 'zonal')
    #     yield(f,)

    def closed_loop_footpoint_value_plots(self, direction, vector_direction):
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
        #  meridional vector components
        mx = zvx.copy()
        my = zvx.copy()
        mz = zvx.copy()
        date = dt.datetime(2000, 1, 1)
        # set up multi
        if self.dc is not None:
            import itertools
            targets = itertools.cycle(dc.ids)
            pending = []
            for i, p_lat in enumerate(p_lats):
                # iterate through target cyclicly and run commands
                print (i, p_lat)
                dview.targets = next(targets)
                pending.append(
                    dview.apply_async(OMMBV.closed_loop_edge_lengths_via_footpoint,
                                      [p_lat]*len(p_longs), p_longs,
                                      p_alts, [date]*len(p_longs),
                                      direction,
                                      vector_direction,
                                      edge_length=25.,
                                      edge_steps=5))
            for i, p_lat in enumerate(p_lats):
                print ('collecting ', i, p_lat)
                # collect output from first run
                mx[i, :-1], my[i, :-1], mz[i, :-1] = pending.pop(0).get()

        else:
            for i, p_lat in enumerate(p_lats):
                print (i, p_lat)
                mx[i, :-1], my[i, :-1], mz[i, :-1] = OMMBV.closed_loop_edge_lengths_via_footpoint([p_lat]*len(p_longs),
                                                                                                 p_longs,
                                                                                                 p_alts,
                                                                                                 [date]*len(p_longs),
                                                                                                 direction,
                                                                                                 vector_direction,
                                                                                                 edge_length=25.,
                                                                                                 edge_steps=5)

        # account for periodicity
        mx[:, -1] = mx[:, 0]
        my[:, -1] = my[:, 0]
        mz[:, -1] = mz[:, 0]

        ytickarr = np.array([0, 0.25, 0.5, 0.75, 1])*(len(p_lats) - 1)
        xtickarr = np.array([0, 0.2, 0.4, 0.6, 0.8, 1])*len(p_longs)

        try:

            fig = plt.figure()
            plt.imshow(np.log10(np.abs(mx)), origin='lower')
            plt.colorbar()
            plt.yticks(ytickarr, ['-50', '-25', '0', '25', '50'])
            plt.xticks(xtickarr, ['0', '72', '144', '216', '288', '360'])
            plt.title('Log Closed Loop Edge Length, Footpoint Path')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.savefig(direction + '_' + vector_direction + '_closed_loop_footpoint_edge.pdf')
            plt.close()

            fig = plt.figure()
            plt.imshow(np.log10(np.abs(my)), origin='lower')
            plt.colorbar()
            plt.yticks(ytickarr, ['-50', '-25', '0', '25', '50'])
            plt.xticks(xtickarr, ['0', '72', '144', '216', '288', '360'])
            plt.title('Log Closed Loop Closest Approach, Pos Footpoint Path')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.savefig(direction + '_' + vector_direction + '_closed_loop_footpoint_pos.pdf')
            plt.close()

            fig = plt.figure()
            plt.imshow(np.log10(np.abs(mz)), origin='lower')
            plt.colorbar()
            plt.yticks(ytickarr, ['-50', '-25', '0', '25', '50'])
            plt.xticks(xtickarr, ['0', '72', '144', '216', '288', '360'])
            plt.title('Log Closed Loop Closest Approach, Minus Footpoint Path')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.savefig(direction + '_' + vector_direction + '_closed_loop_footpoint_min.pdf')
            plt.close()

            # calculate mean and standard deviation and then plot those
            plt.figure()
            plt.errorbar(p_longs, np.log10(np.nanmedian(np.abs(mx[:, :-1]), axis=0)),
                         yerr=np.abs(np.log10(mx.std(axis=0)) - np.log10(mx.median(axis=0))), label='Edge')
            plt.errorbar(p_longs, np.log10(np.nanmedian(np.abs(my[:, :-1]), axis=0)),
                         yerr=np.abs(np.log10(my.std(axis=0)) - np.log10(my.median(axis=0))), label='Positive')
            plt.errorbar(p_longs, np.log10(np.nanmedian(np.abs(my[:, :-1]), axis=0)),
                         yerr=np.abs(np.log10(mz.std(axis=0)) - np.log10(mz.median(axis=0))), label='Minus')
            plt.xlabel('Longitude (Degrees)')
            plt.ylabel('Log Closed Loop Values')
            plt.title("Closed Loop Values")
            plt.legend()
            plt.tight_layout()
            plt.savefig(direction + '_' + vector_direction + '_length_v_longitude.pdf')
            plt.close()

        except:
            pass

    # def test_closed_loop_footpoint_value_plots(self):
    #     f = functools.partial(self.closed_loop_footpoint_value_plots, 'north', 'meridional')
    #     yield(f,)
    #     f = functools.partial(self.closed_loop_footpoint_value_plots, 'south', 'meridional')
    #     yield(f,)
    #     f = functools.partial(self.closed_loop_footpoint_value_plots, 'north', 'zonal')
    #     yield(f,)
    #     f = functools.partial(self.closed_loop_footpoint_value_plots, 'south', 'zonal')
    #     yield(f,)

    def test_integrated_unit_vector_component_stepsize_sensitivity_plots(self):
        import matplotlib.pyplot as plt
        # from mpl_toolkits.mplot3d import Axes3D

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
            import itertools
            targets = itertools.cycle(dc.ids)
            pending = []
            for i, p_lat in enumerate(p_lats):
                # iterate through target cyclicly and run commands
                print (i, p_lat)
                dview.targets = next(targets)
                pending.append(
                    dview.apply_async(OMMBV.calculate_integrated_mag_drift_unit_vectors_ecef, [p_lat]*len(p_longs),
                                      p_longs,
                                      p_alts, [date]*len(p_longs),
                                      steps=None, max_steps=10000, step_size=10.,
                                      ref_height=120.))
                pending.append(
                    dview.apply_async(OMMBV.calculate_integrated_mag_drift_unit_vectors_ecef, [p_lat]*len(p_longs),
                                      p_longs,
                                      p_alts, [date]*len(p_longs),
                                      steps=None, max_steps=1000, step_size=100.,
                                      ref_height=120.))

            for i, p_lat in enumerate(p_lats):
                print ('collecting ', i, p_lat)
                # collect output from first run
                tzx, tzy, tzz, tbx, tby, tbz, tmx, tmy, tmz = pending.pop(0).get()
                zvx[i, :-1], zvy[i, :-1], zvz[i, :-1] = OMMBV.ecef_to_enu_vector(tzx, tzy, tzz,
                                                                                [p_lat]*len(p_longs),
                                                                                p_longs)
                bx[i, :-1], by[i, :-1], bz[i, :-1] = OMMBV.ecef_to_enu_vector(tbx, tby, tbz,
                                                                             [p_lat]*len(p_longs),
                                                                             p_longs)
                mx[i, :-1], my[i, :-1], mz[i, :-1] = OMMBV.ecef_to_enu_vector(tmx, tmy, tmz,
                                                                             [p_lat]*len(p_longs),
                                                                             p_longs)
                # collect output from second run
                tzx, tzy, tzz, tbx, tby, tbz, tmx, tmy, tmz = pending.pop(0).get()
                _a, _b, _c = OMMBV.ecef_to_enu_vector(tzx, tzy, tzz, [p_lat]*len(p_longs), p_longs)
                # take difference with first run
                zvx[i, :-1] = (zvx[i, :-1] - _a) / zvx[i, :-1]
                zvy[i, :-1] = (zvy[i, :-1] - _b) / zvy[i, :-1]
                zvz[i, :-1] = (zvz[i, :-1] - _c) / zvz[i, :-1]

                _a, _b, _c = OMMBV.ecef_to_enu_vector(tbx, tby, tbz, [p_lat]*len(p_longs), p_longs)
                # take difference with first run
                bx[i, :-1] = (bx[i, :-1] - _a) / bx[i, :-1]
                by[i, :-1] = (by[i, :-1] - _b) / by[i, :-1]
                bz[i, :-1] = (bz[i, :-1] - _c) / bz[i, :-1]

                _a, _b, _c = OMMBV.ecef_to_enu_vector(tmx, tmy, tmz, [p_lat]*len(p_longs), p_longs)
                # take difference with first run
                mx[i, :-1] = (mx[i, :-1] - _a) / mx[i, :-1]
                my[i, :-1] = (my[i, :-1] - _b) / my[i, :-1]
                mz[i, :-1] = (mz[i, :-1] - _c) / mz[i, :-1]

        else:
            for i, p_lat in enumerate(p_lats):
                print (i, p_lat)
                tzx, tzy, tzz, tbx, tby, tbz, tmx, tmy, tmz = OMMBV.calculate_integrated_mag_drift_unit_vectors_ecef(
                    [p_lat]*len(p_longs), p_longs,
                    p_alts, [date]*len(p_longs),
                    steps=None, max_steps=10000, step_size=10.,
                    ref_height=120.)
                zvx[i, :-1], zvy[i, :-1], zvz[i, :-1] = OMMBV.ecef_to_enu_vector(tzx, tzy, tzz,
                                                                                [p_lat]*len(p_longs),
                                                                                p_longs)
                bx[i, :-1], by[i, :-1], bz[i, :-1] = OMMBV.ecef_to_enu_vector(tbx, tby, tbz,
                                                                             [p_lat]*len(p_longs),
                                                                             p_longs)
                mx[i, :-1], my[i, :-1], mz[i, :-1] = OMMBV.ecef_to_enu_vector(tmx, tmy, tmz,
                                                                             [p_lat]*len(p_longs),
                                                                             p_longs)

                # second run
                tzx, tzy, tzz, tbx, tby, tbz, tmx, tmy, tmz = OMMBV.calculate_integrated_mag_drift_unit_vectors_ecef(
                    [p_lat]*len(p_longs), p_longs,
                    p_alts, [date]*len(p_longs),
                    steps=None, max_steps=1000, step_size=100.,
                    ref_height=120.)
                _a, _b, _c = OMMBV.ecef_to_enu_vector(tzx, tzy, tzz, [p_lat]*len(p_longs), p_longs)
                # take difference with first run
                zvx[i, :-1] = (zvx[i, :-1] - _a) / zvx[i, :-1]
                zvy[i, :-1] = (zvy[i, :-1] - _b) / zvy[i, :-1]
                zvz[i, :-1] = (zvz[i, :-1] - _c) / zvz[i, :-1]

                _a, _b, _c = OMMBV.ecef_to_enu_vector(tbx, tby, tbz, [p_lat]*len(p_longs), p_longs)
                # take difference with first run
                bx[i, :-1] = (bx[i, :-1] - _a) / bx[i, :-1]
                by[i, :-1] = (by[i, :-1] - _b) / by[i, :-1]
                bz[i, :-1] = (bz[i, :-1] - _c) / bz[i, :-1]

                _a, _b, _c = OMMBV.ecef_to_enu_vector(tmx, tmy, tmz, [p_lat]*len(p_longs), p_longs)
                # take difference with first run
                mx[i, :-1] = (mx[i, :-1] - _a) / mx[i, :-1]
                my[i, :-1] = (my[i, :-1] - _b) / my[i, :-1]
                mz[i, :-1] = (mz[i, :-1] - _c) / mz[i, :-1]

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

        ytickarr = np.array([0, 0.25, 0.5, 0.75, 1])*(len(p_lats) - 1)
        xtickarr = np.array([0, 0.2, 0.4, 0.6, 0.8, 1])*len(p_longs)

        try:
            fig = plt.figure()
            plt.imshow(np.log10(np.abs(zvx)), origin='lower')
            plt.colorbar()
            plt.yticks(ytickarr, ['-50', '-25', '0', '25', '50'])
            plt.xticks(xtickarr, ['0', '72', '144', '216', '288', '360'])
            plt.title('Log Zonal Unit Vector Nornalized Difference - Eastward')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.savefig('int_zonal_east_norm_diff.pdf')
            plt.close()

            fig = plt.figure()
            plt.imshow(np.log10(np.abs(zvy)), origin='lower')
            plt.colorbar()
            plt.yticks(ytickarr, ['-50', '-25', '0', '25', '50'])
            plt.xticks(xtickarr, ['0', '72', '144', '216', '288', '360'])
            plt.title('Log Zonal Unit Vector Nornalized Difference - Northward')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.savefig('int_zonal_north_norm_diff.pdf')
            plt.close()

            fig = plt.figure()
            plt.imshow(np.log10(np.abs(zvz)), origin='lower')
            plt.colorbar()
            plt.yticks(ytickarr, ['-50', '-25', '0', '25', '50'])
            plt.xticks(xtickarr, ['0', '72', '144', '216', '288', '360'])
            plt.title('Log Zonal Unit Vector Nornalized Difference - Upward')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.savefig('int_zonal_up_norm_diff.pdf')
            plt.close()

            fig = plt.figure()
            plt.imshow(np.log10(np.abs(bx)), origin='lower')
            plt.colorbar()
            plt.yticks(ytickarr, ['-50', '-25', '0', '25', '50'])
            plt.xticks(xtickarr, ['0', '72', '144', '216', '288', '360'])
            plt.title('Log Field Aligned Unit Vector Nornalized Difference - Eastward')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.savefig('int_fa_east_norm_diff.pdf')
            plt.close()

            fig = plt.figure()
            plt.imshow(np.log10(np.abs(by)), origin='lower')
            plt.colorbar()
            plt.yticks(ytickarr, ['-50', '-25', '0', '25', '50'])
            plt.xticks(xtickarr, ['0', '72', '144', '216', '288', '360'])
            plt.title('Log Field Aligned Unit Vector Nornalized Difference - Northward')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.savefig('int_fa_north_norm_diff.pdf')
            plt.close()

            fig = plt.figure()
            plt.imshow(np.log10(np.abs(bz)), origin='lower')
            plt.colorbar()
            plt.yticks(ytickarr, ['-50', '-25', '0', '25', '50'])
            plt.xticks(xtickarr, ['0', '72', '144', '216', '288', '360'])
            plt.title('Log Field Aligned Unit Vector Nornalized Difference - Upward')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.savefig('int_fa_up_norm_diff.pdf')
            plt.close()

            fig = plt.figure()
            plt.imshow(np.log10(np.abs(mx)), origin='lower')
            plt.colorbar()
            plt.yticks(ytickarr, ['-50', '-25', '0', '25', '50'])
            plt.xticks(xtickarr, ['0', '72', '144', '216', '288', '360'])
            plt.title('Log Meridional Unit Vector Nornalized Difference - Eastward')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.savefig('int_mer_east_norm_diff.pdf')
            plt.close()

            fig = plt.figure()
            plt.imshow(np.log10(np.abs(my)), origin='lower')
            plt.colorbar()
            plt.yticks(ytickarr, ['-50', '-25', '0', '25', '50'])
            plt.xticks(xtickarr, ['0', '72', '144', '216', '288', '360'])
            plt.title('Log Meridional Unit Vector Nornalized Difference - Northward')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.savefig('int_mer_north_norm_diff.pdf')
            plt.close()

            fig = plt.figure()
            plt.imshow(np.log10(np.abs(mz)), origin='lower')
            plt.colorbar()
            plt.yticks(ytickarr, ['-50', '-25', '0', '25', '50'])
            plt.xticks(xtickarr, ['0', '72', '144', '216', '288', '360'])
            plt.title('Log Meridional Unit Vector Nornalized Difference - Upward')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.savefig('int_mer_up_norm_diff.pdf')
            plt.close()

            # calculate mean and standard deviation and then plot those
            # print (p_longs)
            # print (np.nanmean(np.abs(zvx), axis=0))
            # print (np.nanstd(np.abs(zvx), axis=0))
            plt.figure()
            yerrx = np.nanstd(np.log10(zvx[:, :-1]), axis=0)
            yerry = np.nanstd(np.log10(zvy[:, :-1]), axis=0)
            yerrz = np.nanstd(np.log10(zvz[:, :-1]), axis=0)

            plt.errorbar(p_longs, np.log10(np.nanmedian(np.abs(zvx[:, :-1]), axis=0)),
                         yerr=yerrx, label='East')
            plt.errorbar(p_longs, np.log10(np.nanmedian(np.abs(zvy[:, :-1]), axis=0)),
                         yerr=yerry, label='North')
            plt.errorbar(p_longs, np.log10(np.nanmedian(np.abs(zvz[:, :-1]), axis=0)),
                         yerr=yerrz, label='Up')
            plt.xlabel('Longitude (Degrees)')
            plt.ylabel('Log Normalized Change in Zonal Vector')
            plt.title("Sensitivity of Zonal Unit Vector")
            plt.legend()
            plt.tight_layout()
            plt.savefig('int_zonal_diff_v_longitude.pdf')
            plt.close()

            # # calculate mean and standard deviation and then plot those
            # plt.figure()
            # plt.errorbar(p_longs, np.log10(np.nanmedian(np.abs(bx[:,:-1]), axis=0)),
            #                 yerr=np.nanstd(np.log10(np.abs(bx[:,:-1])), axis=0), label='East')
            # plt.errorbar(p_longs, np.log10(np.nanmedian(np.abs(by[:,:-1]), axis=0)),
            #                 yerr=np.nanstd(np.log10(np.abs(by[:,:-1])), axis=0), label='North')
            # plt.errorbar(p_longs, np.log10(np.nanmedian(np.abs(bz[:,:-1]), axis=0)),
            #                 yerr=np.nanstd(np.log10(np.abs(bz[:,:-1])), axis=0), label='Up')
            # plt.xlabel('Longitude (Degrees)')
            # plt.ylabel('Log Normalized Change in Field-Aligned Vector')
            # plt.title("Sensitivity of Field-Aligned Unit Vector")
            # plt.legend()
            # plt.tight_layout()
            # plt.savefig('int_fa_diff_v_longitude.pdf' )
            # plt.close()

            # calculate mean and standard deviation and then plot those
            plt.figure()
            yerrx = np.nanstd(np.log10(mx[:, :-1]), axis=0)
            yerry = np.nanstd(np.log10(my[:, :-1]), axis=0)
            yerrz = np.nanstd(np.log10(mz[:, :-1]), axis=0)
            plt.errorbar(p_longs, np.log10(np.nanmedian(np.abs(mx[:, :-1]), axis=0)),
                         yerr=yerrx, label='East')
            plt.errorbar(p_longs, np.log10(np.nanmedian(np.abs(my[:, :-1]), axis=0)),
                         yerr=yerry, label='North')
            plt.errorbar(p_longs, np.log10(np.nanmedian(np.abs(mz[:, :-1]), axis=0)),
                         yerr=yerrz, label='Up')
            plt.xlabel('Longitude (Degrees)')
            plt.ylabel('Log Normalized Change in Meridional Vector')
            plt.title("Sensitivity of Meridional Unit Vector")
            plt.legend()
            plt.tight_layout()
            plt.savefig('int_mer_diff_v_longitude.pdf')
            plt.close()

        except:
            pass

    def test_integrated_unit_vector_component_refheight_sensitivity_plots(self):
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
        zvy = zvx.copy();
        zvz = zvx.copy()
        #  meridional vecrtor components
        mx = zvx.copy();
        my = zvx.copy();
        mz = zvx.copy()
        # field aligned, along B
        bx = zvx.copy();
        by = zvx.copy();
        bz = zvx.copy()
        date = dt.datetime(2000, 1, 1)
        # set up multi
        if self.dc is not None:
            import itertools
            targets = itertools.cycle(dc.ids)
            pending = []
            for i, p_lat in enumerate(p_lats):
                # iterate through target cyclicly and run commands
                print (i, p_lat)
                dview.targets = next(targets)
                pending.append(
                    dview.apply_async(OMMBV.calculate_integrated_mag_drift_unit_vectors_ecef,
                                      [p_lat]*len(p_longs),
                                      p_longs,
                                      p_alts, [date]*len(p_longs),
                                      steps=None, max_steps=1000, step_size=10.,
                                      ref_height=240.))
                pending.append(
                    dview.apply_async(OMMBV.calculate_integrated_mag_drift_unit_vectors_ecef,
                                      [p_lat]*len(p_longs),
                                      p_longs,
                                      p_alts, [date]*len(p_longs),
                                      steps=None, max_steps=1000, step_size=10.,
                                      ref_height=0.))

            for i, p_lat in enumerate(p_lats):
                print ('collecting ', i, p_lat)
                # collect output from first run
                tzx, tzy, tzz, tbx, tby, tbz, tmx, tmy, tmz = pending.pop(0).get()
                zvx[i, :-1], zvy[i, :-1], zvz[i, :-1] = OMMBV.ecef_to_enu_vector(tzx, tzy, tzz,
                                                                                [p_lat]*len(p_longs),
                                                                                p_longs)
                bx[i, :-1], by[i, :-1], bz[i, :-1] = OMMBV.ecef_to_enu_vector(tbx, tby, tbz,
                                                                             [p_lat]*len(p_longs),
                                                                             p_longs)
                mx[i, :-1], my[i, :-1], mz[i, :-1] = OMMBV.ecef_to_enu_vector(tmx, tmy, tmz,
                                                                             [p_lat]*len(p_longs),
                                                                             p_longs)
                # collect output from second run
                tzx, tzy, tzz, tbx, tby, tbz, tmx, tmy, tmz = pending.pop(0).get()
                _a, _b, _c = OMMBV.ecef_to_enu_vector(tzx, tzy, tzz, [p_lat]*len(p_longs), p_longs)
                # take difference with first run
                zvx[i, :-1] = (zvx[i, :-1] - _a) / zvx[i, :-1]
                zvy[i, :-1] = (zvy[i, :-1] - _b) / zvy[i, :-1]
                zvz[i, :-1] = (zvz[i, :-1] - _c) / zvz[i, :-1]

                _a, _b, _c = OMMBV.ecef_to_enu_vector(tbx, tby, tbz, [p_lat]*len(p_longs), p_longs)
                # take difference with first run
                bx[i, :-1] = (bx[i, :-1] - _a) / bx[i, :-1]
                by[i, :-1] = (by[i, :-1] - _b) / by[i, :-1]
                bz[i, :-1] = (bz[i, :-1] - _c) / bz[i, :-1]

                _a, _b, _c = OMMBV.ecef_to_enu_vector(tmx, tmy, tmz, [p_lat]*len(p_longs), p_longs)
                # take difference with first run
                mx[i, :-1] = (mx[i, :-1] - _a) / mx[i, :-1]
                my[i, :-1] = (my[i, :-1] - _b) / my[i, :-1]
                mz[i, :-1] = (mz[i, :-1] - _c) / mz[i, :-1]

        else:
            for i, p_lat in enumerate(p_lats):
                print (i, p_lat)
                tzx, tzy, tzz, tbx, tby, tbz, tmx, tmy, tmz = OMMBV.calculate_integrated_mag_drift_unit_vectors_ecef(
                    [p_lat]*len(p_longs), p_longs,
                    p_alts, [date]*len(p_longs),
                    steps=None, max_steps=10000, step_size=10.,
                    ref_height=240.)
                zvx[i, :-1], zvy[i, :-1], zvz[i, :-1] = OMMBV.ecef_to_enu_vector(tzx, tzy, tzz,
                                                                                [p_lat]*len(p_longs),
                                                                                p_longs)
                bx[i, :-1], by[i, :-1], bz[i, :-1] = OMMBV.ecef_to_enu_vector(tbx, tby, tbz,
                                                                             [p_lat]*len(p_longs),
                                                                             p_longs)
                mx[i, :-1], my[i, :-1], mz[i, :-1] = OMMBV.ecef_to_enu_vector(tmx, tmy, tmz,
                                                                             [p_lat]*len(p_longs),
                                                                             p_longs)

                # second run
                tzx, tzy, tzz, tbx, tby, tbz, tmx, tmy, tmz = OMMBV.calculate_integrated_mag_drift_unit_vectors_ecef(
                    [p_lat]*len(p_longs), p_longs,
                    p_alts, [date]*len(p_longs),
                    steps=None, max_steps=10000, step_size=10.,
                    ref_height=0.)
                _a, _b, _c = OMMBV.ecef_to_enu_vector(tzx, tzy, tzz, [p_lat]*len(p_longs), p_longs)
                # take difference with first run
                zvx[i, :-1] = (zvx[i, :-1] - _a) / zvx[i, :-1]
                zvy[i, :-1] = (zvy[i, :-1] - _b) / zvy[i, :-1]
                zvz[i, :-1] = (zvz[i, :-1] - _c) / zvz[i, :-1]

                _a, _b, _c = OMMBV.ecef_to_enu_vector(tbx, tby, tbz, [p_lat]*len(p_longs), p_longs)
                # take difference with first run
                bx[i, :-1] = (bx[i, :-1] - _a) / bx[i, :-1]
                by[i, :-1] = (by[i, :-1] - _b) / by[i, :-1]
                bz[i, :-1] = (bz[i, :-1] - _c) / bz[i, :-1]

                _a, _b, _c = OMMBV.ecef_to_enu_vector(tmx, tmy, tmz, [p_lat]*len(p_longs), p_longs)
                # take difference with first run
                mx[i, :-1] = (mx[i, :-1] - _a) / mx[i, :-1]
                my[i, :-1] = (my[i, :-1] - _b) / my[i, :-1]
                mz[i, :-1] = (mz[i, :-1] - _c) / mz[i, :-1]

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

        ytickarr = np.array([0, 0.25, 0.5, 0.75, 1])*(len(p_lats) - 1)
        xtickarr = np.array([0, 0.2, 0.4, 0.6, 0.8, 1])*len(p_longs)

        try:
            fig = plt.figure()
            plt.imshow(np.log10(np.abs(zvx)), origin='lower')
            plt.colorbar()
            plt.yticks(ytickarr, ['-50', '-25', '0', '25', '50'])
            plt.xticks(xtickarr, ['0', '72', '144', '216', '288', '360'])
            plt.title('Log Zonal Unit Vector Nornalized Difference - Eastward')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.savefig('int_zonal_east_diff_height.pdf')
            plt.close()

            fig = plt.figure()
            plt.imshow(np.log10(np.abs(zvy)), origin='lower')
            plt.colorbar()
            plt.yticks(ytickarr, ['-50', '-25', '0', '25', '50'])
            plt.xticks(xtickarr, ['0', '72', '144', '216', '288', '360'])
            plt.title('Log Zonal Unit Vector Nornalized Difference - Northward')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.savefig('int_zonal_north_diff_height.pdf')
            plt.close()

            fig = plt.figure()
            plt.imshow(np.log10(np.abs(zvz)), origin='lower')
            plt.colorbar()
            plt.yticks(ytickarr, ['-50', '-25', '0', '25', '50'])
            plt.xticks(xtickarr, ['0', '72', '144', '216', '288', '360'])
            plt.title('Log Zonal Unit Vector Nornalized Difference - Upward')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.savefig('int_zonal_up_diff_height.pdf')
            plt.close()

            fig = plt.figure()
            plt.imshow(np.log10(np.abs(bx)), origin='lower')
            plt.colorbar()
            plt.yticks(ytickarr, ['-50', '-25', '0', '25', '50'])
            plt.xticks(xtickarr, ['0', '72', '144', '216', '288', '360'])
            plt.title('Log Field Aligned Unit Vector Nornalized Difference - Eastward')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.savefig('int_fa_east_diff_height.pdf')
            plt.close()

            fig = plt.figure()
            plt.imshow(np.log10(np.abs(by)), origin='lower')
            plt.colorbar()
            plt.yticks(ytickarr, ['-50', '-25', '0', '25', '50'])
            plt.xticks(xtickarr, ['0', '72', '144', '216', '288', '360'])
            plt.title('Log Field Aligned Unit Vector Nornalized Difference - Northward')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.savefig('int_fa_north_diff_height.pdf')
            plt.close()

            fig = plt.figure()
            plt.imshow(np.log10(np.abs(bz)), origin='lower')
            plt.colorbar()
            plt.yticks(ytickarr, ['-50', '-25', '0', '25', '50'])
            plt.xticks(xtickarr, ['0', '72', '144', '216', '288', '360'])
            plt.title('Log Field Aligned Unit Vector Nornalized Difference - Upward')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.savefig('int_fa_up_diff_height.pdf')
            plt.close()

            fig = plt.figure()
            plt.imshow(np.log10(np.abs(mx)), origin='lower')
            plt.colorbar()
            plt.yticks(ytickarr, ['-50', '-25', '0', '25', '50'])
            plt.xticks(xtickarr, ['0', '72', '144', '216', '288', '360'])
            plt.title('Log Meridional Unit Vector Nornalized Difference - Eastward')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.savefig('int_mer_east_diff_height.pdf')
            plt.close()

            fig = plt.figure()
            plt.imshow(np.log10(np.abs(my)), origin='lower')
            plt.colorbar()
            plt.yticks(ytickarr, ['-50', '-25', '0', '25', '50'])
            plt.xticks(xtickarr, ['0', '72', '144', '216', '288', '360'])
            plt.title('Log Meridional Unit Vector Nornalized Difference - Northward')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.savefig('int_mer_north_diff_height.pdf')
            plt.close()

            fig = plt.figure()
            plt.imshow(np.log10(np.abs(mz)), origin='lower')
            plt.colorbar()
            plt.yticks(ytickarr, ['-50', '-25', '0', '25', '50'])
            plt.xticks(xtickarr, ['0', '72', '144', '216', '288', '360'])
            plt.title('Log Meridional Unit Vector Nornalized Difference - Upward')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.savefig('int_mer_up_diff_height.pdf')
            plt.close()

            # calculate mean and standard deviation and then plot those
            fig = plt.figure()
            yerrx = np.nanstd(np.log10(zvx[:, :-1]), axis=0)
            yerry = np.nanstd(np.log10(zvy[:, :-1]), axis=0)
            yerrz = np.nanstd(np.log10(zvz[:, :-1]), axis=0)
            plt.errorbar(p_longs, np.log10(np.nanmedian(np.abs(zvx[:, :-1]), axis=0)),
                         yerr=yerrx, label='East')
            plt.errorbar(p_longs, np.log10(np.nanmedian(np.abs(zvy[:, :-1]), axis=0)),
                         yerr=yerry, label='North')
            plt.errorbar(p_longs, np.log10(np.nanmedian(np.abs(zvz[:, :-1]), axis=0)),
                         yerr=yerrz, label='Up')
            plt.xlabel('Longitude (Degrees)')
            plt.ylabel('Log Normalized Change in Zonal Vector')
            plt.title("Sensitivity of Zonal Unit Vector")
            plt.legend()
            plt.tight_layout()
            plt.savefig('int_zonal_diff_v_longitude_height.pdf')
            plt.close()

            # calculate mean and standard deviation and then plot those
            fig = plt.figure()
            yerrx = np.nanstd(np.log10(mx[:, :-1]), axis=0)
            yerry = np.nanstd(np.log10(my[:, :-1]), axis=0)
            yerrz = np.nanstd(np.log10(mz[:, :-1]), axis=0)
            plt.errorbar(p_longs, np.log10(np.nanmedian(np.abs(mx[:, :-1]), axis=0)),
                         yerr=yerrx, label='East')
            plt.errorbar(p_longs, np.log10(np.nanmedian(np.abs(my[:, :-1]), axis=0)),
                         yerr=yerry, label='North')
            plt.errorbar(p_longs, np.log10(np.nanmedian(np.abs(mz[:, :-1]), axis=0)),
                         yerr=yerrz, label='Up')
            plt.xlabel('Longitude (Degrees)')
            plt.ylabel('Log Normalized Change in Meridional Vector')
            plt.title("Sensitivity of Meridional Unit Vector")
            plt.legend()
            plt.tight_layout()
            plt.savefig('int_mer_diff_v_longitude_height.pdf')
            plt.close()

            # # calculate mean and standard deviation and then plot those
            # fig = plt.figure()
            # plt.errorbar(p_longs, np.log10(np.nanmedian(np.abs(bx[:,:-1]), axis=0)),
            #                 yerr=np.nanstd(np.log10(np.abs(bx[:,:-1])), axis=0), label='East')
            # plt.errorbar(p_longs, np.log10(np.nanmedian(np.abs(by[:,:-1]), axis=0)),
            #                 yerr=np.nanstd(np.log10(np.abs(by[:,:-1])), axis=0), label='North')
            # plt.errorbar(p_longs, np.log10(np.nanmedian(np.abs(bz[:,:-1]), axis=0)),
            #                 yerr=np.nanstd(np.log10(np.abs(bz[:,:-1])), axis=0), label='Up')
            # plt.xlabel('Longitude (Degrees)')
            # plt.ylabel('Log Normalized Change in Field-Aligned Vector')
            # plt.title("Sensitivity of Field-Aligned Unit Vector")
            # plt.legend()
            # plt.tight_layout()
            # plt.savefig('int_fa_diff_v_longitude_height.pdf' )
            # plt.close()

        except:
            pass

    def test_vector_method_sensitivity_plots(self):
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
        zvy = zvx.copy();
        zvz = zvx.copy()
        #  meridional vecrtor components
        mx = zvx.copy();
        my = zvx.copy();
        mz = zvx.copy()
        # field aligned, along B
        bx = zvx.copy();
        by = zvx.copy();
        bz = zvx.copy()
        date = dt.datetime(2000, 1, 1)
        # set up multi
        if self.dc is not None:
            import itertools
            targets = itertools.cycle(dc.ids)
            pending = []
            for i, p_lat in enumerate(p_lats):
                # iterate through target cyclicly and run commands
                print (i, p_lat)
                dview.targets = next(targets)
                pending.append(
                    dview.apply_async(OMMBV.calculate_integrated_mag_drift_unit_vectors_ecef,
                                      [p_lat]*len(p_longs),
                                      p_longs,
                                      p_alts, [date]*len(p_longs)))
                pending.append(
                    dview.apply_async(OMMBV.calculate_mag_drift_unit_vectors_ecef,
                                      [p_lat]*len(p_longs), p_longs,
                                      p_alts, [date]*len(p_longs)))

            for i, p_lat in enumerate(p_lats):
                print ('collecting ', i, p_lat)
                # collect output from first run
                tzx, tzy, tzz, tbx, tby, tbz, tmx, tmy, tmz = pending.pop(0).get()
                zvx[i, :-1], zvy[i, :-1], zvz[i, :-1] = OMMBV.ecef_to_enu_vector(tzx, tzy, tzz,
                                                                                [p_lat]*len(p_longs),
                                                                                p_longs)
                bx[i, :-1], by[i, :-1], bz[i, :-1] = OMMBV.ecef_to_enu_vector(tbx, tby, tbz,
                                                                             [p_lat]*len(p_longs),
                                                                             p_longs)
                mx[i, :-1], my[i, :-1], mz[i, :-1] = OMMBV.ecef_to_enu_vector(tmx, tmy, tmz,
                                                                             [p_lat]*len(p_longs),
                                                                             p_longs)
                # collect output from second run
                tzx, tzy, tzz, tbx, tby, tbz, tmx, tmy, tmz = pending.pop(0).get()
                _a, _b, _c = OMMBV.ecef_to_enu_vector(tzx, tzy, tzz, [p_lat]*len(p_longs), p_longs)
                # take difference with first run
                zvx[i, :-1] = (zvx[i, :-1] - _a)  # /zvx[i,:-1]
                zvy[i, :-1] = (zvy[i, :-1] - _b)  # /zvy[i,:-1]
                zvz[i, :-1] = (zvz[i, :-1] - _c)  # /zvz[i,:-1]

                _a, _b, _c = OMMBV.ecef_to_enu_vector(tbx, tby, tbz, [p_lat]*len(p_longs), p_longs)
                # take difference with first run
                bx[i, :-1] = (bx[i, :-1] - _a)  # /bx[i,:-1]
                by[i, :-1] = (by[i, :-1] - _b)  # /by[i,:-1]
                bz[i, :-1] = (bz[i, :-1] - _c)  # /bz[i,:-1]

                _a, _b, _c = OMMBV.ecef_to_enu_vector(tmx, tmy, tmz, [p_lat]*len(p_longs), p_longs)
                # take difference with first run
                mx[i, :-1] = (mx[i, :-1] - _a)  # /mx[i,:-1]
                my[i, :-1] = (my[i, :-1] - _b)  # /my[i,:-1]
                mz[i, :-1] = (mz[i, :-1] - _c)  # /mz[i,:-1]

        else:
            for i, p_lat in enumerate(p_lats):
                print (i, p_lat)
                tzx, tzy, tzz, tbx, tby, tbz, tmx, tmy, tmz = OMMBV.calculate_integrated_mag_drift_unit_vectors_ecef(
                    [p_lat]*len(p_longs), p_longs,
                    p_alts, [date]*len(p_longs))
                zvx[i, :-1], zvy[i, :-1], zvz[i, :-1] = OMMBV.ecef_to_enu_vector(tzx, tzy, tzz,
                                                                                [p_lat]*len(p_longs),
                                                                                p_longs)
                bx[i, :-1], by[i, :-1], bz[i, :-1] = OMMBV.ecef_to_enu_vector(tbx, tby, tbz,
                                                                             [p_lat]*len(p_longs),
                                                                             p_longs)
                mx[i, :-1], my[i, :-1], mz[i, :-1] = OMMBV.ecef_to_enu_vector(tmx, tmy, tmz,
                                                                             [p_lat]*len(p_longs),
                                                                             p_longs)

                # second run
                tzx, tzy, tzz, tbx, tby, tbz, tmx, tmy, tmz = OMMBV.calculate_mag_drift_unit_vectors_ecef(
                    [p_lat]*len(p_longs), p_longs,
                    p_alts, [date]*len(p_longs))
                _a, _b, _c = OMMBV.ecef_to_enu_vector(tzx, tzy, tzz, [p_lat]*len(p_longs), p_longs)
                # take difference with first run
                zvx[i, :-1] = (zvx[i, :-1] - _a)  # /zvx[i,:-1]
                zvy[i, :-1] = (zvy[i, :-1] - _b)  # /zvy[i,:-1]
                zvz[i, :-1] = (zvz[i, :-1] - _c)  # /zvz[i,:-1]

                _a, _b, _c = OMMBV.ecef_to_enu_vector(tbx, tby, tbz, [p_lat]*len(p_longs), p_longs)
                # take difference with first run
                bx[i, :-1] = (bx[i, :-1] - _a)  # /bx[i,:-1]
                by[i, :-1] = (by[i, :-1] - _b)  # /by[i,:-1]
                bz[i, :-1] = (bz[i, :-1] - _c)  # /bz[i,:-1]

                _a, _b, _c = OMMBV.ecef_to_enu_vector(tmx, tmy, tmz, [p_lat]*len(p_longs), p_longs)
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
            plt.savefig('method_zonal_east_diff.pdf')
            plt.close()

            fig = plt.figure()
            plt.imshow(np.log10(np.abs(zvy)), origin='lower')
            plt.colorbar()
            plt.yticks(ytickarr, ['-50', '-25', '0', '25', '50'])
            plt.xticks(xtickarr, ['0', '72', '144', '216', '288', '360'])
            plt.title('Log Zonal Unit Vector Difference - Northward')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.savefig('method_zonal_north_diff.pdf')
            plt.close()

            fig = plt.figure()
            plt.imshow(np.log10(np.abs(zvz)), origin='lower')
            plt.colorbar()
            plt.yticks(ytickarr, ['-50', '-25', '0', '25', '50'])
            plt.xticks(xtickarr, ['0', '72', '144', '216', '288', '360'])
            plt.title('Log Zonal Unit Vector Difference - Upward')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.savefig('method_zonal_up_diff.pdf')
            plt.close()

            fig = plt.figure()
            plt.imshow(np.log10(np.abs(bx)), origin='lower')
            plt.colorbar()
            plt.yticks(ytickarr, ['-50', '-25', '0', '25', '50'])
            plt.xticks(xtickarr, ['0', '72', '144', '216', '288', '360'])
            plt.title('Log Field Aligned Unit Vector Difference - Eastward')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.savefig('method_fa_east_diff.pdf')
            plt.close()

            fig = plt.figure()
            plt.imshow(np.log10(np.abs(by)), origin='lower')
            plt.colorbar()
            plt.yticks(ytickarr, ['-50', '-25', '0', '25', '50'])
            plt.xticks(xtickarr, ['0', '72', '144', '216', '288', '360'])
            plt.title('Log Field Aligned Unit Vector Difference - Northward')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.savefig('method_fa_north_diff.pdf')
            plt.close()

            fig = plt.figure()
            plt.imshow(np.log10(np.abs(bz)), origin='lower')
            plt.colorbar()
            plt.yticks(ytickarr, ['-50', '-25', '0', '25', '50'])
            plt.xticks(xtickarr, ['0', '72', '144', '216', '288', '360'])
            plt.title('Log Field Aligned Unit Vector Difference - Upward')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.savefig('method_fa_up_diff.pdf')
            plt.close()

            fig = plt.figure()
            plt.imshow(np.log10(np.abs(mx)), origin='lower')
            plt.colorbar()
            plt.yticks(ytickarr, ['-50', '-25', '0', '25', '50'])
            plt.xticks(xtickarr, ['0', '72', '144', '216', '288', '360'])
            plt.title('Log Meridional Unit Vector Difference - Eastward')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.savefig('method_mer_east_diff.pdf')
            plt.close()

            fig = plt.figure()
            plt.imshow(np.log10(np.abs(my)), origin='lower')
            plt.colorbar()
            plt.yticks(ytickarr, ['-50', '-25', '0', '25', '50'])
            plt.xticks(xtickarr, ['0', '72', '144', '216', '288', '360'])
            plt.title('Log Meridional Unit Vector Difference - Northward')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.savefig('method_mer_north_diff.pdf')
            plt.close()

            fig = plt.figure()
            plt.imshow(np.log10(np.abs(mz)), origin='lower')
            plt.colorbar()
            plt.yticks(ytickarr, ['-50', '-25', '0', '25', '50'])
            plt.xticks(xtickarr, ['0', '72', '144', '216', '288', '360'])
            plt.title('Log Meridional Unit Vector Difference - Upward')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.savefig('method_mer_up_diff.pdf')
            plt.close()

            # calculate mean and standard deviation and then plot those
            # print (p_longs)
            # print (np.nanmean(np.abs(zvx), axis=0))
            # print (np.nanstd(np.abs(zvx), axis=0))
            plt.figure()
            yerrx = np.nanstd(np.log10(zvx[:, :-1]), axis=0)
            yerry = np.nanstd(np.log10(zvy[:, :-1]), axis=0)
            yerrz = np.nanstd(np.log10(zvz[:, :-1]), axis=0)
            plt.errorbar(p_longs, np.log10(np.nanmedian(np.abs(zvx[:, :-1]), axis=0)),
                         yerr=yerrx, label='East')
            plt.errorbar(p_longs, np.log10(np.nanmedian(np.abs(zvy[:, :-1]), axis=0)),
                         yerr=yerry, label='North')
            plt.errorbar(p_longs, np.log10(np.nanmedian(np.abs(zvz[:, :-1]), axis=0)),
                         yerr=yerrz, label='Up')
            plt.xlabel('Longitude (Degrees)')
            plt.ylabel('Log Change in Zonal Vector')
            plt.title("Sensitivity of Zonal Unit Vector")
            plt.legend()
            plt.tight_layout()
            plt.savefig('method_zonal_diff_v_longitude.pdf')
            plt.close()

            # # calculate mean and standard deviation and then plot those
            # plt.figure()
            # plt.errorbar(p_longs, np.log10(np.nanmedian(np.abs(bx[:,:-1]), axis=0)),
            #                 yerr=np.nanstd(np.log10(np.abs(bx[:,:-1])), axis=0), label='East')
            # plt.errorbar(p_longs, np.log10(np.nanmedian(np.abs(by[:,:-1]), axis=0)),
            #                 yerr=np.nanstd(np.log10(np.abs(by[:,:-1])), axis=0), label='North')
            # plt.errorbar(p_longs, np.log10(np.nanmedian(np.abs(bz[:,:-1]), axis=0)),
            #                 yerr=np.nanstd(np.log10(np.abs(bz[:,:-1])), axis=0), label='Up')
            # plt.xlabel('Longitude (Degrees)')
            # plt.ylabel('Log Change in Field-Aligned Vector')
            # plt.title("Sensitivity of Field-Aligned Unit Vector")
            # plt.legend()
            # plt.tight_layout()
            # plt.savefig('method_fa_diff_v_longitude.pdf' )
            # plt.close()

            # calculate mean and standard deviation and then plot those
            plt.figure()
            yerrx = np.nanstd(np.log10(mx[:, :-1]), axis=0)
            yerry = np.nanstd(np.log10(my[:, :-1]), axis=0)
            yerrz = np.nanstd(np.log10(mz[:, :-1]), axis=0)
            plt.errorbar(p_longs, np.log10(np.nanmedian(np.abs(mx[:, :-1]), axis=0)),
                         yerr=yerrx, label='East')
            plt.errorbar(p_longs, np.log10(np.nanmedian(np.abs(my[:, :-1]), axis=0)),
                         yerr=yerry, label='North')
            plt.errorbar(p_longs, np.log10(np.nanmedian(np.abs(mz[:, :-1]), axis=0)),
                         yerr=yerrz, label='Up')
            plt.xlabel('Longitude (Degrees)')
            plt.ylabel('Log Change in Meridional Vector')
            plt.title("Sensitivity of Meridional Unit Vector")
            plt.legend()
            plt.tight_layout()
            plt.savefig('method_mer_diff_v_longitude.pdf')
            plt.close()

        except:
            pass
