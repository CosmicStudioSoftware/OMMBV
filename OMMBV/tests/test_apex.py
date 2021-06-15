import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pds

import OMMBV
import pysat

from OMMBV.tests.test_core import gen_data_fixed_alt, gen_trace_data_fixed_alt
from OMMBV.tests.test_core import gen_plot_grid_fixed_alt

from OMMBV.tests.test_core import dview, dc


class TestMaxApexHeight():

    def test_plot_apex_heights(self):
        """Check meridional vector along max in apex height gradient"""
        date = dt.datetime(2010, 1, 1)

        delta = 1.

        ecef_x, ecef_y, ecef_z = OMMBV.geodetic_to_ecef([0.], [320.], [550.])

        # get basis vectors
        zx, zy, zz, _, _, _, mx, my, mz = OMMBV.calculate_mag_drift_unit_vectors_ecef(ecef_x, ecef_y, ecef_z,
                                                                                     [date], ecef_input=True)

        # get apex height for step along meridional directions, then around that direction
        _, _, _, _, _, nominal_max = OMMBV.apex_location_info(ecef_x + delta * mx,
                                                              ecef_y + delta * my,
                                                              ecef_z + delta * mz,
                                                              [date],
                                                              ecef_input=True,
                                                              return_geodetic=True)

        steps = (np.arange(101) - 50.) * delta / 10000.
        output_max = []
        for step in steps:
            del_x = delta * mx + step * zx
            del_y = delta * my + step * zy
            del_z = delta * mz + step * zz
            norm = np.sqrt(del_x ** 2 + del_y ** 2 + del_z ** 2)
            del_x /= norm
            del_y /= norm
            del_z /= norm
            _, _, _, _, _, loop_h = OMMBV.apex_location_info(ecef_x + del_x,
                                                             ecef_y + del_y,
                                                             ecef_z + del_z,
                                                             [date],
                                                             ecef_input=True,
                                                             return_geodetic=True)
            output_max.append(loop_h)

        try:
            plt.figure()
            plt.plot(steps, output_max)
            plt.plot([0], nominal_max, color='r', marker='o', markersize=12)
            plt.ylabel('Apex Height (km)')
            plt.xlabel('Distance along Zonal Direction (km)')
            plt.savefig('comparison_apex_heights_and_meridional.pdf')
            plt.close()
        except:
            pass

        # make sure meridional direction is correct
        assert np.all(np.max(output_max) == nominal_max)


class TestApex():

    def __init__(self):
        # placeholder for data management features
        self.inst = pysat.Instrument('pysat', 'testing')
        self.inst.yr = 2010.
        self.inst.doy = 1.
        self.dview = dview
        self.dc = dc

        return

    def test_apex_info_accuracy(self):
        """Characterize performance of apex_location_info as fine_step_size varied"""
        lats, longs, alts = gen_trace_data_fixed_alt(550.)
        ecf_x, ecf_y, ecf_z = OMMBV.geodetic_to_ecef(lats,
                                                     longs,
                                                     alts)
        # step size to be tried
        fine_steps_goal = np.array([25.6, 12.8, 6.4, 3.2, 1.6, 0.8, 0.4, 0.2,
                                    0.1, 0.05, .025, .0125, .00625, .003125,
                                    .0015625, .00078125, .000390625, .0001953125,
                                    .0001953125 / 2., .0001953125 / 4., .0001953125 / 8.,
                                    .0001953125 / 16., .0001953125 / 32., .0001953125 / 64.,
                                    .0001953125 / 128., .0001953125 / 256., .0001953125 / 512.,
                                    .0001953125 / 1024., .0001953125 / 2048., .0001953125 / 4096.])

        date = dt.datetime(2000, 1, 1)
        dx = []
        dy = []
        dz = []
        dh = []

        # set up multi
        if self.dc is not None:
            import itertools
            targets = itertools.cycle(dc.ids)
            pending = []
            for lat, lon, alt in zip(lats, longs, alts):
                for steps in fine_steps_goal:
                    # iterate through target cyclicly and run commands
                    dview.targets = next(targets)
                    pending.append(dview.apply_async(OMMBV.apex_location_info, [lat],
                                                     [lon], [alt], [date], fine_step_size=steps,
                                                     return_geodetic=True))
                out = []
                for steps in fine_steps_goal:
                    # collect output
                    x, y, z, _, _, apex_height = pending.pop(0).get()
                    pt = [x[0], y[0], z[0], apex_height[0]]
                    out.append(pt)

                final_pt = pds.DataFrame(out, columns=['x', 'y', 'z', 'h'])
                dx.append(np.abs(final_pt.loc[1:, 'x'].values - final_pt.loc[:, 'x'].values[:-1]))
                dy.append(np.abs(final_pt.loc[1:, 'y'].values - final_pt.loc[:, 'y'].values[:-1]))
                dz.append(np.abs(final_pt.loc[1:, 'z'].values - final_pt.loc[:, 'z'].values[:-1]))
                dh.append(np.abs(final_pt.loc[1:, 'h'].values - final_pt.loc[:, 'h'].values[:-1]))
        else:
            for lat, lon, alt in zip(lats, longs, alts):
                out = []
                for steps in fine_steps_goal:
                    x, y, z, _, _, apex_height = OMMBV.apex_location_info([lat], [lon], [alt], [date],
                                                                          fine_step_size=steps,
                                                                          return_geodetic=True)
                    pt = [x[0], y[0], z[0], apex_height[0]]
                    out.append(pt)

                final_pt = pds.DataFrame(out, columns=['x', 'y', 'z', 'h'])
                dx.append(np.abs(final_pt.loc[1:, 'x'].values - final_pt.loc[:, 'x'].values[:-1]))
                dy.append(np.abs(final_pt.loc[1:, 'y'].values - final_pt.loc[:, 'y'].values[:-1]))
                dz.append(np.abs(final_pt.loc[1:, 'z'].values - final_pt.loc[:, 'z'].values[:-1]))
                dh.append(np.abs(final_pt.loc[1:, 'h'].values - final_pt.loc[:, 'h'].values[:-1]))

        dx = pds.DataFrame(dx)
        dy = pds.DataFrame(dy)
        dz = pds.DataFrame(dz)
        dh = pds.DataFrame(dh)

        try:

            plt.figure()
            yerrx = np.nanstd(np.log10(dx), axis=0)
            yerry = np.nanstd(np.log10(dy), axis=0)
            yerrz = np.nanstd(np.log10(dz), axis=0)
            yerrh = np.nanstd(np.log10(dh), axis=0)

            plt.errorbar(np.log10(fine_steps_goal[1:]), np.log10(dx.mean(axis=0)),
                         yerr=yerrx,
                         label='x')
            plt.errorbar(np.log10(fine_steps_goal[1:]), np.log10(dy.mean(axis=0)),
                         yerr=yerry,
                         label='y')
            plt.errorbar(np.log10(fine_steps_goal[1:]), np.log10(dz.mean(axis=0)),
                         yerr=yerrz,
                         label='z')
            plt.errorbar(np.log10(fine_steps_goal[1:]), np.log10(dh.mean(axis=0)),
                         yerr=yerrh,
                         label='h')

            plt.xlabel('Log Step Size (km)')
            plt.ylabel('Change in Apex Position (km)')
            plt.title("Change in Field Apex Position vs Fine Step Size")
            plt.legend()
            plt.tight_layout()
            plt.savefig('apex_location_vs_step_size.pdf')
            plt.close()
        except:
            pass

    def test_apex_plots(self):
        """Plot basic apex parameters"""
        import matplotlib.pyplot as plt

        p_lats, p_longs, p_alts = gen_plot_grid_fixed_alt(120.)
        # data returned are the locations along each direction
        # the full range of points obtained by iterating over all
        # recasting alts into a more convenient form for later calculation
        p_alts = [p_alts[0]] * len(p_longs)
        # set the date
        date = dt.datetime(2000, 1, 1)
        # memory for results
        apex_lat = np.zeros((len(p_lats), len(p_longs) + 1))
        apex_lon = np.zeros((len(p_lats), len(p_longs) + 1))
        apex_alt = np.zeros((len(p_lats), len(p_longs) + 1))

        # set up multi
        if self.dc is not None:
            import itertools
            targets = itertools.cycle(dc.ids)
            pending = []
            for i, p_lat in enumerate(p_lats):
                print (i, p_lat)
                # iterate through target cyclicly and run commands
                dview.targets = next(targets)
                pending.append(dview.apply_async(OMMBV.apex_location_info, [p_lat] * len(p_longs), p_longs,
                                                 p_alts, [date] * len(p_longs),
                                                 return_geodetic=True))
            for i, p_lat in enumerate(p_lats):
                print ('collecting ', i, p_lat)
                # collect output
                x, y, z, olat, olon, oalt = pending.pop(0).get()
                apex_lat[i, :-1] = olat
                apex_lon[i, :-1] = olon
                apex_alt[i, :-1] = oalt

        else:
            # single processor case
            for i, p_lat in enumerate(p_lats):
                print (i, p_lat)
                x, y, z, olat, olon, oalt = OMMBV.apex_location_info([p_lat] * len(p_longs), p_longs,
                                                                     p_alts, [date] * len(p_longs),
                                                                     return_geodetic=True)
                apex_lat[i, :-1] = olat
                apex_lon[i, :-1] = olon
                apex_alt[i, :-1] = oalt

        # calculate difference between apex longitude and original longitude
        # values for apex long are -180 to 180, shift to 0 to 360
        # process degrees a bit to make the degree difference the most meaningful (close to 0)
        idx, idy, = np.where(apex_lon < 0.)
        apex_lon[idx, idy] += 360.
        idx, idy, = np.where(apex_lon >= 360.)
        apex_lon[idx, idy] -= 360.
        apex_lon[:, :-1] -= p_longs
        idx, idy, = np.where(apex_lon > 180.)
        apex_lon[idx, idy] -= 360.
        idx, idy, = np.where(apex_lon <= -180.)
        apex_lon[idx, idy] += 360.

        # account for periodicity
        apex_lat[:, -1] = apex_lat[:, 0]
        apex_lon[:, -1] = apex_lon[:, 0]
        apex_alt[:, -1] = apex_alt[:, 0]

        ytickarr = np.array([0, 0.25, 0.5, 0.75, 1]) * (len(p_lats) - 1)
        xtickarr = np.array([0, 0.2, 0.4, 0.6, 0.8, 1]) * len(p_longs)
        ytickvals = ['-25', '-12.5', '0', '12.5', '25']

        try:
            fig = plt.figure()
            plt.imshow(apex_lat, origin='lower')
            plt.colorbar()
            plt.yticks(ytickarr, ytickvals)
            plt.xticks(xtickarr, ['0', '72', '144', '216', '288', '360'])
            plt.title('Apex Latitude (Degrees) at 120 km')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.savefig('apex_lat.pdf')
            plt.close()

            fig = plt.figure()
            plt.imshow(apex_lon, origin='lower')
            plt.colorbar()
            plt.yticks(ytickarr, ytickvals)
            plt.xticks(xtickarr, ['0', '72', '144', '216', '288', '360'])
            plt.title('Apex Longitude Difference (Degrees) at 120 km')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.savefig('apex_lon.pdf')
            plt.close()

            fig = plt.figure()
            plt.imshow(np.log10(apex_alt), origin='lower')
            plt.colorbar()
            plt.yticks(ytickarr, ytickvals)
            plt.xticks(xtickarr, ['0', '72', '144', '216', '288', '360'])
            plt.title('Log Apex Altitude (km) at 120 km')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.savefig('apex_alt.pdf')
            plt.close()
        except:
            pass

    def test_apex_diff_plots(self):
        """Uncertainty of apex location determination at default fine_step_size"""
        import matplotlib.pyplot as plt
        # on_travis = os.environ.get('ONTRAVIS') == 'True'

        p_lats, p_longs, p_alts = gen_plot_grid_fixed_alt(550.)
        # data returned are the locations along each direction
        # the full range of points obtained by iterating over all
        # recasting alts into a more convenient form for later calculation
        p_alts = [p_alts[0]] * len(p_longs)
        # set the date
        date = dt.datetime(2000, 1, 1)
        # memory for results
        apex_lat = np.zeros((len(p_lats), len(p_longs) + 1))
        apex_lon = np.zeros((len(p_lats), len(p_longs) + 1))
        apex_alt = np.zeros((len(p_lats), len(p_longs) + 1))
        apex_z = np.zeros((len(p_lats), len(p_longs) + 1))
        norm_alt = np.zeros((len(p_lats), len(p_longs) + 1))

        # set up multi
        if self.dc is not None:
            import itertools
            targets = itertools.cycle(dc.ids)
            pending = []
            for i, p_lat in enumerate(p_lats):
                print (i, p_lat)
                # iterate through target cyclicly and run commands
                dview.targets = next(targets)
                pending.append(dview.apply_async(OMMBV.apex_location_info, [p_lat] * len(p_longs), p_longs,
                                                 p_alts, [date] * len(p_longs),
                                                 fine_step_size=1.E-5,
                                                 return_geodetic=True))
                pending.append(dview.apply_async(OMMBV.apex_location_info, [p_lat] * len(p_longs), p_longs,
                                                 p_alts, [date] * len(p_longs),
                                                 fine_step_size=5.E-6,
                                                 return_geodetic=True))
            for i, p_lat in enumerate(p_lats):
                print ('collecting ', i, p_lat)
                # collect output
                x, y, z, _, _, h = pending.pop(0).get()
                x2, y2, z2, _, _, h2 = pending.pop(0).get()
                apex_lat[i, :-1] = np.abs(x2 - x)
                apex_lon[i, :-1] = np.abs(y2 - y)
                apex_z[i, :-1] = np.abs(z2 - z)
                apex_alt[i, :-1] = np.abs(h2 - h)

        else:
            # single processor case
            for i, p_lat in enumerate(p_lats):
                print (i, p_lat)
                x, y, z, _, _, h = OMMBV.apex_location_info([p_lat] * len(p_longs), p_longs,
                                                            p_alts, [date] * len(p_longs),
                                                            fine_step_size=1.E-5, return_geodetic=True)
                x2, y2, z2, _, _, h2 = OMMBV.apex_location_info([p_lat] * len(p_longs), p_longs,
                                                                p_alts, [date] * len(p_longs),
                                                                fine_step_size=5.E-6, return_geodetic=True)

                norm_alt[i, :-1] = h
                apex_lat[i, :-1] = np.abs(x2 - x)
                apex_lon[i, :-1] = np.abs(y2 - y)
                apex_z[i, :-1] = np.abs(z2 - z)
                apex_alt[i, :-1] = np.abs(h2 - h)

        # account for periodicity
        apex_lat[:, -1] = apex_lat[:, 0]
        apex_lon[:, -1] = apex_lon[:, 0]
        apex_z[:, -1] = apex_z[:, 0]
        apex_alt[:, -1] = apex_alt[:, 0]
        norm_alt[:, -1] = norm_alt[:, 0]

        idx, idy, = np.where(apex_lat > 10.)
        print('Locations with large apex x (ECEF) location differences.', p_lats[idx], p_longs[idx])

        ytickarr = np.array([0, 0.25, 0.5, 0.75, 1]) * (len(p_lats) - 1)
        xtickarr = np.array([0, 0.2, 0.4, 0.6, 0.8, 1]) * len(p_longs)
        ytickvals = ['-50', '-25', '0', '25', '50']

        try:
            fig = plt.figure()
            plt.imshow(np.log10(apex_lat), origin='lower')
            plt.colorbar()
            plt.yticks(ytickarr, ytickvals)
            plt.xticks(xtickarr, ['0', '72', '144', '216', '288', '360'])
            plt.title('Log Apex Location Difference (ECEF-x km)')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.savefig('apex_loc_diff_x.pdf')
            plt.close()

            fig = plt.figure()
            plt.imshow(np.log10(apex_lon), origin='lower')
            plt.colorbar()
            plt.yticks(ytickarr, ytickvals)
            plt.xticks(xtickarr, ['0', '72', '144', '216', '288', '360'])
            plt.title('Log Apex Location Difference (ECEF-y km)')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.savefig('apex_loc_diff_y.pdf')
            plt.close()

            fig = plt.figure()
            plt.imshow(np.log10(apex_z), origin='lower')
            plt.colorbar()
            plt.yticks(ytickarr, ytickvals)
            plt.xticks(xtickarr, ['0', '72', '144', '216', '288', '360'])
            plt.title('Log Apex Location Difference (ECEF-z km)')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.savefig('apex_loc_diff_z.pdf')
            plt.close()

            fig = plt.figure()
            plt.imshow(np.log10(apex_alt / norm_alt), origin='lower')
            plt.colorbar()
            plt.yticks(ytickarr, ytickvals)
            plt.xticks(xtickarr, ['0', '72', '144', '216', '288', '360'])
            plt.title('Log Apex Altitude Normalized Difference (km)')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.savefig('apex_norm_loc_diff_h.pdf')
            plt.close()

            fig = plt.figure()
            plt.imshow(np.log10(apex_alt), origin='lower')
            plt.colorbar()
            plt.yticks(ytickarr, ytickvals)
            plt.xticks(xtickarr, ['0', '72', '144', '216', '288', '360'])
            plt.title('Log Apex Altitude Difference (km)')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.savefig('apex_loc_diff_h.pdf')
            plt.close()

        except:
            pass

    def test_apex_fine_max_step_diff_plots(self):
        """Test apex location info for sensitivity to fine_steps parameters"""
        import matplotlib.pyplot as plt
        # on_travis = os.environ.get('ONTRAVIS') == 'True'

        p_lats, p_longs, p_alts = gen_plot_grid_fixed_alt(550.)
        # data returned are the locations along each direction
        # the full range of points obtained by iterating over all
        # recasting alts into a more convenient form for later calculation
        p_alts = [p_alts[0]] * len(p_longs)
        # set the date
        date = dt.datetime(2000, 1, 1)
        # memory for results
        apex_lat = np.zeros((len(p_lats), len(p_longs) + 1))
        apex_lon = np.zeros((len(p_lats), len(p_longs) + 1))
        apex_alt = np.zeros((len(p_lats), len(p_longs) + 1))
        apex_z = np.zeros((len(p_lats), len(p_longs) + 1))
        norm_alt = np.zeros((len(p_lats), len(p_longs) + 1))

        # set up multi
        if self.dc is not None:
            import itertools
            targets = itertools.cycle(dc.ids)
            pending = []
            for i, p_lat in enumerate(p_lats):
                print (i, p_lat)
                # iterate through target cyclicly and run commands
                dview.targets = next(targets)
                pending.append(dview.apply_async(OMMBV.apex_location_info, [p_lat] * len(p_longs), p_longs,
                                                 p_alts, [date] * len(p_longs),
                                                 fine_max_steps=5,
                                                 return_geodetic=True))
                pending.append(dview.apply_async(OMMBV.apex_location_info, [p_lat] * len(p_longs), p_longs,
                                                 p_alts, [date] * len(p_longs),
                                                 fine_max_steps=10,
                                                 return_geodetic=True))
            for i, p_lat in enumerate(p_lats):
                print ('collecting ', i, p_lat)
                # collect output
                x, y, z, _, _, h = pending.pop(0).get()
                x2, y2, z2, _, _, h2 = pending.pop(0).get()
                apex_lat[i, :-1] = np.abs(x2 - x)
                apex_lon[i, :-1] = np.abs(y2 - y)
                apex_z[i, :-1] = np.abs(z2 - z)
                apex_alt[i, :-1] = np.abs(h2 - h)

        else:
            # single processor case
            for i, p_lat in enumerate(p_lats):
                print (i, p_lat)
                x, y, z, _, _, h = OMMBV.apex_location_info([p_lat] * len(p_longs), p_longs,
                                                            p_alts, [date] * len(p_longs),
                                                            fine_max_steps=5, return_geodetic=True)
                x2, y2, z2, _, _, h2 = OMMBV.apex_location_info([p_lat] * len(p_longs), p_longs,
                                                                p_alts, [date] * len(p_longs),
                                                                fine_max_steps=10, return_geodetic=True)

                norm_alt[i, :-1] = h
                apex_lat[i, :-1] = np.abs(x2 - x)
                apex_lon[i, :-1] = np.abs(y2 - y)
                apex_z[i, :-1] = np.abs(z2 - z)
                apex_alt[i, :-1] = np.abs(h2 - h)

        # account for periodicity
        apex_lat[:, -1] = apex_lat[:, 0]
        apex_lon[:, -1] = apex_lon[:, 0]
        apex_z[:, -1] = apex_z[:, 0]
        apex_alt[:, -1] = apex_alt[:, 0]
        norm_alt[:, -1] = norm_alt[:, 0]

        idx, idy, = np.where(apex_lat > 10.)
        print('Locations with large apex x (ECEF) location differences.', p_lats[idx], p_longs[idx])

        ytickarr = np.array([0, 0.25, 0.5, 0.75, 1]) * (len(p_lats) - 1)
        xtickarr = np.array([0, 0.2, 0.4, 0.6, 0.8, 1]) * len(p_longs)
        ytickvals = ['-50', '-25', '0', '25', '50']

        try:
            fig = plt.figure()
            plt.imshow(np.log10(apex_lat), origin='lower')
            plt.colorbar()
            plt.yticks(ytickarr, ytickvals)
            plt.xticks(xtickarr, ['0', '72', '144', '216', '288', '360'])
            plt.title('Log Apex Location Difference (ECEF-x km)')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.savefig('apex_loc_max_steps_diff_x.pdf')
            plt.close()

            fig = plt.figure()
            plt.imshow(np.log10(apex_lon), origin='lower')
            plt.colorbar()
            plt.yticks(ytickarr, ytickvals)
            plt.xticks(xtickarr, ['0', '72', '144', '216', '288', '360'])
            plt.title('Log Apex Location Difference (ECEF-y km)')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.savefig('apex_loc_max_steps_diff_y.pdf')
            plt.close()

            fig = plt.figure()
            plt.imshow(np.log10(apex_z), origin='lower')
            plt.colorbar()
            plt.yticks(ytickarr, ytickvals)
            plt.xticks(xtickarr, ['0', '72', '144', '216', '288', '360'])
            plt.title('Log Apex Location Difference (ECEF-z km)')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.savefig('apex_loc_max_steps_diff_z.pdf')
            plt.close()

            fig = plt.figure()
            plt.imshow(np.log10(apex_alt / norm_alt), origin='lower')
            plt.colorbar()
            plt.yticks(ytickarr, ytickvals)
            plt.xticks(xtickarr, ['0', '72', '144', '216', '288', '360'])
            plt.title('Log Apex Altitude Normalized Difference (km)')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.savefig('apex_norm_loc_max_steps_diff_h.pdf')
            plt.close()

            fig = plt.figure()
            plt.imshow(np.log10(apex_alt), origin='lower')
            plt.colorbar()
            plt.yticks(ytickarr, ytickvals)
            plt.xticks(xtickarr, ['0', '72', '144', '216', '288', '360'])
            plt.title('Log Apex Altitude Normalized Difference (km)')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.savefig('apex_loc_max_steps_diff_h.pdf')
            plt.close()

        except:
            pass

    def test_ecef_geodetic_apex_diff_plots(self):
        """Characterize uncertainty of ECEF and Geodetic transformations"""
        import matplotlib.pyplot as plt
        # on_travis = os.environ.get('ONTRAVIS') == 'True'

        p_lats, p_longs, p_alts = gen_plot_grid_fixed_alt(550.)
        # data returned are the locations along each direction
        # the full range of points obtained by iterating over all
        # recasting alts into a more convenient form for later calculation
        p_alts = [p_alts[0]] * len(p_longs)
        # set the date
        date = dt.datetime(2000, 1, 1)
        # memory for results
        apex_x = np.zeros((len(p_lats), len(p_longs) + 1))
        apex_y = np.zeros((len(p_lats), len(p_longs) + 1))
        apex_z = np.zeros((len(p_lats), len(p_longs) + 1))
        apex_alt = np.zeros((len(p_lats), len(p_longs) + 1))
        norm_alt = np.zeros((len(p_lats), len(p_longs) + 1))

        # set up multi
        if self.dc is not None:
            import itertools
            targets = itertools.cycle(dc.ids)
            pending = []
            for i, p_lat in enumerate(p_lats):
                print (i, p_lat)
                # iterate through target cyclicly and run commands
                dview.targets = next(targets)
                pending.append(dview.apply_async(OMMBV.geodetic_to_ecef, np.array([p_lat] * len(p_longs)), p_longs,
                                                 p_alts))
            for i, p_lat in enumerate(p_lats):
                print ('collecting ', i, p_lat)
                # collect output
                x, y, z = pending.pop(0).get()

                # iterate through target cyclicly and run commands
                dview.targets = next(targets)
                pending.append(dview.apply_async(OMMBV.python_ecef_to_geodetic, x, y, z))

            for i, p_lat in enumerate(p_lats):
                print ('collecting 2', i, p_lat)
                # collect output
                lat2, lon2, alt2 = pending.pop(0).get()

                # iterate through target cyclicly and run commands
                dview.targets = next(targets)
                pending.append(dview.apply_async(OMMBV.apex_location_info, np.array([p_lat] * len(p_longs)), p_longs,
                                                 p_alts, [date] * len(p_longs),
                                                 return_geodetic=True))

                pending.append(dview.apply_async(OMMBV.apex_location_info, lat2, lon2, alt2,
                                                 [date] * len(p_longs),
                                                 return_geodetic=True))

            for i, p_lat in enumerate(p_lats):
                print ('collecting 3', i, p_lat)
                x, y, z, _, _, h = pending.pop(0).get()
                x2, y2, z2, _, _, h2 = pending.pop(0).get()
                norm_alt[i, :-1] = np.abs(h)
                apex_x[i, :-1] = np.abs(x2 - x)
                apex_y[i, :-1] = np.abs(y2 - y)
                apex_z[i, :-1] = np.abs(z2 - z)
                apex_alt[i, :-1] = np.abs(h2 - h)


        else:
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
        apex_alt[:, -1] = apex_alt[:, 0]
        norm_alt[:, -1] = norm_alt[:, 0]

        ytickarr = np.array([0, 0.25, 0.5, 0.75, 1]) * (len(p_lats) - 1)
        xtickarr = np.array([0, 0.2, 0.4, 0.6, 0.8, 1]) * len(p_longs)
        ytickvals = ['-50', '-25', '0', '25', '50']

        try:
            fig = plt.figure()
            plt.imshow(np.log10(apex_x), origin='lower')
            plt.colorbar()
            plt.yticks(ytickarr, ytickvals)
            plt.xticks(xtickarr, ['0', '72', '144', '216', '288', '360'])
            plt.title('Log ECEF-Geodetic Apex Difference (ECEF-x km)')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.savefig('ecef_geodetic_apex_diff_x.pdf')
            plt.close()

            fig = plt.figure()
            plt.imshow(np.log10(apex_y), origin='lower')
            plt.colorbar()
            plt.yticks(ytickarr, ytickvals)
            plt.xticks(xtickarr, ['0', '72', '144', '216', '288', '360'])
            plt.title('Log ECEF-Geodetic Apex Difference (ECEF-y km)')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.savefig('ecef_geodetic_apex_diff_y.pdf')
            plt.close()

            fig = plt.figure()
            plt.imshow(np.log10(apex_z), origin='lower')
            plt.colorbar()
            plt.yticks(ytickarr, ytickvals)
            plt.xticks(xtickarr, ['0', '72', '144', '216', '288', '360'])
            plt.title('Log ECEF-Geodetic Apex Difference (ECEF-z km)')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.savefig('ecef_geodetic_apex_diff_z.pdf')
            plt.close()

            fig = plt.figure()
            plt.imshow(np.log10(apex_alt), origin='lower')
            plt.colorbar()
            plt.yticks(ytickarr, ytickvals)
            plt.xticks(xtickarr, ['0', '72', '144', '216', '288', '360'])
            plt.title('Log ECEF-Geodetic Apex Altitude Difference (km)')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.savefig('ecef_geodetic_apex_diff_h.pdf')
            plt.close()

            fig = plt.figure()
            plt.imshow(np.log10(apex_alt / norm_alt), origin='lower')
            plt.colorbar()
            plt.yticks(ytickarr, ytickvals)
            plt.xticks(xtickarr, ['0', '72', '144', '216', '288', '360'])
            plt.title('Log ECEF-Geodetic Apex Normalized Altitude Difference (km)')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.savefig('ecef_geodetic_apex_norm_diff_h.pdf')
            plt.close()

        except:
            pass
