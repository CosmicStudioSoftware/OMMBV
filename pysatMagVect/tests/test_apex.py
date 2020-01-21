import nose.tools
from nose.tools import assert_raises, raises
from nose.tools import assert_almost_equals as asseq

import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pds

import pysatMagVect as pymv
import pysat

from pysatMagVect.tests.test_core import gen_data_fixed_alt, gen_trace_data_fixed_alt
from pysatMagVect.tests.test_core import gen_plot_grid_fixed_alt

from pysatMagVect.tests.test_core import dview, dc

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

        lats, longs, alts = gen_trace_data_fixed_alt(550.)
        ecf_x,ecf_y,ecf_z = pymv.geodetic_to_ecef(lats,
                                                  longs,
                                                  alts)
        # step size to be tried
        fine_steps_goal = np.array([25.6, 12.8, 6.4, 3.2, 1.6, 0.8, 0.4, 0.2,
                                    0.1, 0.05, .025, .0125, .00625, .003125,
                                    .0015625, .00078125, .000390625, .0001953125,
                                    .0001953125/2., .0001953125/4., .0001953125/8.,
                                    .0001953125/16., .0001953125/32., .0001953125/64.,
                                    .0001953125/128., .0001953125/256., .0001953125/512.,
                                    .0001953125/1024., .0001953125/2048., .0001953125/4096.])

        date = datetime.datetime(2000, 1, 1)
        dx = []
        dy = []
        dz = []
        dh = []

        # set up multi
        if self.dc is not None:
        # if False:
            import itertools
            targets = itertools.cycle(dc.ids)
            pending = []
            for lat, lon, alt in zip(lats, longs, alts):
                for steps in fine_steps_goal:
                    # iterate through target cyclicly and run commands
                    dview.targets = targets.next()
                    pending.append(dview.apply_async(pymv.apex_location_info, [lat],
                                                     [lon], [alt], [date], fine_step_size=steps,
                                                     return_geodetic=True))
            # for x, y, z in zip(ecf_x, ecf_y, ecf_z):
                out = []
                for steps in fine_steps_goal:
                    # collect output
                    x, y, z, _, _, apex_height = pending.pop(0).get()
                    pt = [x[0], y[0], z[0], apex_height[0]]
                    out.append(pt)

                final_pt = pds.DataFrame(out, columns = ['x', 'y', 'z', 'h'])
                dx.append(np.abs(final_pt.ix[1:, 'x'].values - final_pt.ix[:,'x'].values[:-1]))
                dy.append(np.abs(final_pt.ix[1:, 'y'].values - final_pt.ix[:,'y'].values[:-1]))
                dz.append(np.abs(final_pt.ix[1:, 'z'].values - final_pt.ix[:,'z'].values[:-1]))
                dh.append(np.abs(final_pt.ix[1:, 'h'].values - final_pt.ix[:,'h'].values[:-1]))
        else:
            for lat, lon, alt in zip(lats, longs, alts):
                out = []
                for steps in fine_steps_goal:
                    x, y, z, _, _, apex_height = pymv.apex_location_info([lat], [lon], [alt], [date], fine_step_size=steps,
                                                                         return_geodetic=True)
                    pt = [x[0], y[0], z[0], apex_height[0]]
                    out.append(pt)

                final_pt = pds.DataFrame(out, columns = ['x', 'y', 'z', 'h'])
                dx.append(np.abs(final_pt.ix[1:, 'x'].values - final_pt.ix[:,'x'].values[:-1]))
                dy.append(np.abs(final_pt.ix[1:, 'y'].values - final_pt.ix[:,'y'].values[:-1]))
                dz.append(np.abs(final_pt.ix[1:, 'z'].values - final_pt.ix[:,'z'].values[:-1]))
                dh.append(np.abs(final_pt.ix[1:, 'h'].values - final_pt.ix[:,'h'].values[:-1]))


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
            plt.savefig('apex_location_vs_step_size.pdf' )
            plt.close()
        except:
            pass


    def test_apex_plots(self):
        import matplotlib.pyplot as plt
        import os
        # on_travis = os.environ.get('ONTRAVIS') == 'True'

        p_lats, p_longs, p_alts = gen_plot_grid_fixed_alt(120.)
        # data returned are the locations along each direction
        # the full range of points obtained by iterating over all
        # recasting alts into a more convenient form for later calculation
        p_alts = [p_alts[0]]*len(p_longs)
        # set the date
        date = datetime.datetime(2000,1,1)
        # memory for results
        apex_lat = np.zeros((len(p_lats), len(p_longs)+1))
        apex_lon = np.zeros((len(p_lats), len(p_longs)+1))
        apex_alt = np.zeros((len(p_lats), len(p_longs)+1))


        # set up multi
        if self.dc is not None:
            import itertools
            targets = itertools.cycle(dc.ids)
            pending = []
            for i,p_lat in enumerate(p_lats):
                print (i, p_lat)
                # iterate through target cyclicly and run commands
                dview.targets = targets.next()
                pending.append(dview.apply_async(pymv.apex_location_info, [p_lat]*len(p_longs), p_longs,
                                                                            p_alts, [date]*len(p_longs),
                                                                            return_geodetic=True))
            for i,p_lat in enumerate(p_lats):
                print ('collecting ', i, p_lat)
                # collect output
                x, y, z, olat, olon, oalt = pending.pop(0).get()
                apex_lat[i,:-1] = olat
                apex_lon[i,:-1] = olon
                apex_alt[i,:-1] = oalt

        else:
            # single processor case
            for i,p_lat in enumerate(p_lats):
                print (i, p_lat)
                x, y, z, olat, olon, oalt = pymv.apex_location_info([p_lat]*len(p_longs), p_longs,
                                                                        p_alts, [date]*len(p_longs),
                                                                        return_geodetic=True)
                apex_lat[i,:-1] = olat
                apex_lon[i,:-1] = olon
                apex_alt[i,:-1] = oalt

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
        apex_lat[:,-1] = apex_lat[:,0]
        apex_lon[:,-1] = apex_lon[:,0]
        apex_alt[:,-1] = apex_alt[:,0]

        ytickarr = np.array([0, 0.25, 0.5, 0.75, 1])*(len(p_lats)-1)
        xtickarr = np.array([0, 0.2, 0.4, 0.6, 0.8, 1])*len(p_longs)
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
        import matplotlib.pyplot as plt
        import os
        # on_travis = os.environ.get('ONTRAVIS') == 'True'

        p_lats, p_longs, p_alts = gen_plot_grid_fixed_alt(550.)
        # data returned are the locations along each direction
        # the full range of points obtained by iterating over all
        # recasting alts into a more convenient form for later calculation
        p_alts = [p_alts[0]]*len(p_longs)
        # set the date
        date = datetime.datetime(2000,1,1)
        # memory for results
        apex_lat = np.zeros((len(p_lats), len(p_longs)+1))
        apex_lon = np.zeros((len(p_lats), len(p_longs)+1))
        apex_alt = np.zeros((len(p_lats), len(p_longs)+1))
        apex_z = np.zeros((len(p_lats), len(p_longs)+1))
        norm_alt = np.zeros((len(p_lats), len(p_longs)+1))


        # set up multi
        if self.dc is not None:
            import itertools
            targets = itertools.cycle(dc.ids)
            pending = []
            for i,p_lat in enumerate(p_lats):
                print (i, p_lat)
                # iterate through target cyclicly and run commands
                dview.targets = targets.next()
                pending.append(dview.apply_async(pymv.apex_location_info, [p_lat]*len(p_longs), p_longs,
                                                                            p_alts, [date]*len(p_longs),
                                                                            fine_step_size=1.E-5,
                                                                            return_geodetic=True))
                pending.append(dview.apply_async(pymv.apex_location_info, [p_lat]*len(p_longs), p_longs,
                                                                            p_alts, [date]*len(p_longs),
                                                                            fine_step_size=5.E-6,
                                                                            return_geodetic=True))
            for i,p_lat in enumerate(p_lats):
                print ('collecting ', i, p_lat)
                # collect output
                x, y, z, _, _, h = pending.pop(0).get()
                x2, y2, z2, _, _, h2 = pending.pop(0).get()
                apex_lat[i,:-1] = np.abs(x2 - x)
                apex_lon[i,:-1] = np.abs(y2 - y)
                apex_z[i,:-1] = np.abs(z2 - z)
                apex_alt[i,:-1] = np.abs(h2 - h)

        else:
            # single processor case
            for i,p_lat in enumerate(p_lats):
                print (i, p_lat)
                x, y, z, _, _, h = pymv.apex_location_info([p_lat]*len(p_longs), p_longs,
                                                                        p_alts, [date]*len(p_longs),
                                                                        fine_step_size=1.E-5, return_geodetic=True)
                x2, y2, z2, _, _, h2 = pymv.apex_location_info([p_lat]*len(p_longs), p_longs,
                                                                           p_alts, [date]*len(p_longs),
                                                                           fine_step_size=5.E-6, return_geodetic=True)

                norm_alt[i,:-1] = h
                apex_lat[i,:-1] = np.abs(x2 - x)
                apex_lon[i,:-1] = np.abs(y2 - y)
                apex_z[i,:-1] = np.abs(z2 - z)
                apex_alt[i,:-1] = np.abs(h2 - h)

        # account for periodicity
        apex_lat[:,-1] = apex_lat[:,0]
        apex_lon[:,-1] = apex_lon[:,0]
        apex_z[:,-1] = apex_z[:,0]
        apex_alt[:,-1] = apex_alt[:,0]
        norm_alt[:,-1] = norm_alt[:,0]

        idx, idy, = np.where(apex_lat > 10.)
        print('Locations with large apex x (ECEF) location differences.', p_lats[idx], p_longs[idx])

        ytickarr = np.array([0, 0.25, 0.5, 0.75, 1])*(len(p_lats)-1)
        xtickarr = np.array([0, 0.2, 0.4, 0.6, 0.8, 1])*len(p_longs)
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
            plt.imshow(np.log10(apex_alt/norm_alt), origin='lower')
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

### Test apex location info for sensitivity to fine_steps parameters
    def test_apex_fine_max_step_diff_plots(self):
        import matplotlib.pyplot as plt
        import os
        # on_travis = os.environ.get('ONTRAVIS') == 'True'

        p_lats, p_longs, p_alts = gen_plot_grid_fixed_alt(550.)
        # data returned are the locations along each direction
        # the full range of points obtained by iterating over all
        # recasting alts into a more convenient form for later calculation
        p_alts = [p_alts[0]]*len(p_longs)
        # set the date
        date = datetime.datetime(2000,1,1)
        # memory for results
        apex_lat = np.zeros((len(p_lats), len(p_longs)+1))
        apex_lon = np.zeros((len(p_lats), len(p_longs)+1))
        apex_alt = np.zeros((len(p_lats), len(p_longs)+1))
        apex_z = np.zeros((len(p_lats), len(p_longs)+1))
        norm_alt = np.zeros((len(p_lats), len(p_longs)+1))


        # set up multi
        if self.dc is not None:
            import itertools
            targets = itertools.cycle(dc.ids)
            pending = []
            for i,p_lat in enumerate(p_lats):
                print (i, p_lat)
                # iterate through target cyclicly and run commands
                dview.targets = targets.next()
                pending.append(dview.apply_async(pymv.apex_location_info, [p_lat]*len(p_longs), p_longs,
                                                                            p_alts, [date]*len(p_longs),
                                                                            fine_max_steps=5,
                                                                            return_geodetic=True))
                pending.append(dview.apply_async(pymv.apex_location_info, [p_lat]*len(p_longs), p_longs,
                                                                            p_alts, [date]*len(p_longs),
                                                                            fine_max_steps=10,
                                                                            return_geodetic=True))
            for i,p_lat in enumerate(p_lats):
                print ('collecting ', i, p_lat)
                # collect output
                x, y, z, _, _, h = pending.pop(0).get()
                x2, y2, z2, _, _, h2 = pending.pop(0).get()
                apex_lat[i,:-1] = np.abs(x2 - x)
                apex_lon[i,:-1] = np.abs(y2 - y)
                apex_z[i,:-1] = np.abs(z2 - z)
                apex_alt[i,:-1] = np.abs(h2 - h)

        else:
            # single processor case
            for i,p_lat in enumerate(p_lats):
                print (i, p_lat)
                x, y, z, _, _, h = pymv.apex_location_info([p_lat]*len(p_longs), p_longs,
                                                                        p_alts, [date]*len(p_longs),
                                                                        fine_max_steps=5, return_geodetic=True)
                x2, y2, z2, _, _, h2 = pymv.apex_location_info([p_lat]*len(p_longs), p_longs,
                                                                           p_alts, [date]*len(p_longs),
                                                                           fine_max_steps=10, return_geodetic=True)

                norm_alt[i,:-1] = h
                apex_lat[i,:-1] = np.abs(x2 - x)
                apex_lon[i,:-1] = np.abs(y2 - y)
                apex_z[i,:-1] = np.abs(z2 - z)
                apex_alt[i,:-1] = np.abs(h2 - h)

        # account for periodicity
        apex_lat[:,-1] = apex_lat[:,0]
        apex_lon[:,-1] = apex_lon[:,0]
        apex_z[:,-1] = apex_z[:,0]
        apex_alt[:,-1] = apex_alt[:,0]
        norm_alt[:,-1] = norm_alt[:,0]

        idx, idy, = np.where(apex_lat > 10.)
        print('Locations with large apex x (ECEF) location differences.', p_lats[idx], p_longs[idx])

        ytickarr = np.array([0, 0.25, 0.5, 0.75, 1])*(len(p_lats)-1)
        xtickarr = np.array([0, 0.2, 0.4, 0.6, 0.8, 1])*len(p_longs)
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
            plt.imshow(np.log10(apex_alt/norm_alt), origin='lower')
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
        import matplotlib.pyplot as plt
        import os
        # on_travis = os.environ.get('ONTRAVIS') == 'True'

        p_lats, p_longs, p_alts = gen_plot_grid_fixed_alt(550.)
        # data returned are the locations along each direction
        # the full range of points obtained by iterating over all
        # recasting alts into a more convenient form for later calculation
        p_alts = [p_alts[0]]*len(p_longs)
        # set the date
        date = datetime.datetime(2000,1,1)
        # memory for results
        apex_x = np.zeros((len(p_lats), len(p_longs)+1))
        apex_y = np.zeros((len(p_lats), len(p_longs)+1))
        apex_z = np.zeros((len(p_lats), len(p_longs)+1))
        apex_alt = np.zeros((len(p_lats), len(p_longs)+1))
        norm_alt = np.zeros((len(p_lats), len(p_longs)+1))


        # set up multi
        if self.dc is not None:
            import itertools
            targets = itertools.cycle(dc.ids)
            pending = []
            for i,p_lat in enumerate(p_lats):
                print (i, p_lat)
                # iterate through target cyclicly and run commands
                dview.targets = targets.next()
                pending.append(dview.apply_async(pymv.geodetic_to_ecef, np.array([p_lat]*len(p_longs)), p_longs,
                                                                            p_alts))
            for i,p_lat in enumerate(p_lats):
                print ('collecting ', i, p_lat)
                # collect output
                x, y, z = pending.pop(0).get()

                # iterate through target cyclicly and run commands
                dview.targets = targets.next()
                pending.append(dview.apply_async(pymv.python_ecef_to_geodetic, x, y, z))

            for i,p_lat in enumerate(p_lats):
                print ('collecting 2', i, p_lat)
                # collect output
                lat2, lon2, alt2 = pending.pop(0).get()

                # iterate through target cyclicly and run commands
                dview.targets = targets.next()
                pending.append(dview.apply_async(pymv.apex_location_info, np.array([p_lat]*len(p_longs)), p_longs,
                                                                            p_alts, [date]*len(p_longs),
                                                                            return_geodetic=True))

                pending.append(dview.apply_async(pymv.apex_location_info, lat2, lon2, alt2,
                                                                            [date]*len(p_longs),
                                                                            return_geodetic=True))

            for i,p_lat in enumerate(p_lats):
                print ('collecting 3', i, p_lat)
                x, y, z, _, _, h = pending.pop(0).get()
                x2, y2, z2, _, _, h2 = pending.pop(0).get()
                norm_alt[i, :-1] = np.abs(h)
                apex_x[i,:-1] = np.abs(x2 - x)
                apex_y[i,:-1] = np.abs(y2 - y)
                apex_z[i,:-1] = np.abs(z2 - z)
                apex_alt[i, :-1] = np.abs(h2 - h)


        else:
            # single processor case
            for i,p_lat in enumerate(p_lats):
                print (i, p_lat)
                x, y, z = pymv.geodetic_to_ecef([p_lat]*len(p_longs), p_longs, p_alts)
                lat2, lon2, alt2 = pymv.ecef_to_geodetic(x, y, z)
                x2, y2, z2 = pymv.geodetic_to_ecef(lat2, lon2, alt2)
                apex_x[i,:-1] = np.abs(x2 - x)
                apex_y[i,:-1] = np.abs(y2 - y)
                apex_z[i,:-1] = np.abs(z2 - z)


        # account for periodicity
        apex_x[:,-1] = apex_x[:,0]
        apex_y[:,-1] = apex_y[:,0]
        apex_z[:,-1] = apex_z[:,0]
        apex_alt[:,-1] = apex_alt[:,0]
        norm_alt[:,-1] = norm_alt[:,0]


        ytickarr = np.array([0, 0.25, 0.5, 0.75, 1])*(len(p_lats)-1)
        xtickarr = np.array([0, 0.2, 0.4, 0.6, 0.8, 1])*len(p_longs)
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
            plt.imshow(np.log10(apex_alt/norm_alt), origin='lower')
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
