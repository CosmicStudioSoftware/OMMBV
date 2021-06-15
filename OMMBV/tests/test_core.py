import datetime as dt
import itertools
import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pds
import datetime

import functools

import OMMBV as OMMBV
from OMMBV import igrf

import pysat

# multiprocessing boolean flag
multiproc = False
if multiproc:
    # get remote instances
    import ipyparallel
    print('parallel in')
    dc = ipyparallel.Client()
    dview = dc[:]
    print('parallel out')
else:
    # nothing to set
    dc = None
    dview = None


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
    # pull out lats and longs
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
        long_dim = np.arange(0., 360., 1*30.)
        lat_dim = np.arange(-50., 50.1, 0.25*30)

    alt_dim = np.array([alt])
    return lat_dim, long_dim, alt_dim


################## UNIT VECTOR TESTS ###############################
class TestUnitVectors():

    def __init__(self):
        # placeholder for data management features
        self.inst = pysat.Instrument('pysat', 'testing')
        self.inst.yr = 2010.
        self.inst.doy = 1.
        self.dview = dview
        self.dc = dc

        return

    def test_unit_vector_step_size_sensitivity(self):
        """Test sensitivity of unit vectors as step_size decreased"""
        p_lats, p_longs, p_alts = gen_plot_grid_fixed_alt(550.)
        # data returned are the locations along each direction
        # the full range of points obtained by iterating over all
        # recasting alts into a more convenient form for later calculation
        p_alts = [p_alts[0]]*len(p_longs)

        # step size to be tried
        steps_goal = np.arange(7)
        steps_goal = 20. / 2 ** steps_goal

        date = dt.datetime(2000, 1, 1)
        dzx = []
        dzy = []
        dzz = []
        dmx = []
        dmy = []
        dmz = []

        # set up multi
        if self.dc is not None:
            targets = itertools.cycle(dc.ids)
            pending = []
            steps_out = []
            for steps in steps_goal:
                out = []
                for lat in p_lats:
                    lats = [lat]*len(p_longs)
                    # iterate through target cyclicly and run commands
                    dview.targets = next(targets)
                    pending.append(dview.apply_async(OMMBV.calculate_mag_drift_unit_vectors_ecef, lats,
                                                     p_longs, p_alts, [date]*len(p_longs), step_size=steps))
                for lat in p_lats:
                    # collect output
                    zx, zy, zz, _, _, _, mx, my, mz = pending.pop(0).get()
                    pt = {'zx': zx,
                          'zy': zy,
                          'zz': zz,
                          'mx': mx,
                          'my': my,
                          'mz': mz}
                    out.append(pds.DataFrame(pt))
                    # merge all values for single step size together
                out = pds.concat(out)
                steps_out.append(out)

            for i in np.arange(len(steps_out) - 1):
                dzx.append(np.abs(steps_out[i]['zx'].values - steps_out[i + 1]['zx'].values))
                dzy.append(np.abs(steps_out[i]['zy'].values - steps_out[i + 1]['zy'].values))
                dzz.append(np.abs(steps_out[i]['zz'].values - steps_out[i + 1]['zz'].values))
                dmx.append(np.abs(steps_out[i]['mx'].values - steps_out[i + 1]['mx'].values))
                dmy.append(np.abs(steps_out[i]['my'].values - steps_out[i + 1]['my'].values))
                dmz.append(np.abs(steps_out[i]['mz'].values - steps_out[i + 1]['mz'].values))
        else:
            steps_out = []
            for steps in steps_goal:
                out = []
                for lat in p_lats:
                    lats = [lat]*len(p_longs)
                    zx, zy, zz, _, _, _, mx, my, mz = OMMBV.calculate_mag_drift_unit_vectors_ecef(lats,
                                                                                                 p_longs, p_alts,
                                                                                                 [date]*len(p_longs),
                                                                                                 step_size=steps)
                    pt = {'zx': zx,
                          'zy': zy,
                          'zz': zz,
                          'mx': mx,
                          'my': my,
                          'mz': mz}
                    out.append(pds.DataFrame(pt))
                # merge all values for single step size together
                out = pds.concat(out)
                steps_out.append(out)

            for i in np.arange(len(steps_out) - 1):
                dzx.append(np.abs(steps_out[i]['zx'].values - steps_out[i + 1]['zx'].values))
                dzy.append(np.abs(steps_out[i]['zy'].values - steps_out[i + 1]['zy'].values))
                dzz.append(np.abs(steps_out[i]['zz'].values - steps_out[i + 1]['zz'].values))
                dmx.append(np.abs(steps_out[i]['mx'].values - steps_out[i + 1]['mx'].values))
                dmy.append(np.abs(steps_out[i]['my'].values - steps_out[i + 1]['my'].values))
                dmz.append(np.abs(steps_out[i]['mz'].values - steps_out[i + 1]['mz'].values))

        dzx = pds.DataFrame(dzx)
        dzy = pds.DataFrame(dzy)
        dzz = pds.DataFrame(dzz)
        dmx = pds.DataFrame(dmx)
        dmy = pds.DataFrame(dmy)
        dmz = pds.DataFrame(dmz)

        try:
            plt.figure()
            plt.plot(np.log10(steps_goal[1:]), np.log10(dzx.mean(axis=1)),
                     label='x')
            plt.plot(np.log10(steps_goal[1:]), np.log10(dzy.mean(axis=1)),
                     label='y')
            plt.plot(np.log10(steps_goal[1:]), np.log10(dzz.mean(axis=1)),
                     label='z')
            plt.xlabel('Log Step Size (km)')
            plt.ylabel('Change in Vector Component')
            plt.title("Change in Zonal Unit Vector (ECEF)")
            plt.legend()
            plt.tight_layout()
            plt.savefig('overall_zonal_diff_vs_step_size.pdf')
            plt.close()

            plt.figure()
            plt.plot(np.log10(steps_goal[1:]), np.log10(dmx.mean(axis=1)),
                     label='x')
            plt.plot(np.log10(steps_goal[1:]), np.log10(dmy.mean(axis=1)),
                     label='y')
            plt.plot(np.log10(steps_goal[1:]), np.log10(dmz.mean(axis=1)),
                     label='z')
            plt.xlabel('Log Step Size (km)')
            plt.ylabel('Change in Vector Component')
            plt.title("Change in Meridional Unit Vector (ECEF)")
            plt.legend()
            plt.tight_layout()
            plt.savefig('overall_mer_diff_vs_step_size.pdf')
            plt.close()

        except:
            pass

    def test_D_vector_step_size_sensitivity(self):
        """Test sensitivity of D vectors as step_size decreased"""
        p_lats, p_longs, p_alts = gen_plot_grid_fixed_alt(550.)
        # data returned are the locations along each direction
        # the full range of points obtained by iterating over all
        # recasting alts into a more convenient form for later calculation
        p_alts = [p_alts[0]]*len(p_longs)

        # step size to be tried
        steps_goal = np.arange(7)
        steps_goal = 20. / 2 ** steps_goal

        date = dt.datetime(2000, 1, 1)
        dzx = []
        dzy = []
        dzz = []
        dmx = []
        dmy = []
        dmz = []

        # set up multi
        if self.dc is not None:
            targets = itertools.cycle(dc.ids)
            pending = []
            steps_out = []
            for steps in steps_goal:
                out = []
                for lat in p_lats:
                    lats = [lat]*len(p_longs)
                    # iterate through target cyclicly and run commands
                    dview.targets = next(targets)
                    pending.append(dview.apply_async(OMMBV.calculate_mag_drift_unit_vectors_ecef, lats,
                                                     p_longs, p_alts, [date]*len(p_longs), step_size=steps,
                                                     dstep_size=steps, full_output=True))
                for lat in p_lats:
                    # collect output
                    zx, zy, zz, _, _, _, mx, my, mz, d = pending.pop(0).get()
                    pt = {'zx': d['d_zon_x'],
                          'zy': d['d_zon_y'],
                          'zz': d['d_zon_z'],
                          'mx': d['d_mer_x'],
                          'my': d['d_mer_y'],
                          'mz': d['d_mer_z']}
                    out.append(pds.DataFrame(pt))
                    # merge all values for single step size together
                out = pds.concat(out)
                steps_out.append(out)

            for i in np.arange(len(steps_out) - 1):
                dzx.append(np.abs(steps_out[i]['zx'].values - steps_out[i + 1]['zx'].values))
                dzy.append(np.abs(steps_out[i]['zy'].values - steps_out[i + 1]['zy'].values))
                dzz.append(np.abs(steps_out[i]['zz'].values - steps_out[i + 1]['zz'].values))
                dmx.append(np.abs(steps_out[i]['mx'].values - steps_out[i + 1]['mx'].values))
                dmy.append(np.abs(steps_out[i]['my'].values - steps_out[i + 1]['my'].values))
                dmz.append(np.abs(steps_out[i]['mz'].values - steps_out[i + 1]['mz'].values))
        else:
            steps_out = []
            for steps in steps_goal:
                out = []
                for lat in p_lats:
                    lats = [lat]*len(p_longs)
                    zx, zy, zz, _, _, _, mx, my, mz, d = OMMBV.calculate_mag_drift_unit_vectors_ecef(lats,
                                                                                                    p_longs, p_alts,
                                                                                                    [date]*len(p_longs),
                                                                                                    step_size=steps,
                                                                                                    dstep_size=steps,
                                                                                                    full_output=True)
                    pt = {'zx': d['d_zon_x'],
                          'zy': d['d_zon_y'],
                          'zz': d['d_zon_z'],
                          'mx': d['d_mer_x'],
                          'my': d['d_mer_y'],
                          'mz': d['d_mer_z']}
                    out.append(pds.DataFrame(pt))
                # merge all values for single step size together
                out = pds.concat(out)
                steps_out.append(out)

            for i in np.arange(len(steps_out) - 1):
                dzx.append(np.abs(steps_out[i]['zx'].values - steps_out[i + 1]['zx'].values))
                dzy.append(np.abs(steps_out[i]['zy'].values - steps_out[i + 1]['zy'].values))
                dzz.append(np.abs(steps_out[i]['zz'].values - steps_out[i + 1]['zz'].values))
                dmx.append(np.abs(steps_out[i]['mx'].values - steps_out[i + 1]['mx'].values))
                dmy.append(np.abs(steps_out[i]['my'].values - steps_out[i + 1]['my'].values))
                dmz.append(np.abs(steps_out[i]['mz'].values - steps_out[i + 1]['mz'].values))

        dzx = pds.DataFrame(dzx)
        dzy = pds.DataFrame(dzy)
        dzz = pds.DataFrame(dzz)
        dmx = pds.DataFrame(dmx)
        dmy = pds.DataFrame(dmy)
        dmz = pds.DataFrame(dmz)

        try:
            plt.figure()
            plt.plot(np.log10(steps_goal[1:]), np.log10(dzx.mean(axis=1)),
                     label='x')
            plt.plot(np.log10(steps_goal[1:]), np.log10(dzy.mean(axis=1)),
                     label='y')
            plt.plot(np.log10(steps_goal[1:]), np.log10(dzz.mean(axis=1)),
                     label='z')
            plt.xlabel('Log Step Size (km)')
            plt.ylabel('Change in D Vector Component')
            plt.title("Change in D Zonal Vector (ECEF)")
            plt.legend()
            plt.tight_layout()
            plt.savefig('overall_D_zonal_diff_vs_step_size.pdf')
            plt.close()

            plt.figure()
            plt.plot(np.log10(steps_goal[1:]), np.log10(dmx.mean(axis=1)),
                     label='x')
            plt.plot(np.log10(steps_goal[1:]), np.log10(dmy.mean(axis=1)),
                     label='y')
            plt.plot(np.log10(steps_goal[1:]), np.log10(dmz.mean(axis=1)),
                     label='z')
            plt.xlabel('Log Step Size (km)')
            plt.ylabel('Change in Vector Component')
            plt.title("Change in Meridional D Vector (ECEF)")
            plt.legend()
            plt.tight_layout()
            plt.savefig('overall_D_mer_diff_vs_step_size.pdf')
            plt.close()

        except:
            pass

    def test_E_vector_step_size_sensitivity(self):
        """Test sensitivity of E vectors as step_size decreased"""
        p_lats, p_longs, p_alts = gen_plot_grid_fixed_alt(550.)
        # data returned are the locations along each direction
        # the full range of points obtained by iterating over all
        # recasting alts into a more convenient form for later calculation
        p_alts = [p_alts[0]]*len(p_longs)

        # step size to be tried
        steps_goal = np.arange(7)
        steps_goal = 20. / 2 ** steps_goal

        date = dt.datetime(2000, 1, 1)
        dzx = []
        dzy = []
        dzz = []
        dmx = []
        dmy = []
        dmz = []

        # set up multi
        if self.dc is not None:
            targets = itertools.cycle(dc.ids)
            pending = []
            steps_out = []
            for steps in steps_goal:
                out = []
                for lat in p_lats:
                    lats = [lat]*len(p_longs)
                    # iterate through target cyclicly and run commands
                    dview.targets = next(targets)
                    pending.append(dview.apply_async(OMMBV.calculate_mag_drift_unit_vectors_ecef, lats,
                                                     p_longs, p_alts, [date]*len(p_longs), step_size=steps,
                                                     dstep_size=steps, full_output=True))
                for lat in p_lats:
                    # collect output
                    zx, zy, zz, _, _, _, mx, my, mz, d = pending.pop(0).get()
                    pt = {'zx': d['e_zon_x'],
                          'zy': d['e_zon_y'],
                          'zz': d['e_zon_z'],
                          'mx': d['e_mer_x'],
                          'my': d['e_mer_y'],
                          'mz': d['e_mer_z']}
                    out.append(pds.DataFrame(pt))
                    # merge all values for single step size together
                out = pds.concat(out)
                steps_out.append(out)

            for i in np.arange(len(steps_out) - 1):
                dzx.append(np.abs(steps_out[i]['zx'].values - steps_out[i + 1]['zx'].values))
                dzy.append(np.abs(steps_out[i]['zy'].values - steps_out[i + 1]['zy'].values))
                dzz.append(np.abs(steps_out[i]['zz'].values - steps_out[i + 1]['zz'].values))
                dmx.append(np.abs(steps_out[i]['mx'].values - steps_out[i + 1]['mx'].values))
                dmy.append(np.abs(steps_out[i]['my'].values - steps_out[i + 1]['my'].values))
                dmz.append(np.abs(steps_out[i]['mz'].values - steps_out[i + 1]['mz'].values))
        else:
            steps_out = []
            for steps in steps_goal:
                out = []
                for lat in p_lats:
                    lats = [lat]*len(p_longs)
                    zx, zy, zz, _, _, _, mx, my, mz, d = OMMBV.calculate_mag_drift_unit_vectors_ecef(lats,
                                                                                                    p_longs, p_alts,
                                                                                                    [date]*len(p_longs),
                                                                                                    step_size=steps,
                                                                                                    dstep_size=steps,
                                                                                                    full_output=True)
                    pt = {'zx': d['e_zon_x'],
                          'zy': d['e_zon_y'],
                          'zz': d['e_zon_z'],
                          'mx': d['e_mer_x'],
                          'my': d['e_mer_y'],
                          'mz': d['e_mer_z']}
                    out.append(pds.DataFrame(pt))
                # merge all values for single step size together
                out = pds.concat(out)
                steps_out.append(out)

            for i in np.arange(len(steps_out) - 1):
                dzx.append(np.abs(steps_out[i]['zx'].values - steps_out[i + 1]['zx'].values))
                dzy.append(np.abs(steps_out[i]['zy'].values - steps_out[i + 1]['zy'].values))
                dzz.append(np.abs(steps_out[i]['zz'].values - steps_out[i + 1]['zz'].values))
                dmx.append(np.abs(steps_out[i]['mx'].values - steps_out[i + 1]['mx'].values))
                dmy.append(np.abs(steps_out[i]['my'].values - steps_out[i + 1]['my'].values))
                dmz.append(np.abs(steps_out[i]['mz'].values - steps_out[i + 1]['mz'].values))

        dzx = pds.DataFrame(dzx)
        dzy = pds.DataFrame(dzy)
        dzz = pds.DataFrame(dzz)
        dmx = pds.DataFrame(dmx)
        dmy = pds.DataFrame(dmy)
        dmz = pds.DataFrame(dmz)

        try:
            plt.figure()
            plt.plot(np.log10(steps_goal[1:]), np.log10(dzx.mean(axis=1)),
                     label='x')
            plt.plot(np.log10(steps_goal[1:]), np.log10(dzy.mean(axis=1)),
                     label='y')
            plt.plot(np.log10(steps_goal[1:]), np.log10(dzz.mean(axis=1)),
                     label='z')
            plt.xlabel('Log Step Size (km)')
            plt.ylabel('Change in E Vector Component')
            plt.title("Change in E Zonal Vector (ECEF)")
            plt.legend()
            plt.tight_layout()
            plt.savefig('overall_E_zonal_diff_vs_step_size.pdf')
            plt.close()

            plt.figure()
            plt.plot(np.log10(steps_goal[1:]), np.log10(dmx.mean(axis=1)),
                     label='x')
            plt.plot(np.log10(steps_goal[1:]), np.log10(dmy.mean(axis=1)),
                     label='y')
            plt.plot(np.log10(steps_goal[1:]), np.log10(dmz.mean(axis=1)),
                     label='z')
            plt.xlabel('Log Step Size (km)')
            plt.ylabel('Change in Vector Component')
            plt.title("Change in Meridional E Vector (ECEF)")
            plt.legend()
            plt.tight_layout()
            plt.savefig('overall_E_mer_diff_vs_step_size.pdf')
            plt.close()

        except:
            pass

    def test_unit_vector_component_plots(self):
        """Ensure unit vector generation satisfies tolerance and gradient goals.

        Produces variety of plots."""
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

        grad_zon = zvx.copy()
        grad_mer = zvx.copy()
        tol_zon = zvx.copy()
        tol_mer = zvx.copy()
        init_type = zvx.copy()
        num_loops = zvx.copy()

        d_zvx = np.zeros((len(p_lats), len(p_longs) + 1))
        d_zvy = d_zvx.copy()
        d_zvz = d_zvx.copy()
        d2_zvx = np.zeros((len(p_lats), len(p_longs) + 1))
        d2_zvy = d_zvx.copy()
        d2_zvz = d_zvx.copy()
        d_mx = d_zvx.copy()
        d_my = d_zvx.copy()
        d_mz = d_zvx.copy()
        d_fax = d_zvx.copy()
        d_fay = d_zvx.copy()
        d_faz = d_zvx.copy()
        d2_mx = d_zvx.copy()
        d2_my = d_zvx.copy()
        d2_mz = d_zvx.copy()

        e_zvx = np.zeros((len(p_lats), len(p_longs) + 1))
        e_zvy = d_zvx.copy()
        e_zvz = d_zvx.copy()
        e_mx = d_zvx.copy()
        e_my = d_zvx.copy()
        e_mz = d_zvx.copy()
        e_fax = d_zvx.copy()
        e_fay = d_zvx.copy()
        e_faz = d_zvx.copy()

        date = dt.datetime(2000, 1, 1)
        # set up multi
        if self.dc is not None:
            targets = itertools.cycle(dc.ids)
            pending = []
            for i, p_lat in enumerate(p_lats):
                # iterate through target cyclicly and run commands
                print (i, p_lat)
                dview.targets = next(targets)
                pending.append(
                    dview.apply_async(OMMBV.calculate_mag_drift_unit_vectors_ecef, [p_lat]*len(p_longs), p_longs,
                                      p_alts, [date]*len(p_longs), full_output=True,
                                      include_debug=True))
            for i, p_lat in enumerate(p_lats):
                print ('collecting ', i, p_lat)
                # collect output
                tzx, tzy, tzz, tbx, tby, tbz, tmx, tmy, tmz, infod = pending.pop(0).get()
                zvx[i, :-1], zvy[i, :-1], zvz[i, :-1] = OMMBV.ecef_to_enu_vector(tzx, tzy, tzz,
                                                                                [p_lat]*len(p_longs), p_longs)
                bx[i, :-1], by[i, :-1], bz[i, :-1] = OMMBV.ecef_to_enu_vector(tbx, tby, tbz,
                                                                             [p_lat]*len(p_longs), p_longs)
                mx[i, :-1], my[i, :-1], mz[i, :-1] = OMMBV.ecef_to_enu_vector(tmx, tmy, tmz,
                                                                             [p_lat]*len(p_longs), p_longs)

                # pull out info about the vector generation
                grad_zon[i, :-1], grad_mer[i, :-1] = infod['diff_zonal_apex'], infod['diff_mer_apex']
                tol_zon[i, :-1], tol_mer[i, :-1] = infod['diff_zonal_vec'], infod['diff_mer_vec']
                init_type[i, :-1] = infod['vector_seed_type']
                num_loops[i, :-1] = infod['loops']

                # collect outputs on E and D vectors
                dzx, dzy, dzz = infod['d_zon_x'], infod['d_zon_y'], infod['d_zon_z']
                dfx, dfy, dfz = infod['d_fa_x'], infod['d_fa_y'], infod['d_fa_z']
                dmx, dmy, dmz = infod['d_mer_x'], infod['d_mer_y'], infod['d_mer_z']
                d_zvx[i, :-1], d_zvy[i, :-1], d_zvz[i, :-1] = OMMBV.ecef_to_enu_vector(dzx, dzy, dzz,
                                                                                      [p_lat]*len(p_longs), p_longs)
                dzx, dzy, dzz = infod['d_zon2_x'], infod['d_zon2_y'], infod['d_zon2_z']
                d2_zvx[i, :-1], d2_zvy[i, :-1], d2_zvz[i, :-1] = OMMBV.ecef_to_enu_vector(dzx, dzy, dzz,
                                                                                         [p_lat]*len(p_longs),
                                                                                         p_longs)
                d_fax[i, :-1], d_fay[i, :-1], d_faz[i, :-1] = OMMBV.ecef_to_enu_vector(dfx, dfy, dfz,
                                                                                      [p_lat]*len(p_longs), p_longs)
                d_mx[i, :-1], d_my[i, :-1], d_mz[i, :-1] = OMMBV.ecef_to_enu_vector(dmx, dmy, dmz,
                                                                                   [p_lat]*len(p_longs), p_longs)
                dmx, dmy, dmz = infod['d_mer2_x'], infod['d_mer2_y'], infod['d_mer2_z']
                d2_mx[i, :-1], d2_my[i, :-1], d2_mz[i, :-1] = OMMBV.ecef_to_enu_vector(dmx, dmy, dmz,
                                                                                      [p_lat]*len(p_longs), p_longs)

                ezx, ezy, ezz = infod['e_zon_x'], infod['e_zon_y'], infod['e_zon_z']
                efx, efy, efz = infod['e_fa_x'], infod['e_fa_y'], infod['e_fa_z']
                emx, emy, emz = infod['e_mer_x'], infod['e_mer_y'], infod['e_mer_z']
                e_zvx[i, :-1], e_zvy[i, :-1], e_zvz[i, :-1] = OMMBV.ecef_to_enu_vector(ezx, ezy, ezz,
                                                                                      [p_lat]*len(p_longs), p_longs)
                e_fax[i, :-1], e_fay[i, :-1], e_faz[i, :-1] = OMMBV.ecef_to_enu_vector(efx, efy, efz,
                                                                                      [p_lat]*len(p_longs), p_longs)
                e_mx[i, :-1], e_my[i, :-1], e_mz[i, :-1] = OMMBV.ecef_to_enu_vector(emx, emy, emz,
                                                                                   [p_lat]*len(p_longs), p_longs)

        else:
            for i, p_lat in enumerate(p_lats):
                print (i, p_lat)
                tzx, tzy, tzz, tbx, tby, tbz, tmx, tmy, tmz, infod = OMMBV.calculate_mag_drift_unit_vectors_ecef(
                                                                                        [p_lat]*len(p_longs), p_longs,
                                                                                        p_alts, [date]*len(p_longs),
                                                                                        full_output=True,
                                                                                        include_debug=True)
                zvx[i, :-1], zvy[i, :-1], zvz[i, :-1] = OMMBV.ecef_to_enu_vector(tzx, tzy, tzz,
                                                                                [p_lat]*len(p_longs), p_longs)
                bx[i, :-1], by[i, :-1], bz[i, :-1] = OMMBV.ecef_to_enu_vector(tbx, tby, tbz,
                                                                             [p_lat]*len(p_longs), p_longs)
                mx[i, :-1], my[i, :-1], mz[i, :-1] = OMMBV.ecef_to_enu_vector(tmx, tmy, tmz,
                                                                             [p_lat]*len(p_longs), p_longs)
                # pull out info about the vector generation
                grad_zon[i, :-1], grad_mer[i, :-1] = infod['diff_zonal_apex'], infod['diff_mer_apex']
                tol_zon[i, :-1], tol_mer[i, :-1] = infod['diff_zonal_vec'], infod['diff_mer_vec']
                init_type[i, :-1] = infod['vector_seed_type']
                num_loops[i, :-1] = infod['loops']

                # collect outputs on E and D vectors
                dzx, dzy, dzz = infod['d_zon_x'], infod['d_zon_y'], infod['d_zon_z']
                dfx, dfy, dfz = infod['d_fa_x'], infod['d_fa_y'], infod['d_fa_z']
                dmx, dmy, dmz = infod['d_mer_x'], infod['d_mer_y'], infod['d_mer_z']
                d_zvx[i, :-1], d_zvy[i, :-1], d_zvz[i, :-1] = OMMBV.ecef_to_enu_vector(dzx, dzy, dzz,
                                                                                      [p_lat]*len(p_longs),
                                                                                      p_longs)
                dzx, dzy, dzz = infod['d_zon2_x'], infod['d_zon2_y'], infod['d_zon2_z']
                d2_zvx[i, :-1], d2_zvy[i, :-1], d2_zvz[i, :-1] = OMMBV.ecef_to_enu_vector(dzx, dzy, dzz,
                                                                                         [p_lat]*len(p_longs),
                                                                                         p_longs)
                d_fax[i, :-1], d_fay[i, :-1], d_faz[i, :-1] = OMMBV.ecef_to_enu_vector(dfx, dfy, dfz,
                                                                                      [p_lat]*len(p_longs),
                                                                                      p_longs)
                d_mx[i, :-1], d_my[i, :-1], d_mz[i, :-1] = OMMBV.ecef_to_enu_vector(dmx, dmy, dmz,
                                                                                   [p_lat]*len(p_longs),
                                                                                   p_longs)
                dmx, dmy, dmz = infod['d_mer2_x'], infod['d_mer2_y'], infod['d_mer2_z']
                d2_mx[i, :-1], d2_my[i, :-1], d2_mz[i, :-1] = OMMBV.ecef_to_enu_vector(dmx, dmy, dmz,
                                                                                      [p_lat]*len(p_longs),
                                                                                      p_longs)

                ezx, ezy, ezz = infod['e_zon_x'], infod['e_zon_y'], infod['e_zon_z']
                efx, efy, efz = infod['e_fa_x'], infod['e_fa_y'], infod['e_fa_z']
                emx, emy, emz = infod['e_mer_x'], infod['e_mer_y'], infod['e_mer_z']
                e_zvx[i, :-1], e_zvy[i, :-1], e_zvz[i, :-1] = OMMBV.ecef_to_enu_vector(ezx, ezy, ezz,
                                                                                      [p_lat]*len(p_longs),
                                                                                      p_longs)
                e_fax[i, :-1], e_fay[i, :-1], e_faz[i, :-1] = OMMBV.ecef_to_enu_vector(efx, efy, efz,
                                                                                      [p_lat]*len(p_longs),
                                                                                      p_longs)
                e_mx[i, :-1], e_my[i, :-1], e_mz[i, :-1] = OMMBV.ecef_to_enu_vector(emx, emy, emz,
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
        grad_zon[:, -1] = grad_zon[:, 0]
        grad_mer[:, -1] = grad_mer[:, 0]
        tol_zon[:, -1] = tol_zon[:, 0]
        tol_mer[:, -1] = tol_mer[:, 0]
        init_type[:, -1] = init_type[:, 0]
        num_loops[:, -1] = num_loops[:, 0]

        d_zvx[:, -1] = d_zvx[:, 0]
        d_zvy[:, -1] = d_zvy[:, 0]
        d_zvz[:, -1] = d_zvz[:, 0]
        d2_zvx[:, -1] = d2_zvx[:, 0]
        d2_zvy[:, -1] = d2_zvy[:, 0]
        d2_zvz[:, -1] = d2_zvz[:, 0]
        d_fax[:, -1] = d_fax[:, 0]
        d_fay[:, -1] = d_fay[:, 0]
        d_faz[:, -1] = d_faz[:, 0]
        d_mx[:, -1] = d_mx[:, 0]
        d_my[:, -1] = d_my[:, 0]
        d_mz[:, -1] = d_mz[:, 0]
        d2_mx[:, -1] = d2_mx[:, 0]
        d2_my[:, -1] = d2_my[:, 0]
        d2_mz[:, -1] = d2_mz[:, 0]

        e_zvx[:, -1] = e_zvx[:, 0]
        e_zvy[:, -1] = e_zvy[:, 0]
        e_zvz[:, -1] = e_zvz[:, 0]
        e_fax[:, -1] = e_fax[:, 0]
        e_fay[:, -1] = e_fay[:, 0]
        e_faz[:, -1] = e_faz[:, 0]
        e_mx[:, -1] = e_mx[:, 0]
        e_my[:, -1] = e_my[:, 0]
        e_mz[:, -1] = e_mz[:, 0]

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
            plt.tight_layout();
            plt.savefig('zonal_east.pdf')
            plt.close()

            fig = plt.figure()
            plt.imshow(zvy, origin='lower')
            plt.colorbar()
            plt.yticks(ytickarr, ['-50', '-25', '0', '25', '50'])
            plt.xticks(xtickarr, ['0', '72', '144', '216', '288', '360'])
            plt.title('Zonal Unit Vector - Northward')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.tight_layout();
            plt.savefig('zonal_north.pdf')
            plt.close()

            fig = plt.figure()
            plt.imshow(zvz, origin='lower')
            plt.colorbar()
            plt.yticks(ytickarr, ['-50', '-25', '0', '25', '50'])
            plt.xticks(xtickarr, ['0', '72', '144', '216', '288', '360'])
            plt.title('Zonal Unit Vector - Upward')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.tight_layout();
            plt.savefig('zonal_up.pdf')
            plt.close()

            fig = plt.figure()
            plt.imshow(bx, origin='lower')
            plt.colorbar()
            plt.yticks(ytickarr, ['-50', '-25', '0', '25', '50'])
            plt.xticks(xtickarr, ['0', '72', '144', '216', '288', '360'])
            plt.title('Field-Aligned Unit Vector - Eastward')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.tight_layout();
            plt.savefig('fa_east.pdf')
            plt.close()

            fig = plt.figure()
            plt.imshow(by, origin='lower')
            plt.colorbar()
            plt.yticks(ytickarr, ['-50', '-25', '0', '25', '50'])
            plt.xticks(xtickarr, ['0', '72', '144', '216', '288', '360'])
            plt.title('Field-Aligned Unit Vector - Northward')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.tight_layout();
            plt.savefig('fa_north.pdf')
            plt.close()

            fig = plt.figure()
            plt.imshow(bz, origin='lower')
            plt.colorbar()
            plt.yticks(ytickarr, ['-50', '-25', '0', '25', '50'])
            plt.xticks(xtickarr, ['0', '72', '144', '216', '288', '360'])
            plt.title('Field-Aligned Unit Vector - Upward')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.tight_layout();
            plt.savefig('fa_up.pdf')
            plt.close()

            fig = plt.figure()
            plt.imshow(mx, origin='lower')
            plt.colorbar()
            plt.yticks(ytickarr, ['-50', '-25', '0', '25', '50'])
            plt.xticks(xtickarr, ['0', '72', '144', '216', '288', '360'])
            plt.title('Meridional Unit Vector - Eastward')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.tight_layout();
            plt.savefig('mer_east.pdf')
            plt.close()

            fig = plt.figure()
            plt.imshow(my, origin='lower')
            plt.colorbar()
            plt.yticks(ytickarr, ['-50', '-25', '0', '25', '50'])
            plt.xticks(xtickarr, ['0', '72', '144', '216', '288', '360'])
            plt.title('Meridional Unit Vector - Northward')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.tight_layout();
            plt.savefig('mer_north.pdf')
            plt.close()

            fig = plt.figure()
            plt.imshow(mz, origin='lower')
            plt.colorbar()
            plt.yticks(ytickarr, ['-50', '-25', '0', '25', '50'])
            plt.xticks(xtickarr, ['0', '72', '144', '216', '288', '360'])
            plt.title('Meridional Unit Vector - Upward')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.tight_layout();
            plt.savefig('mer_up.pdf')
            plt.close()

            # D Vectors
            fig = plt.figure()
            plt.imshow(np.sqrt(d_zvx ** 2 + d_zvy ** 2 + d_zvz ** 2), origin='lower')
            plt.colorbar()
            plt.yticks(ytickarr, ['-50', '-25', '0', '25', '50'])
            plt.xticks(xtickarr, ['0', '72', '144', '216', '288', '360'])
            plt.title('D Field Aligned Magnitude')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.tight_layout();
            plt.savefig('d_zon.pdf')
            plt.close()

            fig = plt.figure()
            plt.imshow(d_zvx, origin='lower')
            plt.colorbar()
            plt.yticks(ytickarr, ['-50', '-25', '0', '25', '50'])
            plt.xticks(xtickarr, ['0', '72', '144', '216', '288', '360'])
            plt.title('D Zonal Unit Vector - Eastward')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.tight_layout();
            plt.savefig('d_zonal_east.pdf')
            plt.close()

            fig = plt.figure()
            plt.imshow(d_zvy, origin='lower')
            plt.colorbar()
            plt.yticks(ytickarr, ['-50', '-25', '0', '25', '50'])
            plt.xticks(xtickarr, ['0', '72', '144', '216', '288', '360'])
            plt.title('D Zonal Unit Vector - Northward')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.tight_layout();
            plt.savefig('d_zonal_north.pdf')
            plt.close()

            fig = plt.figure()
            plt.imshow(d_zvz, origin='lower')
            plt.colorbar()
            plt.yticks(ytickarr, ['-50', '-25', '0', '25', '50'])
            plt.xticks(xtickarr, ['0', '72', '144', '216', '288', '360'])
            plt.title('D Zonal Unit Vector - Upward')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.tight_layout();
            plt.savefig('d_zonal_up.pdf')
            plt.close()

            fig = plt.figure()
            plt.imshow(np.sqrt(d_fax ** 2 + d_fay ** 2 + d_faz ** 2), origin='lower')
            plt.colorbar()
            plt.yticks(ytickarr, ['-50', '-25', '0', '25', '50'])
            plt.xticks(xtickarr, ['0', '72', '144', '216', '288', '360'])
            plt.title('D Field Aligned Magnitude')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.tight_layout();
            plt.savefig('d_fa.pdf')
            plt.close()

            fig = plt.figure()
            plt.imshow(d_fax, origin='lower')
            plt.colorbar()
            plt.yticks(ytickarr, ['-50', '-25', '0', '25', '50'])
            plt.xticks(xtickarr, ['0', '72', '144', '216', '288', '360'])
            plt.title('D Field Aligned Unit Vector - Eastward')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.tight_layout();
            plt.savefig('d_fa_east.pdf')
            plt.close()

            fig = plt.figure()
            plt.imshow(d_fay, origin='lower')
            plt.colorbar()
            plt.yticks(ytickarr, ['-50', '-25', '0', '25', '50'])
            plt.xticks(xtickarr, ['0', '72', '144', '216', '288', '360'])
            plt.title('D Field Aligned Unit Vector - Northward')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.tight_layout();
            plt.savefig('d_fa_north.pdf')
            plt.close()

            fig = plt.figure()
            plt.imshow(d_faz, origin='lower')
            plt.colorbar()
            plt.yticks(ytickarr, ['-50', '-25', '0', '25', '50'])
            plt.xticks(xtickarr, ['0', '72', '144', '216', '288', '360'])
            plt.title('D Field Aligned Unit Vector - Upward')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.tight_layout();
            plt.savefig('d_fa_up.pdf')
            plt.close()

            fig = plt.figure()
            plt.imshow(np.sqrt(d_mx ** 2 + d_my ** 2 + d_mz ** 2), origin='lower')
            plt.colorbar()
            plt.yticks(ytickarr, ['-50', '-25', '0', '25', '50'])
            plt.xticks(xtickarr, ['0', '72', '144', '216', '288', '360'])
            plt.title('D Meridional Magnitude')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.tight_layout();
            plt.savefig('d_mer.pdf')
            plt.close()

            fig = plt.figure()
            plt.imshow(d_mx, origin='lower')
            plt.colorbar()
            plt.yticks(ytickarr, ['-50', '-25', '0', '25', '50'])
            plt.xticks(xtickarr, ['0', '72', '144', '216', '288', '360'])
            plt.title('D Meridional Unit Vector - Eastward')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.tight_layout();
            plt.savefig('d_mer_east.pdf')
            plt.close()

            fig = plt.figure()
            plt.imshow(d_my, origin='lower')
            plt.colorbar()
            plt.yticks(ytickarr, ['-50', '-25', '0', '25', '50'])
            plt.xticks(xtickarr, ['0', '72', '144', '216', '288', '360'])
            plt.title('D Meridional Unit Vector - Northward')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.tight_layout();
            plt.savefig('d_mer_north.pdf')
            plt.close()

            fig = plt.figure()
            plt.imshow(d_mz, origin='lower')
            plt.colorbar()
            plt.yticks(ytickarr, ['-50', '-25', '0', '25', '50'])
            plt.xticks(xtickarr, ['0', '72', '144', '216', '288', '360'])
            plt.title('D Meridional Unit Vector - Upward')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.tight_layout();
            plt.savefig('d_mer_up.pdf')
            plt.close()

            fig = plt.figure()
            dmag = np.sqrt(d_mx ** 2 + d_my ** 2 + d_mz ** 2)
            dmag2 = np.sqrt(d2_mx ** 2 + d2_my ** 2 + d2_mz ** 2)
            plt.imshow(np.log10(np.abs(dmag - dmag2) / dmag), origin='lower')
            plt.colorbar()
            plt.yticks(ytickarr, ['-50', '-25', '0', '25', '50'])
            plt.xticks(xtickarr, ['0', '72', '144', '216', '288', '360'])
            plt.title('Log D Meridional Vector Normalized Difference')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.tight_layout();
            plt.savefig('d_diff_mer_norm.pdf')
            plt.close()

            fig = plt.figure()
            dmag = np.sqrt(d_mx ** 2 + d_my ** 2 + d_mz ** 2)
            plt.imshow(np.log10(np.abs(d2_mx - d_mx)), origin='lower')
            plt.colorbar()
            plt.yticks(ytickarr, ['-50', '-25', '0', '25', '50'])
            plt.xticks(xtickarr, ['0', '72', '144', '216', '288', '360'])
            plt.title('Log D Meridional Vector Difference - Eastward')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.tight_layout();
            plt.savefig('d_diff_mer_east.pdf')
            plt.close()

            fig = plt.figure()
            plt.imshow(np.log10(np.abs(d2_mx - d_mx) / dmag), origin='lower')
            plt.colorbar()
            plt.yticks(ytickarr, ['-50', '-25', '0', '25', '50'])
            plt.xticks(xtickarr, ['0', '72', '144', '216', '288', '360'])
            plt.title('Log D Meridional Vector Difference - Eastward')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.tight_layout();
            plt.savefig('d_diff_mer_east_norm.pdf')
            plt.close()

            fig = plt.figure()
            plt.imshow(np.log10(np.abs(d2_my - d_my)), origin='lower')
            plt.colorbar()
            plt.yticks(ytickarr, ['-50', '-25', '0', '25', '50'])
            plt.xticks(xtickarr, ['0', '72', '144', '216', '288', '360'])
            plt.title('Log D Meridional Vector Difference - Northward')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.tight_layout();
            plt.savefig('d_diff_mer_north.pdf')
            plt.close()

            fig = plt.figure()
            plt.imshow(np.log10(np.abs(d2_my - d_my) / dmag), origin='lower')
            plt.colorbar()
            plt.yticks(ytickarr, ['-50', '-25', '0', '25', '50'])
            plt.xticks(xtickarr, ['0', '72', '144', '216', '288', '360'])
            plt.title('Log D Meridional Vector Difference - Northward')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.tight_layout();
            plt.savefig('d_diff_mer_north_norm.pdf')
            plt.close()

            fig = plt.figure()
            plt.imshow(np.log10(np.abs(d2_mz - d_mz)), origin='lower')
            plt.colorbar()
            plt.yticks(ytickarr, ['-50', '-25', '0', '25', '50'])
            plt.xticks(xtickarr, ['0', '72', '144', '216', '288', '360'])
            plt.title('Log D Meridional Vector Difference - Upward')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.tight_layout();
            plt.savefig('d_diff_mer_up.pdf')
            plt.close()

            fig = plt.figure()
            plt.imshow(np.log10(np.abs(d2_mz - d_mz) / dmag), origin='lower')
            plt.colorbar()
            plt.yticks(ytickarr, ['-50', '-25', '0', '25', '50'])
            plt.xticks(xtickarr, ['0', '72', '144', '216', '288', '360'])
            plt.title('Log D Meridional Vector Difference - Upward')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.tight_layout();
            plt.savefig('d_diff_mer_up_norm.pdf')
            plt.close()

            fig = plt.figure()
            dmag = np.sqrt(d_zvx ** 2 + d_zvy ** 2 + d_zvz ** 2)
            dmag2 = np.sqrt(d2_zvx ** 2 + d2_zvy ** 2 + d2_zvz ** 2)
            plt.imshow(np.log10(np.abs(dmag2 - dmag) / dmag), origin='lower')
            plt.colorbar()
            plt.yticks(ytickarr, ['-50', '-25', '0', '25', '50'])
            plt.xticks(xtickarr, ['0', '72', '144', '216', '288', '360'])
            plt.title('Log D Zonal Vector Normalized Difference')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.tight_layout();
            plt.savefig('d_diff_zon_norm.pdf')
            plt.close()

            fig = plt.figure()
            dmag = np.sqrt(d_zvx ** 2 + d_zvy ** 2 + d_zvz ** 2)
            plt.imshow(np.log10(np.abs(d2_zvx - d_zvx)), origin='lower')
            plt.colorbar()
            plt.yticks(ytickarr, ['-50', '-25', '0', '25', '50'])
            plt.xticks(xtickarr, ['0', '72', '144', '216', '288', '360'])
            plt.title('Log D Zonal Vector Difference - Eastward')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.tight_layout();
            plt.savefig('d_diff_zon_east.pdf')
            plt.close()

            fig = plt.figure()
            plt.imshow(np.log10(np.abs(d2_zvy - d_zvy)), origin='lower')
            plt.colorbar()
            plt.yticks(ytickarr, ['-50', '-25', '0', '25', '50'])
            plt.xticks(xtickarr, ['0', '72', '144', '216', '288', '360'])
            plt.title('Log D Zonal Vector Difference - Northward')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.tight_layout();
            plt.savefig('d_diff_zon_north.pdf')
            plt.close()

            fig = plt.figure()
            plt.imshow(np.log10(np.abs(d2_zvz - d_zvz)), origin='lower')
            plt.colorbar()
            plt.yticks(ytickarr, ['-50', '-25', '0', '25', '50'])
            plt.xticks(xtickarr, ['0', '72', '144', '216', '288', '360'])
            plt.title('Log D Zonal Vector Difference - Upward')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.tight_layout();
            plt.savefig('d_diff_zon_up.pdf')
            plt.close()

            plt.imshow(np.log10(np.abs(d2_zvx - d_zvx) / dmag), origin='lower')
            plt.colorbar()
            plt.yticks(ytickarr, ['-50', '-25', '0', '25', '50'])
            plt.xticks(xtickarr, ['0', '72', '144', '216', '288', '360'])
            plt.title('Log D Zonal Vector Difference - Eastward')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.tight_layout();
            plt.savefig('d_diff_zon_east_norm.pdf')
            plt.close()

            fig = plt.figure()
            plt.imshow(np.log10(np.abs(d2_zvy - d_zvy) / dmag), origin='lower')
            plt.colorbar()
            plt.yticks(ytickarr, ['-50', '-25', '0', '25', '50'])
            plt.xticks(xtickarr, ['0', '72', '144', '216', '288', '360'])
            plt.title('Log D Zonal Vector Difference - Northward')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.tight_layout();
            plt.savefig('d_diff_zon_north_norm.pdf')
            plt.close()

            fig = plt.figure()
            plt.imshow(np.log10(np.abs(d2_zvz - d_zvz) / dmag), origin='lower')
            plt.colorbar()
            plt.yticks(ytickarr, ['-50', '-25', '0', '25', '50'])
            plt.xticks(xtickarr, ['0', '72', '144', '216', '288', '360'])
            plt.title('Log D Zonal Vector Difference - Upward')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.tight_layout();
            plt.savefig('d_diff_zon_up_norm.pdf')
            plt.close()

            # E Vectors
            fig = plt.figure()
            plt.imshow(np.sqrt(e_zvx ** 2 + e_zvy ** 2 + e_zvz ** 2), origin='lower')
            plt.colorbar()
            plt.yticks(ytickarr, ['-50', '-25', '0', '25', '50'])
            plt.xticks(xtickarr, ['0', '72', '144', '216', '288', '360'])
            plt.title('E Field Aligned Magnitude')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.tight_layout();
            plt.savefig('e_zon.pdf')
            plt.close()

            fig = plt.figure()
            plt.imshow(e_zvx, origin='lower')
            plt.colorbar()
            plt.yticks(ytickarr, ['-50', '-25', '0', '25', '50'])
            plt.xticks(xtickarr, ['0', '72', '144', '216', '288', '360'])
            plt.title('E Zonal Unit Vector - Eastward')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.tight_layout();
            plt.savefig('e_zonal_east.pdf')
            plt.close()

            fig = plt.figure()
            plt.imshow(e_zvy, origin='lower')
            plt.colorbar()
            plt.yticks(ytickarr, ['-50', '-25', '0', '25', '50'])
            plt.xticks(xtickarr, ['0', '72', '144', '216', '288', '360'])
            plt.title('E Zonal Unit Vector - Northward')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.tight_layout();
            plt.savefig('e_zonal_north.pdf')
            plt.close()

            fig = plt.figure()
            plt.imshow(e_zvz, origin='lower')
            plt.colorbar()
            plt.yticks(ytickarr, ['-50', '-25', '0', '25', '50'])
            plt.xticks(xtickarr, ['0', '72', '144', '216', '288', '360'])
            plt.title('E Zonal Unit Vector - Upward')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.tight_layout();
            plt.savefig('e_zonal_up.pdf')
            plt.close()

            fig = plt.figure()
            plt.imshow(np.sqrt(e_fax ** 2 + e_fay ** 2 + e_faz ** 2), origin='lower')
            plt.colorbar()
            plt.yticks(ytickarr, ['-50', '-25', '0', '25', '50'])
            plt.xticks(xtickarr, ['0', '72', '144', '216', '288', '360'])
            plt.title('E Field Aligned Magnitude')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.tight_layout();
            plt.savefig('e_fa.pdf')
            plt.close()

            fig = plt.figure()
            plt.imshow(e_fax, origin='lower')
            plt.colorbar()
            plt.yticks(ytickarr, ['-50', '-25', '0', '25', '50'])
            plt.xticks(xtickarr, ['0', '72', '144', '216', '288', '360'])
            plt.title('E Field Aligned Unit Vector - Eastward')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.tight_layout();
            plt.savefig('e_fa_east.pdf')
            plt.close()

            fig = plt.figure()
            plt.imshow(e_fay, origin='lower')
            plt.colorbar()
            plt.yticks(ytickarr, ['-50', '-25', '0', '25', '50'])
            plt.xticks(xtickarr, ['0', '72', '144', '216', '288', '360'])
            plt.title('E Field Aligned Unit Vector - Northward')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.tight_layout();
            plt.savefig('e_fa_north.pdf')
            plt.close()

            fig = plt.figure()
            plt.imshow(e_faz, origin='lower')
            plt.colorbar()
            plt.yticks(ytickarr, ['-50', '-25', '0', '25', '50'])
            plt.xticks(xtickarr, ['0', '72', '144', '216', '288', '360'])
            plt.title('E Field Aligned Unit Vector - Upward')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.tight_layout();
            plt.savefig('e_fa_up.pdf')
            plt.close()

            fig = plt.figure()
            plt.imshow(np.sqrt(e_mx ** 2 + e_my ** 2 + e_mz ** 2), origin='lower')
            plt.colorbar()
            plt.yticks(ytickarr, ['-50', '-25', '0', '25', '50'])
            plt.xticks(xtickarr, ['0', '72', '144', '216', '288', '360'])
            plt.title('E Meridional Magnitude')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.tight_layout();
            plt.savefig('e_mer.pdf')
            plt.close()

            fig = plt.figure()
            plt.imshow(e_mx, origin='lower')
            plt.colorbar()
            plt.yticks(ytickarr, ['-50', '-25', '0', '25', '50'])
            plt.xticks(xtickarr, ['0', '72', '144', '216', '288', '360'])
            plt.title('E Meridional Unit Vector - Eastward')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.tight_layout();
            plt.savefig('e_mer_east.pdf')
            plt.close()

            fig = plt.figure()
            plt.imshow(e_my, origin='lower')
            plt.colorbar()
            plt.yticks(ytickarr, ['-50', '-25', '0', '25', '50'])
            plt.xticks(xtickarr, ['0', '72', '144', '216', '288', '360'])
            plt.title('E Meridional Unit Vector - Northward')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.tight_layout();
            plt.savefig('e_mer_north.pdf')
            plt.close()

            fig = plt.figure()
            plt.imshow(e_mz, origin='lower')
            plt.colorbar()
            plt.yticks(ytickarr, ['-50', '-25', '0', '25', '50'])
            plt.xticks(xtickarr, ['0', '72', '144', '216', '288', '360'])
            plt.title('E Meridional Unit Vector - Upward')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.tight_layout();
            plt.savefig('e_mer_up.pdf')
            plt.close()

            # Kroenecker Delta Vectors
            fig = plt.figure()
            plt.imshow(np.log10(e_zvx*d_zvx + e_zvy*d_zvy + e_zvz*d_zvz), origin='lower')
            plt.colorbar()
            plt.yticks(ytickarr, ['-50', '-25', '0', '25', '50'])
            plt.xticks(xtickarr, ['0', '72', '144', '216', '288', '360'])
            plt.title('ED Zonal - Eastward')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.tight_layout();
            plt.savefig('ed_dot_zonal.pdf')
            plt.close()

            fig = plt.figure()
            plt.imshow(np.log10(e_fax*d_fax + e_fay*d_fay + e_faz*d_faz), origin='lower')
            plt.colorbar()
            plt.yticks(ytickarr, ['-50', '-25', '0', '25', '50'])
            plt.xticks(xtickarr, ['0', '72', '144', '216', '288', '360'])
            plt.title('ED Field Aligned - Eastward')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.tight_layout();
            plt.savefig('ed_dot_fa.pdf')
            plt.close()

            fig = plt.figure()
            plt.imshow(np.log10(e_mx*d_mx + e_my*d_my + e_mz*d_mz), origin='lower')
            plt.colorbar()
            plt.yticks(ytickarr, ['-50', '-25', '0', '25', '50'])
            plt.xticks(xtickarr, ['0', '72', '144', '216', '288', '360'])
            plt.title('ED Meridional - Eastward')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.tight_layout();
            plt.savefig('ed_dot_mer.pdf')
            plt.close()

            fig = plt.figure()
            plt.imshow(np.log10(np.abs(grad_zon)), origin='lower')
            plt.colorbar()
            plt.yticks(ytickarr, ['-50', '-25', '0', '25', '50'])
            plt.xticks(xtickarr, ['0', '72', '144', '216', '288', '360'])
            plt.title('Log Gradient in Apex Height (km/km) - Zonal')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.tight_layout();
            plt.savefig('unit_vector_grad_zonal.pdf')
            plt.close()

            fig = plt.figure()
            plt.imshow(np.log10(np.abs(grad_mer)), origin='lower')
            plt.colorbar()
            plt.yticks(ytickarr, ['-50', '-25', '0', '25', '50'])
            plt.xticks(xtickarr, ['0', '72', '144', '216', '288', '360'])
            plt.title('Log Gradient in Apex Height (km/km) - Meridional')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.tight_layout();
            plt.savefig('unit_vector_grad_meridional.pdf')
            plt.close()

            fig = plt.figure()
            plt.imshow(np.log10(np.abs(tol_zon)), origin='lower')
            plt.colorbar()
            plt.yticks(ytickarr, ['-50', '-25', '0', '25', '50'])
            plt.xticks(xtickarr, ['0', '72', '144', '216', '288', '360'])
            plt.title('Log Achieved Tolerance - Zonal Unit Vector')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.tight_layout();
            plt.savefig('unit_vector_tol_zonal.pdf')
            plt.close()

            fig = plt.figure()
            plt.imshow(np.log10(np.abs(tol_mer)), origin='lower')
            plt.colorbar()
            plt.yticks(ytickarr, ['-50', '-25', '0', '25', '50'])
            plt.xticks(xtickarr, ['0', '72', '144', '216', '288', '360'])
            plt.title('Log Achieved Tolerance - Meridional')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.tight_layout();
            plt.savefig('unit_vector_tol_meridional.pdf')
            plt.close()

            fig = plt.figure()
            plt.imshow(init_type, origin='lower')
            plt.colorbar()
            plt.yticks(ytickarr, ['-50', '-25', '0', '25', '50'])
            plt.xticks(xtickarr, ['0', '72', '144', '216', '288', '360'])
            plt.title('Initial Seed Vector Type')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.tight_layout();
            plt.savefig('unit_vector_seed_vector_type.pdf')
            plt.close()

            fig = plt.figure()
            plt.imshow(num_loops, origin='lower')
            plt.colorbar()
            plt.yticks(ytickarr, ['-50', '-25', '0', '25', '50'])
            plt.xticks(xtickarr, ['0', '72', '144', '216', '288', '360'])
            plt.title('Number of Iterative Loops')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.tight_layout();
            plt.savefig('unit_vector_num_loops.pdf')
            plt.close()


        except:
            pass

        assert np.all(np.abs(tol_zon) <= 1.E-4)
        assert np.all(np.abs(tol_mer) <= 1.E-4)
        assert np.all(np.abs(grad_zon) <= 1.E-4)

    def test_unit_vector_component_plots_edge_steps(self):
        """Check precision D of vectors as edge_steps increased"""
        import matplotlib.pyplot as plt

        p_lats, p_longs, p_alts = gen_plot_grid_fixed_alt(550.)
        # data returned are the locations along each direction
        # the full range of points obtained by iterating over all
        # recasting alts into a more convenient form for later calculation
        p_alts = [p_alts[0]]*len(p_longs)

        d_zvx = np.zeros((len(p_lats), len(p_longs) + 1))
        d_zvy = d_zvx.copy();
        d_zvz = d_zvx.copy()
        d2_zvx = np.zeros((len(p_lats), len(p_longs) + 1))
        d2_zvy = d_zvx.copy();
        d2_zvz = d_zvx.copy()
        d_mx = d_zvx.copy();
        d_my = d_zvx.copy();
        d_mz = d_zvx.copy()
        d_fax = d_zvx.copy();
        d_fay = d_zvx.copy();
        d_faz = d_zvx.copy()
        d2_mx = d_zvx.copy();
        d2_my = d_zvx.copy();
        d2_mz = d_zvx.copy()

        date = dt.datetime(2000, 1, 1)
        # set up multi
        if self.dc is not None:
            targets = itertools.cycle(dc.ids)
            pending = []
            for i, p_lat in enumerate(p_lats):
                # iterate through target cyclicly and run commands
                print (i, p_lat)
                dview.targets = next(targets)
                pending.append(
                    dview.apply_async(OMMBV.calculate_mag_drift_unit_vectors_ecef,
                                      [p_lat]*len(p_longs), p_longs,
                                      p_alts, [date]*len(p_longs), full_output=True,
                                      include_debug=True, edge_steps=5))
            for i, p_lat in enumerate(p_lats):
                print ('collecting ', i, p_lat)
                # collect output
                tzx, tzy, tzz, tbx, tby, tbz, tmx, tmy, tmz, infod = pending.pop(0).get()

                # collect outputs on E and D vectors
                dzx, dzy, dzz = infod['d_zon_x'], infod['d_zon_y'], infod['d_zon_z']
                dfx, dfy, dfz = infod['d_fa_x'], infod['d_fa_y'], infod['d_fa_z']
                dmx, dmy, dmz = infod['d_mer_x'], infod['d_mer_y'], infod['d_mer_z']
                d_zvx[i, :-1], d_zvy[i, :-1], d_zvz[i, :-1] = OMMBV.ecef_to_enu_vector(dzx, dzy, dzz,
                                                                                      [p_lat]*len(p_longs),
                                                                                      p_longs)
                dzx, dzy, dzz = infod['d_zon2_x'], infod['d_zon2_y'], infod['d_zon2_z']
                d2_zvx[i, :-1], d2_zvy[i, :-1], d2_zvz[i, :-1] = OMMBV.ecef_to_enu_vector(dzx, dzy, dzz,
                                                                                         [p_lat]*len(p_longs),
                                                                                         p_longs)
                d_fax[i, :-1], d_fay[i, :-1], d_faz[i, :-1] = OMMBV.ecef_to_enu_vector(dfx, dfy, dfz,
                                                                                      [p_lat]*len(p_longs),
                                                                                      p_longs)
                d_mx[i, :-1], d_my[i, :-1], d_mz[i, :-1] = OMMBV.ecef_to_enu_vector(dmx, dmy, dmz,
                                                                                   [p_lat]*len(p_longs),
                                                                                   p_longs)
                dmx, dmy, dmz = infod['d_mer2_x'], infod['d_mer2_y'], infod['d_mer2_z']
                d2_mx[i, :-1], d2_my[i, :-1], d2_mz[i, :-1] = OMMBV.ecef_to_enu_vector(dmx, dmy, dmz,
                                                                                      [p_lat]*len(p_longs),
                                                                                      p_longs)


        else:
            for i, p_lat in enumerate(p_lats):
                print (i, p_lat)
                tzx, tzy, tzz, tbx, tby, tbz, tmx, tmy, tmz, infod = OMMBV.calculate_mag_drift_unit_vectors_ecef(
                                                                                        [p_lat]*len(p_longs), p_longs,
                                                                                        p_alts, [date]*len(p_longs),
                                                                                        full_output=True,
                                                                                        include_debug=True,
                                                                                        edge_steps=5)

                # collect outputs on E and D vectors
                dzx, dzy, dzz = infod['d_zon_x'], infod['d_zon_y'], infod['d_zon_z']
                dfx, dfy, dfz = infod['d_fa_x'], infod['d_fa_y'], infod['d_fa_z']
                dmx, dmy, dmz = infod['d_mer_x'], infod['d_mer_y'], infod['d_mer_z']
                d_zvx[i, :-1], d_zvy[i, :-1], d_zvz[i, :-1] = OMMBV.ecef_to_enu_vector(dzx, dzy, dzz,
                                                                                      [p_lat]*len(p_longs), p_longs)
                dzx, dzy, dzz = infod['d_zon2_x'], infod['d_zon2_y'], infod['d_zon2_z']
                d2_zvx[i, :-1], d2_zvy[i, :-1], d2_zvz[i, :-1] = OMMBV.ecef_to_enu_vector(dzx, dzy, dzz,
                                                                                         [p_lat]*len(p_longs),
                                                                                         p_longs)
                d_fax[i, :-1], d_fay[i, :-1], d_faz[i, :-1] = OMMBV.ecef_to_enu_vector(dfx, dfy, dfz,
                                                                                      [p_lat]*len(p_longs), p_longs)
                d_mx[i, :-1], d_my[i, :-1], d_mz[i, :-1] = OMMBV.ecef_to_enu_vector(dmx, dmy, dmz,
                                                                                   [p_lat]*len(p_longs), p_longs)
                dmx, dmy, dmz = infod['d_mer2_x'], infod['d_mer2_y'], infod['d_mer2_z']
                d2_mx[i, :-1], d2_my[i, :-1], d2_mz[i, :-1] = OMMBV.ecef_to_enu_vector(dmx, dmy, dmz,
                                                                                      [p_lat]*len(p_longs), p_longs)

        # account for periodicity

        d_zvx[:, -1] = d_zvx[:, 0]
        d_zvy[:, -1] = d_zvy[:, 0]
        d_zvz[:, -1] = d_zvz[:, 0]
        d2_zvx[:, -1] = d2_zvx[:, 0]
        d2_zvy[:, -1] = d2_zvy[:, 0]
        d2_zvz[:, -1] = d2_zvz[:, 0]
        d_fax[:, -1] = d_fax[:, 0]
        d_fay[:, -1] = d_fay[:, 0]
        d_faz[:, -1] = d_faz[:, 0]
        d_mx[:, -1] = d_mx[:, 0]
        d_my[:, -1] = d_my[:, 0]
        d_mz[:, -1] = d_mz[:, 0]
        d2_mx[:, -1] = d2_mx[:, 0]
        d2_my[:, -1] = d2_my[:, 0]
        d2_mz[:, -1] = d2_mz[:, 0]

        ytickarr = np.array([0, 0.25, 0.5, 0.75, 1])*(len(p_lats) - 1)
        xtickarr = np.array([0, 0.2, 0.4, 0.6, 0.8, 1])*len(p_longs)

        try:
            fig = plt.figure()
            dmag = np.sqrt(d_mx ** 2 + d_my ** 2 + d_mz ** 2)
            dmag2 = np.sqrt(d2_mx ** 2 + d2_my ** 2 + d2_mz ** 2)
            plt.imshow(np.log10(np.abs(dmag - dmag2) / dmag), origin='lower')
            plt.colorbar()
            plt.yticks(ytickarr, ['-50', '-25', '0', '25', '50'])
            plt.xticks(xtickarr, ['0', '72', '144', '216', '288', '360'])
            plt.title('Log D Meridional Vector Normalized Difference')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.tight_layout();
            plt.savefig('d_diff_mer_norm_edgesteps.pdf')
            plt.close()

            fig = plt.figure()
            dmag = np.sqrt(d_zvx ** 2 + d_zvy ** 2 + d_zvz ** 2)
            dmag2 = np.sqrt(d2_zvx ** 2 + d2_zvy ** 2 + d2_zvz ** 2)
            plt.imshow(np.log10(np.abs(dmag2 - dmag) / dmag), origin='lower')
            plt.colorbar()
            plt.yticks(ytickarr, ['-50', '-25', '0', '25', '50'])
            plt.xticks(xtickarr, ['0', '72', '144', '216', '288', '360'])
            plt.title('Log D Zonal Vector Normalized Difference')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.tight_layout();
            plt.savefig('d_diff_zon_norm_edegesteps.pdf')
            plt.close()
        except:
            pass

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
            targets = itertools.cycle(dc.ids)
            pending = []
            for i, p_lat in enumerate(p_lats):
                # iterate through target cyclicly and run commands
                print (i, p_lat)
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
                tzx, tzy, tzz, tbx, tby, tbz, tmx, tmy, tmz = OMMBV.calculate_mag_drift_unit_vectors_ecef(
                                                                        [p_lat]*len(p_longs), p_longs,
                                                                        p_alts, [date]*len(p_longs),
                                                                        step_size=1.)
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
                    p_alts, [date]*len(p_longs),
                    step_size=2.)
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
            plt.tight_layout();
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
            plt.tight_layout();
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
            plt.tight_layout();
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
            plt.tight_layout();
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
            plt.tight_layout();
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
            plt.tight_layout();
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
            plt.tight_layout();
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
            plt.tight_layout();
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
            plt.tight_layout();
            plt.savefig('mer_up_diff.pdf')
            plt.close()

            # calculate mean and standard deviation and then plot those
            plt.figure()
            plt.errorbar(p_longs, np.log10(np.nanmedian(np.abs(zvx[:, :-1]), axis=0)),
                         yerr=np.abs(np.log10(np.nanstd(zvx[:, :-1], axis=0))), label='East')
            plt.errorbar(p_longs, np.log10(np.nanmedian(np.abs(zvy[:, :-1]), axis=0)),
                         yerr=np.abs(np.log10(np.nanstd(zvy[:, :-1], axis=0))), label='North')
            plt.errorbar(p_longs, np.log10(np.nanmedian(np.abs(zvz[:, :-1]), axis=0)),
                         yerr=np.abs(np.log10(np.nanstd(zvz[:, :-1], axis=0))), label='Up')
            plt.xlabel('Longitude (Degrees)')
            plt.ylabel('Log Change in Zonal Vector')
            plt.title("Sensitivity of Zonal Unit Vector")
            plt.legend()
            plt.tight_layout()
            plt.tight_layout();
            plt.savefig('zonal_diff_v_longitude.pdf')
            plt.close()

            # calculate mean and standard deviation and then plot those
            plt.figure()
            plt.errorbar(p_longs, np.log10(np.nanmedian(np.abs(mx[:, :-1]), axis=0)),
                         yerr=np.abs(np.log10(np.nanstd(mx[:, :-1], axis=0))), label='East')
            plt.errorbar(p_longs, np.log10(np.nanmedian(np.abs(my[:, :-1]), axis=0)),
                         yerr=np.abs(np.log10(np.nanstd(my[:, :-1], axis=0))), label='North')
            plt.errorbar(p_longs, np.log10(np.nanmedian(np.abs(mz[:, :-1]), axis=0)),
                         yerr=np.abs(np.log10(np.nanstd(mz[:, :-1], axis=0))), label='Up')
            plt.xlabel('Longitude (Degrees)')
            plt.ylabel('Log Change in Meridional Vector')
            plt.title("Sensitivity of Meridional Unit Vector")
            plt.legend()
            plt.tight_layout()
            plt.tight_layout();
            plt.savefig('mer_diff_v_longitude.pdf')
            plt.close()

        except:
            print('Skipping plots due to error.')

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
                in_x, in_y, in_z = OMMBV.geodetic_to_ecef([p_lat]*len(p_longs), p_longs, p_alts)
                pending.append(dview.apply_async(OMMBV.step_along_mag_unit_vector,
                                                 in_x, in_y, in_z, dates,
                                                 direction=direction,
                                                 num_steps=5, step_size=25. / 5.))
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
                pending.append(dview.apply_async(OMMBV.apex_location_info,
                                                 tlat, tlon, talt, dates,
                                                 return_geodetic=True))
                # convert all locations to geodetic coordinates
                tlat, tlon, talt = OMMBV.ecef_to_geodetic(x2[i, :-1], y2[i, :-1], z2[i, :-1])
                pending.append(dview.apply_async(OMMBV.apex_location_info,
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
                in_x, in_y, in_z = OMMBV.geodetic_to_ecef([p_lat]*len(p_longs), p_longs, p_alts)

                x[i, :-1], y[i, :-1], z[i, :-1] = OMMBV.step_along_mag_unit_vector(in_x, in_y, in_z, dates,
                                                                                  direction=direction,
                                                                                  num_steps=5, step_size=25. / 5.)
                # second run
                x2[i, :-1], y2[i, :-1], z2[i, :-1] = OMMBV.step_along_mag_unit_vector(in_x, in_y, in_z, dates,
                                                                                     direction=direction,
                                                                                     num_steps=1, step_size=25. / 1.)

            for i, p_lat in enumerate(p_lats):
                # convert all locations to geodetic coordinates
                tlat, tlon, talt = OMMBV.ecef_to_geodetic(x[i, :-1], y[i, :-1], z[i, :-1])
                x[i, :-1], y[i, :-1], z[i, :-1], _, _, h[i, :-1] = OMMBV.apex_location_info(tlat, tlon, talt, dates,
                                                                                           return_geodetic=True)
                # convert all locations to geodetic coordinates
                tlat, tlon, talt = OMMBV.ecef_to_geodetic(x2[i, :-1], y2[i, :-1], z2[i, :-1])
                x2[i, :-1], y2[i, :-1], z2[i, :-1], _, _, h2[i, :-1] = OMMBV.apex_location_info(tlat, tlon, talt, dates,
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
            plt.tight_layout();
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
            plt.tight_layout();
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
            plt.tight_layout();
            plt.savefig(direction + '_step_diff_apex_height_z.pdf')
            plt.close()

            fig = plt.figure()
            plt.imshow(np.log10(np.sqrt(x ** 2 + y ** 2 + z ** 2)), origin='lower')
            plt.colorbar()
            plt.yticks(ytickarr, ['-50', '-25', '0', '25', '50'])
            plt.xticks(xtickarr, ['0', '72', '144', '216', '288', '360'])
            plt.title('Log Difference in Apex Position After Stepping')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.tight_layout();
            plt.savefig(direction + '_step_diff_apex_height_r.pdf')
            plt.close()

            # calculate mean and standard deviation and then plot those
            fig = plt.figure()
            yerrx = np.nanstd(np.log10(x[:, :-1]), axis=0)
            yerry = np.nanstd(np.log10(y[:, :-1]), axis=0)
            yerrz = np.nanstd(np.log10(z[:, :-1]), axis=0)
            vals = np.log10(np.nanmedian(np.abs(x[:, :-1]), axis=0))
            plt.errorbar(p_longs, vals,
                         yerr=yerrx - vals, label='x')
            vals = np.log10(np.nanmedian(np.abs(y[:, :-1]), axis=0))
            plt.errorbar(p_longs, vals,
                         yerr=yerry - vals, label='y')
            vals = np.log10(np.nanmedian(np.abs(z[:, :-1]), axis=0))
            plt.errorbar(p_longs, vals,
                         yerr=yerrz - vals, label='z')
            plt.xlabel('Longitude (Degrees)')
            plt.ylabel('Change in ECEF (km)')
            plt.title('Log Median Difference in Apex Position')
            plt.legend()
            plt.tight_layout()
            plt.tight_layout();
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
            plt.tight_layout();
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
                print (i, p_lat)
                dview.targets = next(targets)
                pending.append(dview.apply_async(OMMBV.scalars_for_mapping_ion_drifts,
                                                 [p_lat]*len(p_longs), p_longs,
                                                 p_alts, [date]*len(p_longs)))
            for i, p_lat in enumerate(p_lats):
                print ('collecting ', i, p_lat)
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
                print (i, p_lat)
                scalars = OMMBV.scalars_for_mapping_ion_drifts([p_lat]*len(p_longs), p_longs,
                                                              p_alts, [date]*len(p_longs))
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
            plt.tight_layout();
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
            plt.tight_layout();
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
            plt.tight_layout();
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
            plt.tight_layout();
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
            plt.tight_layout();
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
            plt.tight_layout();
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
            plt.tight_layout();
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
            plt.tight_layout();
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
            plt.tight_layout();
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
            plt.tight_layout();
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
            plt.tight_layout();
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
            plt.tight_layout();
            plt.savefig('south_mer_drift.pdf')
            plt.close()

        except:
            pass

    def test_heritage_geomag_efield_scalars_plots(self):
        """Summary plots of the heritage code path for scaling electric fields and ion drifts"""
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
                print (i, p_lat)
                dview.targets = next(targets)
                pending.append(
                    dview.apply_async(OMMBV.heritage_scalars_for_mapping_ion_drifts,
                                      [p_lat]*len(p_longs), p_longs,
                                      p_alts, [date]*len(p_longs)))
            for i, p_lat in enumerate(p_lats):
                print ('collecting ', i, p_lat)
                # collect output
                scalars = pending.pop(0).get()
                north_zonal[i, :-1] = scalars['north_mer_fields_scalar']
                north_mer[i, :-1] = scalars['north_zon_fields_scalar']
                south_zonal[i, :-1] = scalars['south_mer_fields_scalar']
                south_mer[i, :-1] = scalars['south_zon_fields_scalar']
                eq_zonal[i, :-1] = scalars['equator_mer_fields_scalar']
                eq_mer[i, :-1] = scalars['equator_zon_fields_scalar']

                north_zonald[i, :-1] = scalars['north_zonal_drifts_scalar']
                north_merd[i, :-1] = scalars['north_mer_drifts_scalar']
                south_zonald[i, :-1] = scalars['south_zonal_drifts_scalar']
                south_merd[i, :-1] = scalars['south_mer_drifts_scalar']
                eq_zonald[i, :-1] = scalars['equator_zonal_drifts_scalar']
                eq_merd[i, :-1] = scalars['equator_mer_drifts_scalar']
        else:
            for i, p_lat in enumerate(p_lats):
                print (i, p_lat)
                scalars = OMMBV.heritage_scalars_for_mapping_ion_drifts([p_lat]*len(p_longs), p_longs,
                                                                       p_alts, [date]*len(p_longs))
                north_zonal[i, :-1] = scalars['north_mer_fields_scalar']
                north_mer[i, :-1] = scalars['north_zon_fields_scalar']
                south_zonal[i, :-1] = scalars['south_mer_fields_scalar']
                south_mer[i, :-1] = scalars['south_zon_fields_scalar']
                eq_zonal[i, :-1] = scalars['equator_mer_fields_scalar']
                eq_mer[i, :-1] = scalars['equator_zon_fields_scalar']

                north_zonald[i, :-1] = scalars['north_zonal_drifts_scalar']
                north_merd[i, :-1] = scalars['north_mer_drifts_scalar']
                south_zonald[i, :-1] = scalars['south_zonal_drifts_scalar']
                south_merd[i, :-1] = scalars['south_mer_drifts_scalar']
                eq_zonald[i, :-1] = scalars['equator_zonal_drifts_scalar']
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
            plt.tight_layout();
            plt.savefig('eq_mer_field_heritage.pdf')
            plt.close()

            fig = plt.figure()
            plt.imshow(eq_mer, origin='lower')  # , vmin=0, vmax=1.)
            plt.colorbar()
            plt.yticks(xtickarr, ['-50', '-25', '0', '25', '50'])
            plt.xticks(ytickarr, ['0', '72', '144', '216', '288', '360'])
            plt.title('Zonal Electric Field Mapping to Magnetic Equator')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.tight_layout();
            plt.savefig('eq_zon_field_heritage.pdf')
            plt.close()

            fig = plt.figure()
            plt.imshow(north_zonal, origin='lower')  # , vmin=0, vmax=2)
            plt.colorbar()
            plt.yticks(xtickarr, ['-50', '-25', '0', '25', '50'])
            plt.xticks(ytickarr, ['0', '72', '144', '216', '288', '360'])
            plt.title('Meridional Electric Field Mapping to Northern Footpoint')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.tight_layout();
            plt.savefig('north_mer_field_heritage.pdf')
            plt.close()

            fig = plt.figure()
            plt.imshow(north_mer, origin='lower')  # , vmin=0, vmax=2)
            plt.colorbar()
            plt.yticks(xtickarr, ['-50', '-25', '0', '25', '50'])
            plt.xticks(ytickarr, ['0', '72', '144', '216', '288', '360'])
            plt.title('Zonal Electric Field Mapping to Northern Footpoint')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.tight_layout();
            plt.savefig('north_zon_field_heritage.pdf')
            plt.close()

            fig = plt.figure()
            plt.imshow(south_zonal, origin='lower')  # , vmin=0, vmax=2)
            plt.colorbar()
            plt.yticks(xtickarr, ['-50', '-25', '0', '25', '50'])
            plt.xticks(ytickarr, ['0', '72', '144', '216', '288', '360'])
            plt.title('Meridional Electric Field Mapping to Southern Footpoint')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.tight_layout();
            plt.savefig('south_mer_field_heritage.pdf')
            plt.close()

            fig = plt.figure()
            plt.imshow(south_mer, origin='lower')  # , vmin=0, vmax=2)
            plt.colorbar()
            plt.yticks(xtickarr, ['-50', '-25', '0', '25', '50'])
            plt.xticks(ytickarr, ['0', '72', '144', '216', '288', '360'])
            plt.title('Zonal Electric Field Mapping to Southern Footpoint')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.tight_layout();
            plt.savefig('south_zon_field_heritage.pdf')
            plt.close()

            fig = plt.figure()
            plt.imshow(np.log10(eq_zonald), origin='lower')  # , vmin=0, vmax=2)
            plt.colorbar()
            plt.yticks(xtickarr, xtickvals)
            plt.xticks(ytickarr, ['0', '72', '144', '216', '288', '360'])
            plt.title('Log Zonal Ion Drift Mapping to Magnetic Equator')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.tight_layout();
            plt.savefig('eq_zonal_drift_heritage.pdf')
            plt.close()

            fig = plt.figure()
            plt.imshow(np.log10(eq_merd), origin='lower')  # , vmin=0, vmax=1.)
            plt.colorbar()
            plt.yticks(xtickarr, xtickvals)
            plt.xticks(ytickarr, ['0', '72', '144', '216', '288', '360'])
            plt.title('Log Meridional Ion Drift Mapping to Magnetic Equator')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.tight_layout();
            plt.savefig('eq_mer_drift_heritage.pdf')
            plt.close()

            fig = plt.figure()
            plt.imshow(north_zonald, origin='lower')  # , vmin=0, vmax=2)
            plt.colorbar()
            plt.yticks(xtickarr, xtickvals)
            plt.xticks(ytickarr, ['0', '72', '144', '216', '288', '360'])
            plt.title('Zonal Ion Drift Mapping to Northern Footpoint')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.tight_layout();
            plt.savefig('north_zonal_drift_heritage.pdf')
            plt.close()

            fig = plt.figure()
            plt.imshow(north_merd, origin='lower')  # , vmin=0, vmax=2)
            plt.colorbar()
            plt.yticks(xtickarr, xtickvals)
            plt.xticks(ytickarr, ['0', '72', '144', '216', '288', '360'])
            plt.title('Meridional Ion Drift Mapping to Northern Footpoint')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.tight_layout();
            plt.savefig('north_mer_drift_heritage.pdf')
            plt.close()

            fig = plt.figure()
            plt.imshow(south_zonald, origin='lower')  # , vmin=0, vmax=2)
            plt.colorbar()
            plt.yticks(xtickarr, xtickvals)
            plt.xticks(ytickarr, ['0', '72', '144', '216', '288', '360'])
            plt.title('Zonal Ion Drift Mapping to Southern Footpoint')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.tight_layout();
            plt.savefig('south_zonal_drift_heritage.pdf')
            plt.close()

            fig = plt.figure()
            plt.imshow(south_merd, origin='lower')  # , vmin=0, vmax=2)
            plt.colorbar()
            plt.yticks(xtickarr, xtickvals)
            plt.xticks(ytickarr, ['0', '72', '144', '216', '288', '360'])
            plt.title('Meridional Ion Drift Mapping to Southern Footpoint')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.tight_layout();
            plt.savefig('south_mer_drift_heritage.pdf')
            plt.close()

        except:
            pass

    def test_unit_vector_and_field_line_plots(self):
        """Test basic vector properties along field lines.

        Produce visualization of field lines around globe
        as well as unit vectors along those field lines

        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        on_travis = os.environ.get('ONTRAVIS') == 'True'

        # convert OMNI position to ECEF
        p_long = np.arange(0., 360., 12.)
        p_alt = 0*p_long + 550.
        p_lats = [5., 10., 15., 20., 25., 30.]

        truthiness = []
        for i, p_lat in enumerate(p_lats):

            trace_s = []
            if not on_travis:
                try:
                    fig = plt.figure()
                    ax = fig.add_subplot(111, projection='3d')
                except:
                    print('Disabling plotting for tests due to error.')
                    on_travis = True


            #
            date = dt.datetime(2000, 1, 1)
            ecef_x, ecef_y, ecef_z = OMMBV.geocentric_to_ecef(p_lat, p_long, p_alt)

            for j, (x, y, z) in enumerate(zip(ecef_x, ecef_y, ecef_z)):
                # perform field line traces
                trace_n = OMMBV.field_line_trace(np.array([x, y, z]), date, 1., 0.,
                                                step_size=.5, max_steps=1.E6)
                trace_s = OMMBV.field_line_trace(np.array([x, y, z]), date, -1., 0.,
                                                step_size=.5, max_steps=1.E6)
                # combine together, S/C position is first for both
                # reverse first array and join so plotting makes sense
                trace = np.vstack((trace_n[::-1], trace_s))
                trace = pds.DataFrame(trace, columns=['x', 'y', 'z'])
                # plot field-line
                if not on_travis:
                    ax.plot(trace['x'], trace['y'], trace['z'], 'b')
                    plt.xlabel('X')
                    plt.ylabel('Y')
                    ax.set_zlabel('Z')
                # clear stored data
                self.inst.data = pds.DataFrame()
                # downselect, reduce number of points
                trace = trace.loc[::1000, :]

                # compute magnetic field vectors
                # need to provide alt, latitude, and longitude in geodetic coords
                latitude, longitude, altitude = OMMBV.ecef_to_geodetic(trace['x'], trace['y'], trace['z'])
                self.inst[:, 'latitude'] = latitude
                self.inst[:, 'longitude'] = longitude
                self.inst[:, 'altitude'] = altitude
                # store values for plotting locations for vectors
                self.inst[:, 'x'] = trace['x'].values
                self.inst[:, 'y'] = trace['y'].values
                self.inst[:, 'z'] = trace['z'].values
                idx, = np.where(self.inst['altitude'] > 250.)
                self.inst.data = self.inst[idx, :]

                # also need to provide transformation from ECEF to S/C
                # going to leave that a null transformation so we can plot in ECF
                self.inst[:, 'sc_xhat_x'], self.inst[:, 'sc_xhat_y'], self.inst[:, 'sc_xhat_z'] = 1., 0., 0.
                self.inst[:, 'sc_yhat_x'], self.inst[:, 'sc_yhat_y'], self.inst[:, 'sc_yhat_z'] = 0., 1., 0.
                self.inst[:, 'sc_zhat_x'], self.inst[:, 'sc_zhat_y'], self.inst[:, 'sc_zhat_z'] = 0., 0., 1.
                self.inst.data.index = pysat.utils.time.create_date_range(dt.datetime(2000, 1, 1),
                                                                          dt.datetime(2000, 1, 1) +
                                                                          pds.DateOffset(
                                                                              seconds=len(self.inst.data) - 1),
                                                                          freq='S')
                OMMBV.satellite.add_mag_drift_unit_vectors(self.inst)

                # if i % 2 == 0:
                length = 500
                vx = self.inst['unit_zon_x']
                vy = self.inst['unit_zon_y']
                vz = self.inst['unit_zon_z']
                if not on_travis:
                    ax.quiver3D(self.inst['x'] + length*vx, self.inst['y'] + length*vy,
                                self.inst['z'] + length*vz, vx, vy, vz, length=500.,
                                color='green')  # , pivot='tail')
                length = 500
                vx = self.inst['unit_fa_x']
                vy = self.inst['unit_fa_y']
                vz = self.inst['unit_fa_z']
                if not on_travis:
                    ax.quiver3D(self.inst['x'] + length*vx, self.inst['y'] + length*vy,
                                self.inst['z'] + length*vz, vx, vy, vz, length=500.,
                                color='purple')  # , pivot='tail')
                length = 500
                vx = self.inst['unit_mer_x']
                vy = self.inst['unit_mer_y']
                vz = self.inst['unit_mer_z']
                if not on_travis:
                    ax.quiver3D(self.inst['x'] + length*vx, self.inst['y'] + length*vy,
                                self.inst['z'] + length*vz, vx, vy, vz, length=500.,
                                color='red')  # , pivot='tail')

                # check that vectors norm to 1
                assert np.all(np.sqrt(self.inst['unit_zon_x'] ** 2 +
                                      self.inst['unit_zon_y'] ** 2 +
                                      self.inst['unit_zon_z'] ** 2) > 0.999999)
                assert np.all(np.sqrt(self.inst['unit_fa_x'] ** 2 +
                                      self.inst['unit_fa_y'] ** 2 +
                                      self.inst['unit_fa_z'] ** 2) > 0.999999)
                assert np.all(np.sqrt(self.inst['unit_mer_x'] ** 2 +
                                      self.inst['unit_mer_y'] ** 2 +
                                      self.inst['unit_mer_z'] ** 2) > 0.999999)
                # confirm vectors are mutually orthogonal
                dot1 = self.inst['unit_zon_x']*self.inst['unit_fa_x'] + self.inst['unit_zon_y']*self.inst[
                    'unit_fa_y'] + self.inst['unit_zon_z']*self.inst['unit_fa_z']
                dot2 = self.inst['unit_zon_x']*self.inst['unit_mer_x'] + self.inst['unit_zon_y']*self.inst[
                    'unit_mer_y'] + self.inst['unit_zon_z']*self.inst['unit_mer_z']
                dot3 = self.inst['unit_fa_x']*self.inst['unit_mer_x'] + self.inst['unit_fa_y']*self.inst[
                    'unit_mer_y'] + self.inst['unit_fa_z']*self.inst['unit_mer_z']
                assert np.all(np.abs(dot1) < 1.E-6)
                assert np.all(np.abs(dot2) < 1.E-6)
                assert np.all(np.abs(dot3) < 1.E-6)

                # ensure that zonal vector is generally eastward
                ones = np.ones(len(self.inst.data.index))
                zeros = np.zeros(len(self.inst.data.index))
                ex, ey, ez = OMMBV.enu_to_ecef_vector(ones, zeros, zeros, self.inst['latitude'], self.inst['longitude'])
                nx, ny, nz = OMMBV.enu_to_ecef_vector(zeros, ones, zeros, self.inst['latitude'], self.inst['longitude'])
                ux, uy, uz = OMMBV.enu_to_ecef_vector(zeros, zeros, ones, self.inst['latitude'], self.inst['longitude'])

                dot1 = self.inst['unit_zon_x']*ex + self.inst['unit_zon_y']*ey + self.inst['unit_zon_z']*ez
                assert np.all(dot1 > 0.)

                dot1 = self.inst['unit_fa_x']*nx + self.inst['unit_fa_y']*ny + self.inst['unit_fa_z']*nz
                assert np.all(dot1 > 0.)

                dot1 = self.inst['unit_mer_x']*ux + self.inst['unit_mer_y']*uy + self.inst['unit_mer_z']*uz
                assert np.all(dot1 > 0.)

            if not on_travis:
                plt.tight_layout();
                plt.savefig(''.join(('magnetic_unit_vectors_', str(int(p_lat)), '.pdf')))
                plt.close()

        assert np.all(truthiness)
