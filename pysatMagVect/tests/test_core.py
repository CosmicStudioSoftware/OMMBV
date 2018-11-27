from nose.tools import assert_raises, raises
import nose.tools
from nose.tools import assert_almost_equals as asseq


import numpy as np
import matplotlib.pyplot as plt
import pandas as pds
import datetime

import functools

import pysatMagVect as pymv
from pysatMagVect import igrf

import pysat

# multiprocessing boolean flag
multiproc = False
if multiproc:
    # get remote instances
    import ipyparallel
    dc = ipyparallel.Client()
    dview = dc[:]
else:
    # nothing to set
    dc = None
    dview = None
    
# results from omniweb calculator
omni_list = [[550. , 20.00   , 0.00 , 29.77,  359.31,  -9.04 ,   3.09],
    [550. , 20.00   , 7.50 , 29.50  ,  7.19 , -8.06 ,   9.54],
    [550. , 20.00   ,15.00 , 28.34  , 15.01 , -7.51 ,  16.20],
    [550. , 20.00   ,22.50 , 27.61  , 22.68 , -7.75 ,  23.10],
    [550. , 20.00   ,30.00 , 27.36  , 30.27 , -8.85 ,  30.34],
    [550. , 20.00   ,37.50 , 27.10  , 37.79 ,-10.13 ,  38.00],
   [550. , 20.00   ,45.00  ,26.89   ,45.24 ,-11.24  , 46.09],
   [ 550.  ,20.00   ,52.50 , 26.77  , 52.65 ,-11.80 ,  54.31],
   [550.  ,20.00   ,60.00  ,26.75   ,60.05 ,-11.77  , 62.31],
   [550.  ,20.00   ,67.50  ,26.80   ,67.49 ,-11.34  , 69.94],
   [550.  ,20.00   ,75.00  ,26.89   ,74.95 ,-10.80  , 77.22],
   [550.  ,20.00   ,82.50  ,26.97   ,82.45 ,-10.37  , 84.27],
   [550.  ,20.00   ,90.00  ,27.01   ,89.94 ,-10.20  , 91.22],
   [550.  ,20.00   ,97.50  ,27.01   ,97.42 ,-10.31  , 98.19],
   [550.  ,20.00  ,105.00  ,26.99  ,104.86 ,-10.64  ,105.27],
   [550.  ,20.00  ,112.50  ,27.02  ,112.26 ,-11.00  ,112.49],
   [550.  ,20.00  ,120.00  ,27.11  ,119.65 ,-11.25  ,119.77],
   [550.  ,20.00  ,127.50  ,27.28  ,127.08 ,-11.35  ,126.99],
   [550.  ,20.00  ,135.00  ,27.50  ,134.59 ,-11.41  ,134.03],
   [550.  ,20.00  ,142.50  ,27.69  ,142.23 ,-11.63  ,140.83],
   [550.  ,20.00  ,150.00  ,27.76  ,149.98 ,-12.20  ,147.44],
   [550.  ,20.00  ,157.50  ,27.66  ,157.81 ,-13.19  ,153.90],
   [550.  ,20.00  ,165.00  ,27.37  ,165.64 ,-14.60  ,160.32],
   [550.  ,20.00  ,172.50  ,26.96  ,173.39 ,-16.33  ,166.77],
   [550.  ,20.00  ,180.00  ,26.50  ,181.04 ,-18.23  ,173.34],
   [550.  ,20.00  ,187.50  ,26.10  ,188.60 ,-20.15  ,180.05],
   [550.  ,20.00  ,195.00  ,25.78  ,196.11 ,-22.00  ,186.89],
   [550.  ,20.00  ,202.50  ,25.53  ,203.62 ,-23.77  ,193.77],
   [550.  ,20.00  ,210.00  ,25.31  ,211.12 ,-25.52  ,200.61],
   [550.  ,20.00  ,217.50  ,25.09  ,218.62 ,-27.37  ,207.40],
   [550.  ,20.00  ,225.00  ,24.87  ,226.09 ,-29.37  ,214.17],
   [550.  ,20.00  ,232.50  ,24.64  ,233.52 ,-31.56  ,220.97],
   [550.  ,20.00  ,240.00  ,24.42  ,240.93 ,-33.92  ,227.85],
   [550.  ,20.00  ,247.50  ,24.19  ,248.29 ,-36.49  ,234.86],
   [550.  ,20.00  ,255.00  ,23.98  ,255.62 ,-39.28  ,242.16],
   [550.  ,20.00  ,262.50  ,23.80  ,262.90 ,-42.28  ,250.04],
   [550.  ,20.00  ,270.00  ,23.66  ,270.13 ,-45.35  ,259.07],
   [550.  ,20.00  ,277.50  ,23.61  ,277.33 ,-48.05 , 270.08],
   [550.  ,20.00  ,285.00  ,23.65  ,284.50 ,-49.36 , 283.83],
   [550.  ,20.00  ,292.50  ,23.81  ,291.67 ,-47.58 , 299.53],
   [550.  ,20.00  ,300.00  ,24.10  ,298.85 ,-41.86 , 313.55],
   [550.  ,20.00  ,307.50  ,24.55  ,306.06 ,-34.06 , 323.42],
   [550.  ,20.00  ,315.00  ,25.14  ,313.34 ,-26.36 , 330.13],
   [550.  ,20.00  ,322.50  ,25.87  ,320.73 ,-19.73 , 335.31],
   [550.  ,20.00  ,330.00  ,26.63  ,328.27 ,-14.56 , 340.08],
   [550.  ,20.00  ,337.50  ,28.33  ,335.63 ,-12.03 , 345.55],
   [550.  ,20.00  ,345.00  ,29.45  ,343.37 ,-10.82 , 351.18],
   [550.  ,20.00  ,352.50  ,30.17  ,351.27 , -9.90 , 356.93]]
omni = pds.DataFrame(omni_list, columns=['p_alt', 'p_lat', 'p_long', 'n_lat', 'n_long', 's_lat', 's_long'])

def gen_data_fixed_alt(alt):
    """Generate data between -90 and 90 degrees latitude, almost."""
    import itertools
    import os
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
    lats = locs[:,1]
    longs = locs[:,0]
    alts = longs*0 + alt_dim
    return lats, longs, alts 

def gen_trace_data_fixed_alt(alt):
    """Generate data between -50 and 50 degrees latitude."""
    import itertools
    import os
    # generate test data set
    on_travis = os.environ.get('ONTRAVIS') == 'True'
    if on_travis:
        # reduced resolution on the test server
        long_dim = np.arange(0., 361., 180.)
        lat_dim = np.arange(-50., 51., 50.)
    else:
        long_dim = np.arange(0., 361., 80.)
        lat_dim = np.arange(-50., 51., 25.)

    alt_dim = alt
    locs = np.array(list(itertools.product(long_dim, lat_dim)))
    # pull out lats and longs
    lats = locs[:,1]
    longs = locs[:,0]
    alts = longs*0 + alt_dim
    return lats, longs, alts 

def gen_plot_grid_fixed_alt(alt):
    """Generate dimensional data between -50 and 50 degrees latitude.
    
    Note
    ----
    Output is different than routines above.
    
    """
    import itertools
    import os
    # generate test data set
    on_travis = os.environ.get('ONTRAVIS') == 'True'
    if on_travis:
        # reduced resolution on the test server
        long_dim = np.arange(0., 360., 180.)
        lat_dim = np.arange(-50., 51., 50.)
    else:
        long_dim = np.arange(0., 360., 1.2*30)
        lat_dim = np.arange(-50., 50.1, 0.25*30)

    alt_dim = np.array([alt])
    return lat_dim, long_dim, alt_dim 


class TestCore():

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
        ecf_x,ecf_y,ecf_z = pymv.geocentric_to_ecef(omni['p_lat'], 
                                                  omni['p_long'],
                                                  omni['p_alt'])
        trace_n = []
        trace_s = []
        date = datetime.datetime(2000, 1, 1)
        for x,y,z in zip(ecf_x,ecf_y,ecf_z):
            # trace north and south, take last points
            trace_n.append(pymv.field_line_trace(np.array([x,y,z]), date, 1., 0., step_size=0.5, max_steps=1.E6)[-1,:])
            trace_s.append(pymv.field_line_trace(np.array([x,y,z]), date, -1., 0., step_size=0.5, max_steps=1.E6)[-1,:])
        trace_n = pds.DataFrame(trace_n, columns = ['x', 'y', 'z'])
        trace_n['lat'], trace_n['long'], trace_n['altitude'] = pymv.ecef_to_geocentric(trace_n['x'], trace_n['y'], trace_n['z'])
        trace_s = pds.DataFrame(trace_s, columns = ['x', 'y', 'z'])
        trace_s['lat'], trace_s['long'], trace_s['altitude'] = pymv.ecef_to_geocentric(trace_s['x'], trace_s['y'], trace_s['z'])

        # ensure longitudes are all 0-360
        idx, = np.where(omni['n_long'] < 0)
        omni.ix[idx, 'n_long'] += 360.
        idx, = np.where(omni['s_long'] < 0)
        omni.ix[idx, 's_long'] += 360.

        idx, = np.where(trace_n['long'] < 0)
        trace_n.ix[idx, 'long'] += 360.
        idx, = np.where(trace_s['long'] < 0)
        trace_s.ix[idx, 'long'] += 360.

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
            plt.xlim((0,360.))
            plt.savefig('magnetic_footpoint_comparison.pdf')
        except:
            pass
        
        # better than 0.5 km accuracy expected for settings above
        assert np.all(np.std(diff_n_lat) < .5)
        assert np.all(np.std(diff_n_lon) < .5)
        assert np.all(np.std(diff_s_lat) < .5)
        assert np.all(np.std(diff_s_lon) < .5)

        
                
    def test_geodetic_to_ecef_to_geodetic(self):
        lats, longs, alts = gen_data_fixed_alt(550.)        
        ecf_x,ecf_y,ecf_z = pymv.geodetic_to_ecef(lats, 
                                                  longs,
                                                  alts)
        lat, elong, alt = pymv.ecef_to_geodetic(ecf_x, ecf_y, ecf_z)
        
        idx, = np.where(elong < 0)
        elong[idx] += 360.

        d_lat = lat - lats
        d_long = elong - longs
        d_alt = alt - alts
        
        assert np.all(np.abs(d_lat) < 1.E-5)
        assert np.all(np.abs(d_long) < 1.E-5)
        assert np.all(np.abs(d_alt) < 1.E-5)

    def test_geodetic_to_ecef_to_geodetic_via_different_methods(self):
            
        lats, longs, alts = gen_data_fixed_alt(550.)        
        ecf_x,ecf_y,ecf_z = pymv.geodetic_to_ecef(lats, 
                                                  longs,
                                                  alts)
        methods = ['closed', 'iterative']
        for method in methods:
            lat, elong, alt = pymv.ecef_to_geodetic(ecf_x, ecf_y, ecf_z,
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
        lats, longs, alts = gen_data_fixed_alt(550.)        
        ecf_x,ecf_y,ecf_z = pymv.geodetic_to_ecef(lats, 
                                                  longs,
                                                  alts)            
        geo_lat, geo_long, geo_alt = pymv.ecef_to_geocentric(ecf_x, ecf_y, ecf_z)

        ecfs_x,ecfs_y,ecfs_z = pymv.geocentric_to_ecef(geo_lat, geo_long, geo_alt)

        lat, elong, alt = pymv.ecef_to_geodetic(ecfs_x, ecfs_y, ecfs_z)
        
        idx, = np.where(elong < 0)
        elong[idx] += 360.

        d_lat = lat - lats
        d_long = elong - longs
        d_alt = alt - alts
        
        assert np.all(np.abs(d_lat) < 1.E-5)
        assert np.all(np.abs(d_long) < 1.E-5)
        assert np.all(np.abs(d_alt) < 1.E-5)
        assert np.all(np.abs(ecf_x-ecfs_x) < 1.E-5)
        assert np.all(np.abs(ecf_y-ecfs_y) < 1.E-5)
        assert np.all(np.abs(ecf_z-ecfs_z) < 1.E-5)


    def test_geocentric_to_ecef_to_geocentric(self):
            
        lats, longs, alts = gen_data_fixed_alt(550.)        
        ecf_x,ecf_y,ecf_z = pymv.geocentric_to_ecef(lats, 
                                                  longs,
                                                  alts)
        lat, elong, alt = pymv.ecef_to_geocentric(ecf_x, ecf_y, ecf_z)

        idx, = np.where(elong < 0)
        elong[idx] += 360.
        
        d_lat = lat - lats
        d_long = elong - longs
        d_alt = alt - alts
        
        assert np.all(np.abs(d_lat) < 1.E-5)
        assert np.all(np.abs(d_long) < 1.E-5)
        assert np.all(np.abs(d_alt) < 1.E-5)



    def test_tracing_accuracy(self):

        lats, longs, alts = gen_trace_data_fixed_alt(550.)        
        ecf_x,ecf_y,ecf_z = pymv.geodetic_to_ecef(lats, 
                                                  longs,
                                                  alts)        
        # step size to be tried
        steps_goal = np.array([1000., 500., 300., 100., 50.,  30., 10., 5., 3., 1., 0.5, 0.3, 0.1])
        # max number of steps (fixed)
        max_steps_goal = steps_goal*0+1E6

        date = datetime.datetime(2000, 1, 1)
        dx = []
        dy = []
        dz = []
        
        # set up multi
        # if self.dc is not None:
        if False:
            import itertools
            targets = itertools.cycle(dc.ids)
            pending = []
            for x, y, z in zip(ecf_x, ecf_y, ecf_z):
                for steps, max_steps in zip(steps_goal, max_steps_goal):
                    # iterate through target cyclicly and run commands
                    dview.targets = targets.next()
                    pending.append(dview.apply_async(pymv.field_line_trace, np.array([x,y,z]), date, 1., 0., 
                                                    step_size=steps, 
                                                    max_steps=max_steps)) 
            # for x, y, z in zip(ecf_x, ecf_y, ecf_z):
                out = []
                for steps, max_steps in zip(steps_goal, max_steps_goal):
                    # collect output 
                    trace_n = pending.pop(0).get()
                    pt = trace_n[-1,:]
                    out.append(pt)
    
                final_pt = pds.DataFrame(out, columns = ['x', 'y', 'z'])
                dx.append(np.abs(final_pt.ix[1:, 'x'].values - final_pt.ix[:,'x'].values[:-1]))
                dy.append(np.abs(final_pt.ix[1:, 'y'].values - final_pt.ix[:,'y'].values[:-1]))
                dz.append(np.abs(final_pt.ix[1:, 'z'].values- final_pt.ix[:,'z'].values[:-1]))
        else:
            for x, y, z in zip(ecf_x, ecf_y, ecf_z):
                out = []
                for steps, max_steps in zip(steps_goal, max_steps_goal):
                    trace_n = pymv.field_line_trace(np.array([x,y,z]), date, 1., 0., 
                                                    step_size=steps, 
                                                    max_steps=max_steps) 
                    pt = trace_n[-1,:]
                    out.append(pt)
    
                final_pt = pds.DataFrame(out, columns = ['x', 'y', 'z'])
                dx.append(np.abs(final_pt.ix[1:, 'x'].values - final_pt.ix[:,'x'].values[:-1]))
                dy.append(np.abs(final_pt.ix[1:, 'y'].values - final_pt.ix[:,'y'].values[:-1]))
                dz.append(np.abs(final_pt.ix[1:, 'z'].values- final_pt.ix[:,'z'].values[:-1]))
            
            
        dx = pds.DataFrame(dx)
        dy = pds.DataFrame(dy)
        dz = pds.DataFrame(dz)
        
        try:
            plt.figure()
            plt.errorbar(np.log10(steps_goal[1:]), dx.mean(axis=0), 
                          yerr=dx.std(axis=0), label='x')
            plt.errorbar(np.log10(steps_goal[1:]), dy.mean(axis=0), 
                        yerr=dy.std(axis=0), label='y')
            plt.errorbar(np.log10(steps_goal[1:]), dz.mean(axis=0), 
                        yerr=dz.std(axis=0), label='z')
            plt.xlabel('Log Step Size (km)')
            plt.ylabel('Change in Foot Point Position (km)')
            plt.title("Change in Final ECEF Position")
            plt.legend()
            plt.tight_layout()
            plt.savefig('Footpoint_position_vs_step_size.pdf' )
            plt.close()
        except:
            pass            
                            
    def test_tracing_accuracy_w_recursion(self):

        lats, longs, alts = gen_trace_data_fixed_alt(550.)        
        ecf_x,ecf_y,ecf_z = pymv.geodetic_to_ecef(lats, 
                                                  longs,
                                                  alts)        
        # step size to be tried
        steps_goal = np.array([5., 5., 5., 5., 5., 5., 5., 5., 5., 5.])
        # max number of steps (fixed)
        max_steps_goal = np.array([100000, 30000, 10000, 3000, 1000, 500, 300, 100, 50, 30])

        date = datetime.datetime(2000, 1, 1)
        dx = []
        dy = []
        dz = []
        for x, y, z in zip(ecf_x, ecf_y, ecf_z):
            out = []
            for steps, max_steps in zip(steps_goal, max_steps_goal):
                trace_n = pymv.field_line_trace(np.array([x,y,z]), date, 1., 0., 
                                                step_size=steps, 
                                                max_steps=max_steps) 
                pt = trace_n[-1,:]
                out.append(pt)

            final_pt = pds.DataFrame(out, columns = ['x', 'y', 'z'])
            dx.append(np.abs(final_pt.ix[1:, 'x'].values - final_pt.ix[:,'x'].values[:-1]))
            dy.append(np.abs(final_pt.ix[1:, 'y'].values - final_pt.ix[:,'y'].values[:-1]))
            dz.append(np.abs(final_pt.ix[1:, 'z'].values- final_pt.ix[:,'z'].values[:-1]))
        dx = pds.DataFrame(dx)
        dy = pds.DataFrame(dy)
        dz = pds.DataFrame(dz)
        
        try:
            plt.figure()
            plt.errorbar(np.log10(max_steps_goal[1:]), np.log10(dx.mean(axis=0)), 
                          yerr=np.log10(dx).std(axis=0), label='x')
            plt.errorbar(np.log10(max_steps_goal[1:]), np.log10(dy.mean(axis=0)), 
                        yerr=np.log10(dy).std(axis=0), label='y')
            plt.errorbar(np.log10(max_steps_goal[1:]), np.log10(dz.mean(axis=0)), 
                        yerr=np.log10(dz).std(axis=0), label='z')
            plt.xlabel('Log Number of Steps per Run')
            plt.ylabel('Change in Foot Point Position (km)')
            plt.title("Change in Final ECEF Position, Recursive Calls")
            plt.legend()
            plt.tight_layout()
            plt.ylabel('Log Change in Foot Point Position (km)')
            plt.savefig('Footpoint_position_vs_max_steps__recursion.pdf' )
            plt.close()
        except:
            pass            

    def test_tracing_accuracy_w_recursion_step_size(self):
        lats, longs, alts = gen_trace_data_fixed_alt(550.)        
        ecf_x,ecf_y,ecf_z = pymv.geodetic_to_ecef(lats, 
                                                  longs,
                                                  alts)        
        # step size to be tried
        steps_goal = np.array([.05, .1, .5, 1., 5., 10., 50., 100., 500.]) 
        # max number of steps (fixed)
        max_steps_goal = np.array([10000., 10000., 10000., 10000., 10000., 10000., 10000., 10000., 10000.])

        date = datetime.datetime(2000, 1, 1)
        dx = []
        dy = []
        dz = []
        for x, y, z in zip(ecf_x, ecf_y, ecf_z):
            out = []
            for steps, max_steps in zip(steps_goal, max_steps_goal):
                trace_n = pymv.field_line_trace(np.array([x,y,z]), date, 1., 0., 
                                                step_size=steps, 
                                                max_steps=max_steps) 
                pt = trace_n[-1,:]
                out.append(pt)

            final_pt = pds.DataFrame(out, columns = ['x', 'y', 'z'])
            dx.append(np.abs(final_pt.ix[1:, 'x'].values - final_pt.ix[:,'x'].values[:-1]))
            dy.append(np.abs(final_pt.ix[1:, 'y'].values - final_pt.ix[:,'y'].values[:-1]))
            dz.append(np.abs(final_pt.ix[1:, 'z'].values- final_pt.ix[:,'z'].values[:-1]))
        dx = pds.DataFrame(dx)
        dy = pds.DataFrame(dy)
        dz = pds.DataFrame(dz)
        
        try:
            plt.figure()
            # plt.errorbar(np.log10(steps_goal[1:]), dx.mean(axis=0), 
            #               yerr=dx.std(axis=0), label='x')
            # plt.errorbar(np.log10(steps_goal[1:]), dy.mean(axis=0), 
            #             yerr=dy.std(axis=0), label='y')
            # plt.errorbar(np.log10(steps_goal[1:]), dz.mean(axis=0), 
            #             yerr=dz.std(axis=0), label='z')
            plt.errorbar(np.log10(steps_goal[1:]), np.log10(dx.mean(axis=0)), 
                          yerr=np.log10(dx).std(axis=0), label='x')
            plt.errorbar(np.log10(steps_goal[1:]), np.log10(dy.mean(axis=0)), 
                        yerr=np.log10(dy).std(axis=0), label='y')
            plt.errorbar(np.log10(steps_goal[1:]), np.log10(dz.mean(axis=0)), 
                        yerr=np.log10(dz).std(axis=0), label='z')
            plt.xlabel('Log Step Size (km)')
            plt.ylabel('Log Change in Foot Point Position (km)')
            plt.title("Change in Final ECEF Position, Recursive Calls")
            plt.legend()
            plt.tight_layout()
            plt.savefig('Footpoint_position_vs_step_size__recursion.pdf' )
            plt.close()
        except:
            pass            
                                                                  
                                                                                                                                                                                                      
    def test_unit_vector_plots(self):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        import os
        on_travis = os.environ.get('ONTRAVIS') == 'True'
        
        # convert OMNI position to ECEF
        p_long = np.arange(0.,360.,12.)
        p_alt = 0*p_long + 550.        
        p_lats = [ 5., 10., 15., 20., 25., 30.]
        
        #ecf_x,ecf_y,ecf_z = pymv.geocentric_to_ecef(p_lat, p_long, p_alt)

        truthiness = []
        for i,p_lat in enumerate(p_lats):

            trace_s = []
            if not on_travis:
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')

            #
            date = datetime.datetime(2000,1,1)
            ecef_x,ecef_y,ecef_z = pymv.geocentric_to_ecef(p_lat, p_long, p_alt)
            
            for j,(x,y,z) in enumerate(zip(ecef_x, ecef_y, ecef_z)):
                # perform field line traces
                trace_n = pymv.field_line_trace(np.array([x,y,z]), date, 1., 0., step_size=.5, max_steps=1.E6)
                trace_s = pymv.field_line_trace(np.array([x,y,z]), date, -1., 0., step_size=.5, max_steps=1.E6)
                # combine together, S/C position is first for both
                # reverse first array and join so plotting makes sense
                trace = np.vstack((trace_n[::-1], trace_s))
                trace = pds.DataFrame(trace, columns=['x','y','z'])
                # plot field-line
                if not on_travis:
                    ax.plot(trace['x'],trace['y'],trace['z'] , 'b')
                    plt.xlabel('X')
                    plt.ylabel('Y')
                    ax.set_zlabel('Z')
                # clear stored data
                self.inst.data = pds.DataFrame()
                # downselect, reduce number of points
                trace = trace.ix[::1000,:]
                
                # compute magnetic field vectors
                # need to provide alt, latitude, and longitude in geodetic coords
                latitude, longitude, altitude = pymv.ecef_to_geodetic(trace['x'], trace['y'], trace['z'])
                self.inst[:,'latitude'] = latitude
                self.inst[:,'longitude'] = longitude
                self.inst[:,'altitude'] = altitude
                # store values for plotting locations for vectors
                self.inst[:,'x'] = trace['x'].values
                self.inst[:,'y'] = trace['y'].values
                self.inst[:,'z'] = trace['z'].values
                self.inst.data = self.inst[self.inst['altitude'] > 250.]
                
                # also need to provide transformation from ECEF to S/C
                # going to leave that a null transformation so we can plot in ECF
                self.inst[:,'sc_xhat_x'], self.inst[:,'sc_xhat_y'], self.inst[:,'sc_xhat_z'] = 1., 0., 0.
                self.inst[:,'sc_yhat_x'], self.inst[:,'sc_yhat_y'], self.inst[:,'sc_yhat_z'] = 0., 1., 0.
                self.inst[:,'sc_zhat_x'], self.inst[:,'sc_zhat_y'], self.inst[:,'sc_zhat_z'] = 0., 0., 1.
                self.inst.data.index = pysat.utils.season_date_range(pysat.datetime(2000,1,1),
                                                                    pysat.datetime(2000,1,1)+pds.DateOffset(seconds=len(self.inst.data)-1),
                                                                    freq='S')
                pymv.satellite.add_mag_drift_unit_vectors(self.inst)
                
                #if i % 2 == 0:
                length = 500
                vx = self.inst['unit_zon_x']
                vy = self.inst['unit_zon_y']
                vz = self.inst['unit_zon_z']
                if not on_travis:
                    ax.quiver3D(self.inst['x'] + length*vx, self.inst['y'] + length*vy, 
                                self.inst['z'] + length*vz, vx, vy,vz, length=500.,
                                color='green') #, pivot='tail')
                length = 500
                vx = self.inst['unit_fa_x']
                vy = self.inst['unit_fa_y']
                vz = self.inst['unit_fa_z']
                if not on_travis:
                    ax.quiver3D(self.inst['x'] + length*vx, self.inst['y'] + length*vy, 
                                self.inst['z'] + length*vz, vx, vy,vz, length=500.,
                                color='purple') #, pivot='tail')
                length = 500
                vx = self.inst['unit_mer_x']
                vy = self.inst['unit_mer_y']
                vz = self.inst['unit_mer_z']
                if not on_travis:
                    ax.quiver3D(self.inst['x'] + length*vx, self.inst['y'] + length*vy, 
                                self.inst['z'] + length*vz, vx, vy,vz, length=500.,
                                color='red') #, pivot='tail')
    
                # check that vectors norm to 1
                assert np.all(np.sqrt(self.inst['unit_zon_x']**2 + self.inst['unit_zon_y']**2 + self.inst['unit_zon_z']**2) > 0.999999)
                assert np.all(np.sqrt(self.inst['unit_fa_x']**2 + self.inst['unit_fa_y']**2 + self.inst['unit_fa_z']**2) > 0.999999)
                assert np.all(np.sqrt(self.inst['unit_mer_x']**2 + self.inst['unit_mer_y']**2 + self.inst['unit_mer_z']**2) > 0.999999)
                # confirm vectors are mutually orthogonal
                dot1 =  self.inst['unit_zon_x']*self.inst['unit_fa_x'] + self.inst['unit_zon_y']*self.inst['unit_fa_y']  + self.inst['unit_zon_z']*self.inst['unit_fa_z']
                dot2 =  self.inst['unit_zon_x']*self.inst['unit_mer_x'] + self.inst['unit_zon_y']*self.inst['unit_mer_y']  + self.inst['unit_zon_z']*self.inst['unit_mer_z']
                dot3 =  self.inst['unit_fa_x']*self.inst['unit_mer_x'] + self.inst['unit_fa_y']*self.inst['unit_mer_y']  + self.inst['unit_fa_z']*self.inst['unit_mer_z']
                assert np.all(np.abs(dot1) < 1.E-6)
                assert np.all(np.abs(dot2) < 1.E-6)
                assert np.all(np.abs(dot3) < 1.E-6)
                
                # ensure that zonal vector is generally eastward
                ones = np.ones(len(self.inst.data.index))
                zeros = np.zeros(len(self.inst.data.index))
                ex, ey, ez = pymv.enu_to_ecef_vector(ones, zeros, zeros, self.inst['latitude'], self.inst['longitude'])
                nx, ny, nz = pymv.enu_to_ecef_vector(zeros, ones, zeros, self.inst['latitude'], self.inst['longitude'])
                ux, uy, uz = pymv.enu_to_ecef_vector(zeros, zeros, ones, self.inst['latitude'], self.inst['longitude'])
                
                dot1 =  self.inst['unit_zon_x']*ex + self.inst['unit_zon_y']*ey  + self.inst['unit_zon_z']*ez
                assert np.all(dot1 > 0.)

                dot1 =  self.inst['unit_fa_x']*nx + self.inst['unit_fa_y']*ny  + self.inst['unit_fa_z']*nz
                assert np.all(dot1 > 0.)
                
                dot1 =  self.inst['unit_mer_x']*ux + self.inst['unit_mer_y']*uy  + self.inst['unit_mer_z']*uz
                assert np.all(dot1 > 0.)

                
            if not on_travis:
                plt.savefig(''.join(('magnetic_unit_vectors_',str(int(p_lat)),'.pdf')))
                plt.close()
        ## plot Earth
        #u = np.linspace(0, 2 * np.pi, 100)
        #v = np.linspace(60.*np.pi/180., 120.*np.pi/180., 100)
        #xx = 6371. * np.outer(np.cos(u), np.sin(v))
        #yy = 6371. * np.outer(np.sin(u), np.sin(v))
        #zz = 6371. * np.outer(np.ones(np.size(u)), np.cos(v))
        #ax.plot_surface(xx, yy, zz, rstride=4, cstride=4, color='darkgray')
        #plt.savefig('magnetic_unit_vectors_w_globe.pdf')
        #print truthiness
        assert np.all(truthiness)

#     def test_geomag_drift_scalars_accuracy_vs_settings(self):
#         import matplotlib.pyplot as plt
#         # from mpl_toolkits.mplot3d import Axes3D
#         import os
#         on_travis = os.environ.get('ONTRAVIS') == 'True'
#         
#         delta = 10.
#         p_lats = np.arange(-50., 50.+delta, delta)
#         p_longs = np.arange(0.,360.,12.)
#         p_alt = 550.
#         
#         step_sizes = [.01, 0.1, 1., 2., 4., 8., 16., 32., 64., 128.]
#         
#         north_zonal = np.zeros((len(p_lats), len(p_longs), len(step_sizes)))
#         north_mer = north_zonal.copy()
#         south_zonal = north_zonal.copy()
#         south_mer = north_zonal.copy()
#         eq_zonal = north_zonal.copy()
#         eq_mer = north_zonal.copy()
#         
#         if not on_travis:
#             fig = plt.figure()
#             ax = fig.add_subplot(111)
#         
#         date = datetime.datetime(2000,1,1)
#         for i,p_lat in enumerate(p_lats):
#             for j, p_long in enumerate(p_longs):
#                 for k, step_size in enumerate(step_sizes):
#                     # print (i,j, date) 
#                     # print (p_lat, p_long, p_alt)
#                     scalars = pymv.scalars_for_mapping_ion_drifts([p_lat], [p_long], [p_alt], [date], step_size=step_size)
#                     north_zonal[i,j,k] = scalars['north_zonal_drifts_scalar'][0]
#                     north_mer[i,j,k] = scalars['north_mer_drifts_scalar'][0]
#                     south_zonal[i,j,k] = scalars['south_zonal_drifts_scalar'][0]
#                     south_mer[i,j,k] = scalars['south_mer_drifts_scalar'][0]
#                     eq_zonal[i,j,k] = scalars['equator_zonal_drifts_scalar'][0]
#                     eq_mer[i,j,k] = scalars['equator_mer_drifts_scalar'][0]
#         
#         n_z_diff = north_zonal[:,:,1:] - north_zonal[:,:,:-1]
#         n_z_diff.mean(axis=0).mean(axis=0) 
#                                   
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
                                                                            p_alts, [date]*len(p_longs))) 
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
                                                                        p_alts, [date]*len(p_longs))
                apex_lat[i,:-1] = olat
                apex_lon[i,:-1] = olon
                apex_alt[i,:-1] = oalt
            
        # calculate difference between apex longitude and original longitude
        # values for apex long are -180 to 180, shift to 0 to 360
        # process degrees a bit to make the degree difference the most meaningful (close to 0)
        idx, idy, = np.where(apex_lon < 0.)
        apex_lon[idx, idy] += 360.
        apex_lon[:, :-1] -= p_longs
        idx, idy, = np.where(apex_lon > 180.)
        apex_lon[idx, idy] -= 360.
            
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
      
    def test_unit_vector_component_plots(self):
        import matplotlib.pyplot as plt
        # from mpl_toolkits.mplot3d import Axes3D
        import os

        p_lats, p_longs, p_alts = gen_plot_grid_fixed_alt(550.)          
        # data returned are the locations along each direction
        # the full range of points obtained by iterating over all
        # recasting alts into a more convenient form for later calculation
        p_alts = [p_alts[0]]*len(p_longs)       
        zvx = np.zeros((len(p_lats), len(p_longs)+1))
        zvy = zvx.copy(); zvz = zvx.copy()
        mx = zvx.copy(); my = zvx.copy(); mz = zvx.copy()
        bx = zvx.copy(); by = zvx.copy(); bz = zvx.copy()
        date = datetime.datetime(2000,1,1)
        # set up multi
        if self.dc is not None:
            import itertools
            targets = itertools.cycle(dc.ids)
            pending = []
            for i,p_lat in enumerate(p_lats):
                # iterate through target cyclicly and run commands
                print (i, p_lat)
                dview.targets = targets.next()
                pending.append(dview.apply_async(pymv.calculate_mag_drift_unit_vectors_ecef,[p_lat]*len(p_longs), p_longs, 
                                                                        p_alts, [date]*len(p_longs), 
                                                                        steps=None, max_steps=10000, step_size=10.,
                                                                        ref_height=120.))
            for i,p_lat in enumerate(p_lats):
                print ('collecting ', i, p_lat)
                    # collect output 
                tzx, tzy, tzz, tbx, tby, tbz, tmx, tmy, tmz = pending.pop(0).get()
                zvx[i,:-1], zvy[i,:-1], zvz[i,:-1] = pymv.ecef_to_enu_vector(tzx, tzy, tzz, [p_lat]*len(p_longs), p_longs)
                bx[i,:-1], by[i,:-1], bz[i,:-1] = pymv.ecef_to_enu_vector(tbx, tby, tbz, [p_lat]*len(p_longs), p_longs)
                mx[i,:-1], my[i,:-1], mz[i,:-1] = pymv.ecef_to_enu_vector(tmx, tmy, tmz, [p_lat]*len(p_longs), p_longs)
        else:
            for i,p_lat in enumerate(p_lats):
                print (i, p_lat)
                tzx, tzy, tzz, tbx, tby, tbz, tmx, tmy, tmz = pymv.calculate_mag_drift_unit_vectors_ecef([p_lat]*len(p_longs), p_longs, 
                                                                                        p_alts, [date]*len(p_longs),
                                                                                        steps=None, max_steps=10000, step_size=10.,
                                                                                        ref_height=120.)
                zvx[i,:-1], zvy[i,:-1], zvz[i,:-1] = pymv.ecef_to_enu_vector(tzx, tzy, tzz, [p_lat]*len(p_longs), p_longs)
                bx[i,:-1], by[i,:-1], bz[i,:-1] = pymv.ecef_to_enu_vector(tbx, tby, tbz, [p_lat]*len(p_longs), p_longs)
                mx[i,:-1], my[i,:-1], mz[i,:-1] = pymv.ecef_to_enu_vector(tmx, tmy, tmz, [p_lat]*len(p_longs), p_longs)
     
        # account for periodicity
        zvx[:,-1] = zvx[:,0]
        zvy[:,-1] = zvy[:,0]
        zvz[:,-1] = zvz[:,0]
        bx[:,-1] = bx[:,0]
        by[:,-1] = by[:,0]
        bz[:,-1] = bz[:,0]
        mx[:,-1] = mx[:,0]
        my[:,-1] = my[:,0]
        mz[:,-1] = mz[:,0]
        
        ytickarr = np.array([0, 0.25, 0.5, 0.75, 1])*(len(p_lats)-1)
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
            plt.savefig('zonal_up.pdf') 
            plt.close()
    
            fig = plt.figure()
            plt.imshow(bx, origin='lower')
            plt.colorbar()
            plt.yticks(ytickarr, ['-50', '-25', '0', '25', '50'])
            plt.xticks(xtickarr, ['0', '72', '144', '216', '288', '360'])       
            plt.title('Field Aligned Unit Vector - Eastward')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.savefig('fa_east.pdf') 
            plt.close()
              
            fig = plt.figure()
            plt.imshow(by, origin='lower')
            plt.colorbar()
            plt.yticks(ytickarr, ['-50', '-25', '0', '25', '50'])
            plt.xticks(xtickarr, ['0', '72', '144', '216', '288', '360'])       
            plt.title('Field Aligned Unit Vector - Northward')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.savefig('fa_north.pdf') 
            plt.close()

            fig = plt.figure()
            plt.imshow(bz, origin='lower')
            plt.colorbar()
            plt.yticks(ytickarr, ['-50', '-25', '0', '25', '50'])
            plt.xticks(xtickarr, ['0', '72', '144', '216', '288', '360'])       
            plt.title('Field Aligned Unit Vector - Upward')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
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
            plt.savefig('mer_up.pdf') 
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
        zvx = np.zeros((len(p_lats), len(p_longs)+1))
        #  meridional vecrtor components
        mx = zvx.copy(); my = zvx.copy(); mz = zvx.copy()
        date = datetime.datetime(2000,1,1)
        # set up multi
        if self.dc is not None:
            import itertools
            targets = itertools.cycle(dc.ids)
            pending = []
            for i,p_lat in enumerate(p_lats):
                # iterate through target cyclicly and run commands
                print (i, p_lat)
                dview.targets = targets.next()
                pending.append(dview.apply_async(pymv.closed_loop_edge_lengths_via_footpoint,[p_lat]*len(p_longs), p_longs, 
                                                                        p_alts, [date]*len(p_longs), 
                                                                        direction,
                                                                        vector_direction, 
                                                                        edge_length=25., 
                                                                        edge_steps=5))
                pending.append(dview.apply_async(pymv.closed_loop_edge_lengths_via_footpoint,[p_lat]*len(p_longs), p_longs, 
                                                                        p_alts, [date]*len(p_longs), 
                                                                        direction,
                                                                        vector_direction, 
                                                                        edge_length=25., 
                                                                        edge_steps=10))
            for i,p_lat in enumerate(p_lats):
                print ('collecting ', i, p_lat)
                # collect output from first run
                mx[i,:-1], my[i,:-1], mz[i,:-1] = pending.pop(0).get()
                # collect output from second run 
                _a, _b, _c = pending.pop(0).get()
                # take difference with first run
                mx[i,:-1] = (mx[i,:-1] - _a)/mx[i,:-1]
                my[i,:-1] = (my[i,:-1] - _b)/my[i,:-1]
                mz[i,:-1] = (mz[i,:-1] - _c)/mz[i,:-1]

        else:
            for i,p_lat in enumerate(p_lats):
                print (i, p_lat)
                mx[i,:-1], my[i,:-1], mz[i,:-1] = pymv.closed_loop_edge_lengths_via_footpoint([p_lat]*len(p_longs), p_longs, 
                                                                        p_alts, [date]*len(p_longs),
                                                                        direction,
                                                                        vector_direction, 
                                                                        edge_length=25., 
                                                                        edge_steps=5)

                # second run
                _a, _b, _c = pymv.closed_loop_edge_lengths_via_footpoint([p_lat]*len(p_longs), p_longs, 
                                                                        p_alts, [date]*len(p_longs), 
                                                                        direction,
                                                                        vector_direction, 
                                                                        edge_length=25., 
                                                                        edge_steps=10)
                # take difference with first run
                mx[i,:-1] = (mx[i,:-1] - _a)/mx[i,:-1]
                my[i,:-1] = (my[i,:-1] - _b)/my[i,:-1]
                mz[i,:-1] = (mz[i,:-1] - _c)/mz[i,:-1]
     
               
        # account for periodicity
        mx[:,-1] = mx[:,0]
        my[:,-1] = my[:,0]
        mz[:,-1] = mz[:,0]
        
        ytickarr = np.array([0, 0.25, 0.5, 0.75, 1])*(len(p_lats)-1)
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
            plt.savefig(direction+'_'+vector_direction+'_closed_loop_footpoint_edge_diff.pdf') 
            plt.close()
              
            fig = plt.figure()
            plt.imshow(np.log10(np.abs(my)), origin='lower')
            plt.colorbar()
            plt.yticks(ytickarr, ['-50', '-25', '0', '25', '50'])
            plt.xticks(xtickarr, ['0', '72', '144', '216', '288', '360'])       
            plt.title('Log Closed Loop Closest Approach Normalized Difference, Pos Footpoint Path')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.savefig(direction+'_'+vector_direction+'_closed_loop_footpoint_pos_diff.pdf') 
            plt.close()

            fig = plt.figure()
            plt.imshow(np.log10(np.abs(mz)), origin='lower')
            plt.colorbar()
            plt.yticks(ytickarr, ['-50', '-25', '0', '25', '50'])
            plt.xticks(xtickarr, ['0', '72', '144', '216', '288', '360'])       
            plt.title('Log Closed Loop Closest Approach Normalized Difference, Minus Footpoint Path')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.savefig(direction+'_'+vector_direction+'_closed_loop_footpoint_min_diff.pdf') 
            plt.close()

            # calculate mean and standard deviation and then plot those
            plt.figure()
            plt.errorbar(p_longs, np.log10(np.median(np.abs(mx[:,:-1]), axis=0)), 
                            yerr=np.std(np.log10(np.abs(mx[:,:-1])), axis=0), label='Edge')
            plt.errorbar(p_longs, np.log10(np.median(np.abs(my[:,:-1]), axis=0)), 
                            yerr=np.std(np.log10(np.abs(my[:,:-1])), axis=0), label='Positive')
            plt.errorbar(p_longs, np.log10(np.median(np.abs(mz[:,:-1]), axis=0)), 
                            yerr=np.std(np.log10(np.abs(mz[:,:-1])), axis=0), label='Minus')
            plt.xlabel('Longitude (Degrees)')
            plt.ylabel('Log Normalized Closed Loop Difference')
            plt.title("Sensitivity of Closed Loop Values")
            plt.legend()
            plt.tight_layout()
            plt.savefig(direction+'_'+vector_direction+'_length_diff_v_longitude.pdf' )
            plt.close()
                        
        except:
            pass

    def test_closed_loop_footpoint_sensitivity_plots(self):
        f = functools.partial(self.closed_loop_footpoint_sensitivity_plots, 'north', 'meridional')
        yield(f,)
        f = functools.partial(self.closed_loop_footpoint_sensitivity_plots, 'south', 'meridional')
        yield(f,)
        f = functools.partial(self.closed_loop_footpoint_sensitivity_plots, 'north', 'zonal')
        yield(f,)
        f = functools.partial(self.closed_loop_footpoint_sensitivity_plots, 'south', 'zonal')
        yield(f,)


    def closed_loop_footpoint_value_plots(self, direction, vector_direction):
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
        zvx = np.zeros((len(p_lats), len(p_longs)+1))
        #  meridional vecrtor components
        mx = zvx.copy(); my = zvx.copy(); mz = zvx.copy()
        date = datetime.datetime(2000,1,1)
        # set up multi
        if self.dc is not None:
            import itertools
            targets = itertools.cycle(dc.ids)
            pending = []
            for i,p_lat in enumerate(p_lats):
                # iterate through target cyclicly and run commands
                print (i, p_lat)
                dview.targets = targets.next()
                pending.append(dview.apply_async(pymv.closed_loop_edge_lengths_via_footpoint,[p_lat]*len(p_longs), p_longs, 
                                                                        p_alts, [date]*len(p_longs), 
                                                                        direction,
                                                                        vector_direction, 
                                                                        edge_length=25., 
                                                                        edge_steps=5))
            for i,p_lat in enumerate(p_lats):
                print ('collecting ', i, p_lat)
                # collect output from first run
                mx[i,:-1], my[i,:-1], mz[i,:-1] = pending.pop(0).get()

        else:
            for i,p_lat in enumerate(p_lats):
                print (i, p_lat)
                mx[i,:-1], my[i,:-1], mz[i,:-1] = pymv.closed_loop_edge_lengths_via_footpoint([p_lat]*len(p_longs), p_longs, 
                                                                        p_alts, [date]*len(p_longs), 
                                                                        direction,
                                                                        vector_direction, 
                                                                        edge_length=25., 
                                                                        edge_steps=5)     
               
        # account for periodicity
        mx[:,-1] = mx[:,0]
        my[:,-1] = my[:,0]
        mz[:,-1] = mz[:,0]
        
        ytickarr = np.array([0, 0.25, 0.5, 0.75, 1])*(len(p_lats)-1)
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
            plt.savefig(direction+'_'+vector_direction+'_closed_loop_footpoint_edge.pdf') 
            plt.close()
              
            fig = plt.figure()
            plt.imshow(np.log10(np.abs(my)), origin='lower')
            plt.colorbar()
            plt.yticks(ytickarr, ['-50', '-25', '0', '25', '50'])
            plt.xticks(xtickarr, ['0', '72', '144', '216', '288', '360'])       
            plt.title('Log Closed Loop Closest Approach, Pos Footpoint Path')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.savefig(direction+'_'+vector_direction+'_closed_loop_footpoint_pos.pdf') 
            plt.close()

            fig = plt.figure()
            plt.imshow(np.log10(np.abs(mz)), origin='lower')
            plt.colorbar()
            plt.yticks(ytickarr, ['-50', '-25', '0', '25', '50'])
            plt.xticks(xtickarr, ['0', '72', '144', '216', '288', '360'])       
            plt.title('Log Closed Loop Closest Approach, Minus Footpoint Path')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.savefig(direction+'_'+vector_direction+'_closed_loop_footpoint_min.pdf') 
            plt.close()

            # calculate mean and standard deviation and then plot those
            plt.figure()
            plt.errorbar(p_longs, np.log10(np.median(np.abs(mx[:,:-1]), axis=0)), 
                            yerr=np.std(np.log10(np.abs(mx[:,:-1])), axis=0), label='Edge')
            plt.errorbar(p_longs, np.log10(np.median(np.abs(my[:,:-1]), axis=0)), 
                            yerr=np.std(np.log10(np.abs(my[:,:-1])), axis=0), label='Positive')
            plt.errorbar(p_longs, np.log10(np.median(np.abs(mz[:,:-1]), axis=0)), 
                            yerr=np.std(np.log10(np.abs(mz[:,:-1])), axis=0), label='Minus')
            plt.xlabel('Longitude (Degrees)')
            plt.ylabel('Log Closed Loop Values')
            plt.title("Closed Loop Values")
            plt.legend()
            plt.tight_layout()
            plt.savefig(direction+'_'+vector_direction+'_length_v_longitude.pdf' )
            plt.close()
                        
        except:
            pass


    def test_closed_loop_footpoint_value_plots(self):
        f = functools.partial(self.closed_loop_footpoint_value_plots, 'north', 'meridional')
        yield(f,)
        f = functools.partial(self.closed_loop_footpoint_value_plots, 'south', 'meridional')
        yield(f,)
        f = functools.partial(self.closed_loop_footpoint_value_plots, 'north', 'zonal')
        yield(f,)
        f = functools.partial(self.closed_loop_footpoint_value_plots, 'south', 'zonal')
        yield(f,)


    def test_unit_vector_component_stepsize_sensitivity_plots(self):
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
        zvx = np.zeros((len(p_lats), len(p_longs)+1))
        zvy = zvx.copy(); zvz = zvx.copy()
        #  meridional vecrtor components
        mx = zvx.copy(); my = zvx.copy(); mz = zvx.copy()
        # field aligned, along B
        bx = zvx.copy(); by = zvx.copy(); bz = zvx.copy()
        date = datetime.datetime(2000,1,1)
        # set up multi
        if self.dc is not None:
            import itertools
            targets = itertools.cycle(dc.ids)
            pending = []
            for i,p_lat in enumerate(p_lats):
                # iterate through target cyclicly and run commands
                print (i, p_lat)
                dview.targets = targets.next()
                pending.append(dview.apply_async(pymv.calculate_mag_drift_unit_vectors_ecef,[p_lat]*len(p_longs), p_longs, 
                                                                        p_alts, [date]*len(p_longs), 
                                                                        steps=None, max_steps=10000, step_size=10.,
                                                                        ref_height=120.))
                pending.append(dview.apply_async(pymv.calculate_mag_drift_unit_vectors_ecef,[p_lat]*len(p_longs), p_longs, 
                                                                        p_alts, [date]*len(p_longs), 
                                                                        steps=None, max_steps=1000, step_size=100.,
                                                                        ref_height=120.))

            for i,p_lat in enumerate(p_lats):
                print ('collecting ', i, p_lat)
                # collect output from first run
                tzx, tzy, tzz, tbx, tby, tbz, tmx, tmy, tmz = pending.pop(0).get()
                zvx[i,:-1], zvy[i,:-1], zvz[i,:-1] = pymv.ecef_to_enu_vector(tzx, tzy, tzz, [p_lat]*len(p_longs), p_longs)
                bx[i,:-1], by[i,:-1], bz[i,:-1] = pymv.ecef_to_enu_vector(tbx, tby, tbz, [p_lat]*len(p_longs), p_longs)
                mx[i,:-1], my[i,:-1], mz[i,:-1] = pymv.ecef_to_enu_vector(tmx, tmy, tmz, [p_lat]*len(p_longs), p_longs)
                # collect output from second run
                tzx, tzy, tzz, tbx, tby, tbz, tmx, tmy, tmz = pending.pop(0).get()
                _a, _b, _c = pymv.ecef_to_enu_vector(tzx, tzy, tzz, [p_lat]*len(p_longs), p_longs)
                # take difference with first run
                zvx[i,:-1] = (zvx[i,:-1] - _a)/zvx[i,:-1]
                zvy[i,:-1] = (zvy[i,:-1] - _b)/zvy[i,:-1]
                zvz[i,:-1] = (zvz[i,:-1] - _c)/zvz[i,:-1]

                _a, _b, _c  = pymv.ecef_to_enu_vector(tbx, tby, tbz, [p_lat]*len(p_longs), p_longs)
                # take difference with first run
                bx[i,:-1] = (bx[i,:-1] - _a)/bx[i,:-1]
                by[i,:-1] = (by[i,:-1] - _b)/by[i,:-1]
                bz[i,:-1] = (bz[i,:-1] - _c)/bz[i,:-1]
                
                _a, _b, _c = pymv.ecef_to_enu_vector(tmx, tmy, tmz, [p_lat]*len(p_longs), p_longs)
                # take difference with first run
                mx[i,:-1] = (mx[i,:-1] - _a)/mx[i,:-1]
                my[i,:-1] = (my[i,:-1] - _b)/my[i,:-1]
                mz[i,:-1] = (mz[i,:-1] - _c)/mz[i,:-1]

        else:
            for i,p_lat in enumerate(p_lats):
                print (i, p_lat)
                tzx, tzy, tzz, tbx, tby, tbz, tmx, tmy, tmz = pymv.calculate_mag_drift_unit_vectors_ecef([p_lat]*len(p_longs), p_longs, 
                                                                                        p_alts, [date]*len(p_longs),
                                                                                        steps=None, max_steps=10000, step_size=10.,
                                                                                        ref_height=120.)
                zvx[i,:-1], zvy[i,:-1], zvz[i,:-1] = pymv.ecef_to_enu_vector(tzx, tzy, tzz, [p_lat]*len(p_longs), p_longs)
                bx[i,:-1], by[i,:-1], bz[i,:-1] = pymv.ecef_to_enu_vector(tbx, tby, tbz, [p_lat]*len(p_longs), p_longs)
                mx[i,:-1], my[i,:-1], mz[i,:-1] = pymv.ecef_to_enu_vector(tmx, tmy, tmz, [p_lat]*len(p_longs), p_longs)

                # second run
                tzx, tzy, tzz, tbx, tby, tbz, tmx, tmy, tmz = pymv.calculate_mag_drift_unit_vectors_ecef([p_lat]*len(p_longs), p_longs, 
                                                                                        p_alts, [date]*len(p_longs),
                                                                                        steps=None, max_steps=1000, step_size=100.,
                                                                                        ref_height=120.)
                _a, _b, _c = pymv.ecef_to_enu_vector(tzx, tzy, tzz, [p_lat]*len(p_longs), p_longs)
                # take difference with first run
                zvx[i,:-1] = (zvx[i,:-1] - _a)/zvx[i,:-1]
                zvy[i,:-1] = (zvy[i,:-1] - _b)/zvy[i,:-1]
                zvz[i,:-1] = (zvz[i,:-1] - _c)/zvz[i,:-1]

                _a, _b, _c  = pymv.ecef_to_enu_vector(tbx, tby, tbz, [p_lat]*len(p_longs), p_longs)
                # take difference with first run
                bx[i,:-1] = (bx[i,:-1] - _a)/bx[i,:-1]
                by[i,:-1] = (by[i,:-1] - _b)/by[i,:-1]
                bz[i,:-1] = (bz[i,:-1] - _c)/bz[i,:-1]
                
                _a, _b, _c = pymv.ecef_to_enu_vector(tmx, tmy, tmz, [p_lat]*len(p_longs), p_longs)
                # take difference with first run
                mx[i,:-1] = (mx[i,:-1] - _a)/mx[i,:-1]
                my[i,:-1] = (my[i,:-1] - _b)/my[i,:-1]
                mz[i,:-1] = (mz[i,:-1] - _c)/mz[i,:-1]
     
               
        # account for periodicity
        zvx[:,-1] = zvx[:,0]
        zvy[:,-1] = zvy[:,0]
        zvz[:,-1] = zvz[:,0]
        bx[:,-1] = bx[:,0]
        by[:,-1] = by[:,0]
        bz[:,-1] = bz[:,0]
        mx[:,-1] = mx[:,0]
        my[:,-1] = my[:,0]
        mz[:,-1] = mz[:,0]
        
        ytickarr = np.array([0, 0.25, 0.5, 0.75, 1])*(len(p_lats)-1)
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
            plt.savefig('zonal_east_norm_diff.pdf') 
            plt.close()
                          
            fig = plt.figure()
            plt.imshow(np.log10(np.abs(zvy)), origin='lower')
            plt.colorbar()
            plt.yticks(ytickarr, ['-50', '-25', '0', '25', '50'])
            plt.xticks(xtickarr, ['0', '72', '144', '216', '288', '360'])       
            plt.title('Log Zonal Unit Vector Nornalized Difference - Northward')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.savefig('zonal_north_norm_diff.pdf') 
            plt.close()

            fig = plt.figure()
            plt.imshow(np.log10(np.abs(zvz)), origin='lower')
            plt.colorbar()
            plt.yticks(ytickarr, ['-50', '-25', '0', '25', '50'])
            plt.xticks(xtickarr, ['0', '72', '144', '216', '288', '360'])       
            plt.title('Log Zonal Unit Vector Nornalized Difference - Upward')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.savefig('zonal_up_norm_diff.pdf') 
            plt.close()

            fig = plt.figure()
            plt.imshow(np.log10(np.abs(bx)), origin='lower')
            plt.colorbar()
            plt.yticks(ytickarr, ['-50', '-25', '0', '25', '50'])
            plt.xticks(xtickarr, ['0', '72', '144', '216', '288', '360'])       
            plt.title('Log Field Aligned Unit Vector Nornalized Difference - Eastward')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.savefig('fa_east_norm_diff.pdf') 
            plt.close()
              
            fig = plt.figure()
            plt.imshow(np.log10(np.abs(by)), origin='lower')
            plt.colorbar()
            plt.yticks(ytickarr, ['-50', '-25', '0', '25', '50'])
            plt.xticks(xtickarr, ['0', '72', '144', '216', '288', '360'])       
            plt.title('Log Field Aligned Unit Vector Nornalized Difference - Northward')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.savefig('fa_north_norm_diff.pdf') 
            plt.close()

            fig = plt.figure()
            plt.imshow(np.log10(np.abs(bz)), origin='lower')
            plt.colorbar()
            plt.yticks(ytickarr, ['-50', '-25', '0', '25', '50'])
            plt.xticks(xtickarr, ['0', '72', '144', '216', '288', '360'])       
            plt.title('Log Field Aligned Unit Vector Nornalized Difference - Upward')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.savefig('fa_up_norm_diff.pdf') 
            plt.close()
    
            fig = plt.figure()
            plt.imshow(np.log10(np.abs(mx)), origin='lower')
            plt.colorbar()
            plt.yticks(ytickarr, ['-50', '-25', '0', '25', '50'])
            plt.xticks(xtickarr, ['0', '72', '144', '216', '288', '360'])       
            plt.title('Log Meridional Unit Vector Nornalized Difference - Eastward')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.savefig('mer_east_norm_diff.pdf') 
            plt.close()
              
            fig = plt.figure()
            plt.imshow(np.log10(np.abs(my)), origin='lower')
            plt.colorbar()
            plt.yticks(ytickarr, ['-50', '-25', '0', '25', '50'])
            plt.xticks(xtickarr, ['0', '72', '144', '216', '288', '360'])       
            plt.title('Log Meridional Unit Vector Nornalized Difference - Northward')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.savefig('mer_north_norm_diff.pdf') 
            plt.close()

            fig = plt.figure()
            plt.imshow(np.log10(np.abs(mz)), origin='lower')
            plt.colorbar()
            plt.yticks(ytickarr, ['-50', '-25', '0', '25', '50'])
            plt.xticks(xtickarr, ['0', '72', '144', '216', '288', '360'])       
            plt.title('Log Meridional Unit Vector Nornalized Difference - Upward')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.savefig('mer_up_norm_diff.pdf') 
            plt.close()

            # calculate mean and standard deviation and then plot those
            # print (p_longs)
            # print (np.mean(np.abs(zvx), axis=0))
            # print (np.std(np.abs(zvx), axis=0))
            plt.figure()
            plt.errorbar(p_longs, np.log10(np.median(np.abs(zvx[:,:-1]), axis=0)), 
                            yerr=np.std(np.log10(np.abs(zvx[:,:-1])), axis=0), label='East')
            plt.errorbar(p_longs, np.log10(np.median(np.abs(zvy[:,:-1]), axis=0)), 
                            yerr=np.std(np.log10(np.abs(zvy[:,:-1])), axis=0), label='North')
            plt.errorbar(p_longs, np.log10(np.median(np.abs(zvz[:,:-1]), axis=0)), 
                            yerr=np.std(np.log10(np.abs(zvz[:,:-1])), axis=0), label='Up')
            plt.xlabel('Longitude (Degrees)')
            plt.ylabel('Log Normalized Change in Zonal Vector')
            plt.title("Sensitivity of Zonal Unit Vector")
            plt.legend()
            plt.tight_layout()
            plt.savefig('zonal_diff_v_longitude.pdf' )
            plt.close()
            
            # calculate mean and standard deviation and then plot those
            plt.figure()
            plt.errorbar(p_longs, np.log10(np.median(np.abs(bx[:,:-1]), axis=0)), 
                            yerr=np.std(np.log10(np.abs(bx[:,:-1])), axis=0), label='East')
            plt.errorbar(p_longs, np.log10(np.median(np.abs(by[:,:-1]), axis=0)), 
                            yerr=np.std(np.log10(np.abs(by[:,:-1])), axis=0), label='North')
            plt.errorbar(p_longs, np.log10(np.median(np.abs(bz[:,:-1]), axis=0)), 
                            yerr=np.std(np.log10(np.abs(bz[:,:-1])), axis=0), label='Up')
            plt.xlabel('Longitude (Degrees)')
            plt.ylabel('Log Normalized Change in Field-Aligned Vector')
            plt.title("Sensitivity of Field-Aligned Unit Vector")
            plt.legend()
            plt.tight_layout()
            plt.savefig('fa_diff_v_longitude.pdf' )
            plt.close()

            # calculate mean and standard deviation and then plot those
            plt.figure()
            plt.errorbar(p_longs, np.log10(np.median(np.abs(mx[:,:-1]), axis=0)), 
                            yerr=np.std(np.log10(np.abs(mx[:,:-1])), axis=0), label='East')
            plt.errorbar(p_longs, np.log10(np.median(np.abs(my[:,:-1]), axis=0)), 
                            yerr=np.std(np.log10(np.abs(my[:,:-1])), axis=0), label='North')
            plt.errorbar(p_longs, np.log10(np.median(np.abs(mz[:,:-1]), axis=0)), 
                            yerr=np.std(np.log10(np.abs(mz[:,:-1])), axis=0), label='Up')
            plt.xlabel('Longitude (Degrees)')
            plt.ylabel('Log Normalized Change in Meridional Vector')
            plt.title("Sensitivity of Meridional Unit Vector")
            plt.legend()
            plt.tight_layout()
            plt.savefig('mer_diff_v_longitude.pdf' )
            plt.close()
            
        except:
            pass
    
    def test_unit_vector_component_refheight_sensitivity_plots(self):
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
        zvx = np.zeros((len(p_lats), len(p_longs)+1))
        zvy = zvx.copy(); zvz = zvx.copy()
        #  meridional vecrtor components
        mx = zvx.copy(); my = zvx.copy(); mz = zvx.copy()
        # field aligned, along B
        bx = zvx.copy(); by = zvx.copy(); bz = zvx.copy()
        date = datetime.datetime(2000,1,1)
        # set up multi
        if self.dc is not None:
            import itertools
            targets = itertools.cycle(dc.ids)
            pending = []
            for i,p_lat in enumerate(p_lats):
                # iterate through target cyclicly and run commands
                print (i, p_lat)
                dview.targets = targets.next()
                pending.append(dview.apply_async(pymv.calculate_mag_drift_unit_vectors_ecef,[p_lat]*len(p_longs), p_longs, 
                                                                        p_alts, [date]*len(p_longs), 
                                                                        steps=None, max_steps=1000, step_size=10.,
                                                                        ref_height=240.))
                pending.append(dview.apply_async(pymv.calculate_mag_drift_unit_vectors_ecef,[p_lat]*len(p_longs), p_longs, 
                                                                        p_alts, [date]*len(p_longs), 
                                                                        steps=None, max_steps=1000, step_size=10.,
                                                                        ref_height=0.))

            for i,p_lat in enumerate(p_lats):
                print ('collecting ', i, p_lat)
                # collect output from first run
                tzx, tzy, tzz, tbx, tby, tbz, tmx, tmy, tmz = pending.pop(0).get()
                zvx[i,:-1], zvy[i,:-1], zvz[i,:-1] = pymv.ecef_to_enu_vector(tzx, tzy, tzz, [p_lat]*len(p_longs), p_longs)
                bx[i,:-1], by[i,:-1], bz[i,:-1] = pymv.ecef_to_enu_vector(tbx, tby, tbz, [p_lat]*len(p_longs), p_longs)
                mx[i,:-1], my[i,:-1], mz[i,:-1] = pymv.ecef_to_enu_vector(tmx, tmy, tmz, [p_lat]*len(p_longs), p_longs)
                # collect output from second run
                tzx, tzy, tzz, tbx, tby, tbz, tmx, tmy, tmz = pending.pop(0).get()
                _a, _b, _c = pymv.ecef_to_enu_vector(tzx, tzy, tzz, [p_lat]*len(p_longs), p_longs)
                # take difference with first run
                zvx[i,:-1] = (zvx[i,:-1] - _a)/zvx[i,:-1]
                zvy[i,:-1] = (zvy[i,:-1] - _b)/zvy[i,:-1]
                zvz[i,:-1] = (zvz[i,:-1] - _c)/zvz[i,:-1]

                _a, _b, _c  = pymv.ecef_to_enu_vector(tbx, tby, tbz, [p_lat]*len(p_longs), p_longs)
                # take difference with first run
                bx[i,:-1] = (bx[i,:-1] - _a)/bx[i,:-1]
                by[i,:-1] = (by[i,:-1] - _b)/by[i,:-1]
                bz[i,:-1] = (bz[i,:-1] - _c)/bz[i,:-1]
                
                _a, _b, _c = pymv.ecef_to_enu_vector(tmx, tmy, tmz, [p_lat]*len(p_longs), p_longs)
                # take difference with first run
                mx[i,:-1] = (mx[i,:-1] - _a)/mx[i,:-1]
                my[i,:-1] = (my[i,:-1] - _b)/my[i,:-1]
                mz[i,:-1] = (mz[i,:-1] - _c)/mz[i,:-1]

        else:
            for i,p_lat in enumerate(p_lats):
                print (i, p_lat)
                tzx, tzy, tzz, tbx, tby, tbz, tmx, tmy, tmz = pymv.calculate_mag_drift_unit_vectors_ecef([p_lat]*len(p_longs), p_longs, 
                                                                                        p_alts, [date]*len(p_longs),
                                                                                        steps=None, max_steps=10000, step_size=10.,
                                                                                        ref_height=240.)
                zvx[i,:-1], zvy[i,:-1], zvz[i,:-1] = pymv.ecef_to_enu_vector(tzx, tzy, tzz, [p_lat]*len(p_longs), p_longs)
                bx[i,:-1], by[i,:-1], bz[i,:-1] = pymv.ecef_to_enu_vector(tbx, tby, tbz, [p_lat]*len(p_longs), p_longs)
                mx[i,:-1], my[i,:-1], mz[i,:-1] = pymv.ecef_to_enu_vector(tmx, tmy, tmz, [p_lat]*len(p_longs), p_longs)

                # second run
                tzx, tzy, tzz, tbx, tby, tbz, tmx, tmy, tmz = pymv.calculate_mag_drift_unit_vectors_ecef([p_lat]*len(p_longs), p_longs, 
                                                                                        p_alts, [date]*len(p_longs),
                                                                                        steps=None, max_steps=10000, step_size=10.,
                                                                                        ref_height=0.)
                _a, _b, _c = pymv.ecef_to_enu_vector(tzx, tzy, tzz, [p_lat]*len(p_longs), p_longs)
                # take difference with first run
                zvx[i,:-1] = (zvx[i,:-1] - _a)/zvx[i,:-1]
                zvy[i,:-1] = (zvy[i,:-1] - _b)/zvy[i,:-1]
                zvz[i,:-1] = (zvz[i,:-1] - _c)/zvz[i,:-1]

                _a, _b, _c  = pymv.ecef_to_enu_vector(tbx, tby, tbz, [p_lat]*len(p_longs), p_longs)
                # take difference with first run
                bx[i,:-1] = (bx[i,:-1] - _a)/bx[i,:-1]
                by[i,:-1] = (by[i,:-1] - _b)/by[i,:-1]
                bz[i,:-1] = (bz[i,:-1] - _c)/bz[i,:-1]
                
                _a, _b, _c = pymv.ecef_to_enu_vector(tmx, tmy, tmz, [p_lat]*len(p_longs), p_longs)
                # take difference with first run
                mx[i,:-1] = (mx[i,:-1] - _a)/mx[i,:-1]
                my[i,:-1] = (my[i,:-1] - _b)/my[i,:-1]
                mz[i,:-1] = (mz[i,:-1] - _c)/mz[i,:-1]
     
               
        # account for periodicity
        zvx[:,-1] = zvx[:,0]
        zvy[:,-1] = zvy[:,0]
        zvz[:,-1] = zvz[:,0]
        bx[:,-1] = bx[:,0]
        by[:,-1] = by[:,0]
        bz[:,-1] = bz[:,0]
        mx[:,-1] = mx[:,0]
        my[:,-1] = my[:,0]
        mz[:,-1] = mz[:,0]
        
        ytickarr = np.array([0, 0.25, 0.5, 0.75, 1])*(len(p_lats)-1)
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
            plt.savefig('zonal_east_diff_height.pdf') 
            plt.close()
                            
            fig = plt.figure()
            plt.imshow(np.log10(np.abs(zvy)), origin='lower')
            plt.colorbar()
            plt.yticks(ytickarr, ['-50', '-25', '0', '25', '50'])
            plt.xticks(xtickarr, ['0', '72', '144', '216', '288', '360'])       
            plt.title('Log Zonal Unit Vector Nornalized Difference - Northward')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.savefig('zonal_north_diff_height.pdf') 
            plt.close()
    
            fig = plt.figure()
            plt.imshow(np.log10(np.abs(zvz)), origin='lower')
            plt.colorbar()
            plt.yticks(ytickarr, ['-50', '-25', '0', '25', '50'])
            plt.xticks(xtickarr, ['0', '72', '144', '216', '288', '360'])       
            plt.title('Log Zonal Unit Vector Nornalized Difference - Upward')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.savefig('zonal_up_diff_height.pdf') 
            plt.close()
    
    
            fig = plt.figure()
            plt.imshow(np.log10(np.abs(bx)), origin='lower')
            plt.colorbar()
            plt.yticks(ytickarr, ['-50', '-25', '0', '25', '50'])
            plt.xticks(xtickarr, ['0', '72', '144', '216', '288', '360'])       
            plt.title('Log Field Aligned Unit Vector Nornalized Difference - Eastward')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.savefig('fa_east_diff_height.pdf') 
            plt.close()
                
            fig = plt.figure()
            plt.imshow(np.log10(np.abs(by)), origin='lower')
            plt.colorbar()
            plt.yticks(ytickarr, ['-50', '-25', '0', '25', '50'])
            plt.xticks(xtickarr, ['0', '72', '144', '216', '288', '360'])       
            plt.title('Log Field Aligned Unit Vector Nornalized Difference - Northward')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.savefig('fa_north_diff_height.pdf') 
            plt.close()
    
            fig = plt.figure()
            plt.imshow(np.log10(np.abs(bz)), origin='lower')
            plt.colorbar()
            plt.yticks(ytickarr, ['-50', '-25', '0', '25', '50'])
            plt.xticks(xtickarr, ['0', '72', '144', '216', '288', '360'])       
            plt.title('Log Field Aligned Unit Vector Nornalized Difference - Upward')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.savefig('fa_up_diff_height.pdf') 
            plt.close()
        
            fig = plt.figure()
            plt.imshow(np.log10(np.abs(mx)), origin='lower')
            plt.colorbar()
            plt.yticks(ytickarr, ['-50', '-25', '0', '25', '50'])
            plt.xticks(xtickarr, ['0', '72', '144', '216', '288', '360'])       
            plt.title('Log Meridional Unit Vector Nornalized Difference - Eastward')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.savefig('mer_east_diff_height.pdf') 
            plt.close()
                
            fig = plt.figure()
            plt.imshow(np.log10(np.abs(my)), origin='lower')
            plt.colorbar()
            plt.yticks(ytickarr, ['-50', '-25', '0', '25', '50'])
            plt.xticks(xtickarr, ['0', '72', '144', '216', '288', '360'])       
            plt.title('Log Meridional Unit Vector Nornalized Difference - Northward')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.savefig('mer_north_diff_height.pdf') 
            plt.close()
    
            fig = plt.figure()
            plt.imshow(np.log10(np.abs(mz)), origin='lower')
            plt.colorbar()
            plt.yticks(ytickarr, ['-50', '-25', '0', '25', '50'])
            plt.xticks(xtickarr, ['0', '72', '144', '216', '288', '360'])       
            plt.title('Log Meridional Unit Vector Nornalized Difference - Upward')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.savefig('mer_up_diff_height.pdf') 
            plt.close()

            # calculate mean and standard deviation and then plot those
            fig = plt.figure()
            plt.errorbar(p_longs, np.log10(np.median(np.abs(zvx[:,:-1]), axis=0)), 
                            yerr=np.std(np.log10(np.abs(zvx[:,:-1])), axis=0), label='East')
            plt.errorbar(p_longs, np.log10(np.median(np.abs(zvy[:,:-1]), axis=0)), 
                            yerr=np.std(np.log10(np.abs(zvy[:,:-1])), axis=0), label='North')
            plt.errorbar(p_longs, np.log10(np.median(np.abs(zvz[:,:-1]), axis=0)), 
                            yerr=np.std(np.log10(np.abs(zvz[:,:-1])), axis=0), label='Up')
            plt.xlabel('Longitude (Degrees)')
            plt.ylabel('Log Normalized Change in Zonal Vector')
            plt.title("Sensitivity of Zonal Unit Vector")
            plt.legend()
            plt.tight_layout()
            plt.savefig('zonal_diff_v_longitude_height.pdf' )
            plt.close()
    
            # calculate mean and standard deviation and then plot those
            fig = plt.figure()
            plt.errorbar(p_longs, np.log10(np.median(np.abs(mx[:,:-1]), axis=0)), 
                            yerr=np.std(np.log10(np.abs(mx[:,:-1])), axis=0), label='East')
            plt.errorbar(p_longs, np.log10(np.median(np.abs(my[:,:-1]), axis=0)), 
                            yerr=np.std(np.log10(np.abs(my[:,:-1])), axis=0), label='North')
            plt.errorbar(p_longs, np.log10(np.median(np.abs(mz[:,:-1]), axis=0)), 
                            yerr=np.std(np.log10(np.abs(mz[:,:-1])), axis=0), label='Up')
            plt.xlabel('Longitude (Degrees)')
            plt.ylabel('Log Normalized Change in Meridional Vector')
            plt.title("Sensitivity of Meridional Unit Vector")
            plt.legend()
            plt.tight_layout()
            plt.savefig('mer_diff_v_longitude_height.pdf' )
            plt.close()

            # calculate mean and standard deviation and then plot those
            fig = plt.figure()
            plt.errorbar(p_longs, np.log10(np.median(np.abs(bx[:,:-1]), axis=0)), 
                            yerr=np.std(np.log10(np.abs(bx[:,:-1])), axis=0), label='East')
            plt.errorbar(p_longs, np.log10(np.median(np.abs(by[:,:-1]), axis=0)), 
                            yerr=np.std(np.log10(np.abs(by[:,:-1])), axis=0), label='North')
            plt.errorbar(p_longs, np.log10(np.median(np.abs(bz[:,:-1]), axis=0)), 
                            yerr=np.std(np.log10(np.abs(bz[:,:-1])), axis=0), label='Up')
            plt.xlabel('Longitude (Degrees)')
            plt.ylabel('Log Normalized Change in Field-Aligned Vector')
            plt.title("Sensitivity of Field-Aligned Unit Vector")
            plt.legend()
            plt.tight_layout()
            plt.savefig('fa_diff_v_longitude_height.pdf' )
            plt.close()
        
        except:
            pass

    
    def step_along_mag_unit_vector_sensitivity_plots(self, direction=None):
        import matplotlib.pyplot as plt
        # from mpl_toolkits.mplot3d import Axes3D
        import os

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
        x = np.zeros((len(p_lats), len(p_longs)+1))
        y = x.copy(); z = x.copy()
        # second set of outputs
        x2 = np.zeros((len(p_lats), len(p_longs)+1))
        y2 = x2.copy(); z2 = x2.copy()

        date = datetime.datetime(2000,1,1)
        dates = [date]*len(p_lats)
        # set up multi
        if self.dc is not None:
            import itertools
            targets = itertools.cycle(dc.ids)
            pending = []
            for i,p_lat in enumerate(p_lats):
                print (i, p_lat)
                for j,p_long in enumerate(p_longs):
                    # iterate through target cyclicly and run commands

                    dview.targets = targets.next()
                    # inputs are ECEF locations
                    in_x, in_y, in_z = pymv.geodetic_to_ecef(p_lat, p_long, p_alts[0])
                    pending.append(dview.apply_async(pymv.step_along_mag_unit_vector, in_x, in_y, in_z, date, 
                                                                            direction=direction,
                                                                            num_steps=10, step_size=25./10.))
                    pending.append(dview.apply_async(pymv.step_along_mag_unit_vector, in_x, in_y, in_z, date, 
                                                                            direction=direction,
                                                                            num_steps=5, step_size=25./5.))
            # for i,p_lat in enumerate(p_lats):
                print ('collecting ', i, p_lat)
                for j,p_long in enumerate(p_longs):
                    # collect output from first run
                    x[i,j], y[i,j], z[i,j] = pending.pop(0).get()          
                    # collect output from second run                    
                    x2[i,j], y2[i,j], z2[i,j] = pending.pop(0).get()
            # trace each location to its apex
            # this provides an increase in the spatial difference that results
            # from innacurate movement between field lines from step_along_mag_unit_vector                    
            for i,p_lat in enumerate(p_lats):
                dview.targets = targets.next()
                # convert all locations to geodetic coordinates
                tlat, tlon, talt = pymv.ecef_to_geodetic(x[i,:-1], y[i,:-1], z[i,:-1])        
                pending.append(dview.apply_async(pymv.apex_location_info, tlat, tlon, talt, dates))
                # convert all locations to geodetic coordinates
                tlat, tlon, talt = pymv.ecef_to_geodetic(x2[i,:-1], y2[i,:-1], z2[i,:-1])        
                pending.append(dview.apply_async(pymv.apex_location_info, tlat, tlon, talt, dates))
            for i,p_lat in enumerate(p_lats):
                x[i,:-1], y[i,:-1], z[i,:-1], _, _, _ = pending.pop(0).get() 
                x2[i,:-1], y2[i,:-1], z2[i,:-1], _, _, _ = pending.pop(0).get() 
            # take difference in locations
            x = x - x2
            y = y - y2
            z = z - z2                    

        else:
            for i,p_lat in enumerate(p_lats):
                print (i, p_lat)
                for j,p_long in enumerate(p_longs):
                    in_x, in_y, in_z = pymv.geodetic_to_ecef(p_lat, p_long, p_alts[0])
                    x[i,j], y[i,j], z[i,j] = pymv.step_along_mag_unit_vector(in_x, in_y, in_z, date, 
                                                                             direction=direction, 
                                                                             num_steps=10, step_size=25./10.)
                    # second run
                    x2[i,j], y2[i,j], z2[i,j] = pymv.step_along_mag_unit_vector(in_x, in_y, in_z, date, 
                                                                            direction=direction, 
                                                                            num_steps=5, step_size=25./5.)
            for i,p_lat in enumerate(p_lats):
                # convert all locations to geodetic coordinates
                tlat, tlon, talt = pymv.ecef_to_geodetic(x[i,:-1], y[i,:-1], z[i,:-1])        
                x[i,:-1], y[i,:-1], z[i,:-1], _, _, _ = pymv.apex_location_info(tlat, tlon, talt, dates)
                # convert all locations to geodetic coordinates
                tlat, tlon, talt = pymv.ecef_to_geodetic(x2[i,:-1], y2[i,:-1], z2[i,:-1])        
                x2[i,:-1], y2[i,:-1], z2[i,:-1], _, _, _ = pymv.apex_location_info(tlat, tlon, talt, dates)
            # take difference in locations
            x = x - x2
            y = y - y2
            z = z - z2                    
               
        # account for periodicity
        x[:,-1] = x[:,0]
        y[:,-1] = y[:,0]
        z[:,-1] = z[:,0]
        # plot tick locations and labels        
        ytickarr = np.array([0, 0.25, 0.5, 0.75, 1])*(len(p_lats)-1)
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
            plt.savefig(direction+'_step_diff_apex_height_x.pdf') 
            plt.close()
            
                          
            fig = plt.figure()
            plt.imshow(np.log10(np.abs(y)), origin='lower')
            plt.colorbar()
            plt.yticks(ytickarr, ['-50', '-25', '0', '25', '50'])
            plt.xticks(xtickarr, ['0', '72', '144', '216', '288', '360'])       
            plt.title('Log Difference in Apex Position (Y - km) After Stepping')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.savefig(direction+'_step_diff_apex_height_y.pdf') 
            plt.close()

            fig = plt.figure()
            plt.imshow(np.log10(np.abs(z)), origin='lower')
            plt.colorbar()
            plt.yticks(ytickarr, ['-50', '-25', '0', '25', '50'])
            plt.xticks(xtickarr, ['0', '72', '144', '216', '288', '360'])       
            plt.title('Log Difference in Apex Position (Z - km) After Stepping')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.savefig(direction+'_step_diff_apex_height_z.pdf') 
            plt.close()
    
            # calculate mean and standard deviation and then plot those
            fig = plt.figure()
            plt.errorbar(p_longs, np.median(np.abs(x[:,:-1]), axis=0), 
                         yerr=np.std(np.abs(x[:,:-1]), axis=0), label='x')
            plt.errorbar(p_longs, np.median(np.abs(y[:,:-1]), axis=0), 
                         yerr=np.std(np.abs(y[:,:-1]), axis=0), label='y')
            plt.errorbar(p_longs, np.median(np.abs(z[:,:-1]), axis=0), 
                         yerr=np.std(np.abs(z[:,:-1]), axis=0), label='z')
            plt.xlabel('Longitude (Degrees)')
            plt.ylabel('Change in ECEF (km)')
            plt.title('Median Difference in Apex Position')
            plt.legend()
            plt.tight_layout()
            plt.savefig(direction+'_step_diff_v_longitude.pdf' )
            plt.close()

        except:
            pass
    
    def test_step_sensitivity(self):
        f = functools.partial(self.step_along_mag_unit_vector_sensitivity_plots, direction='zonal')
        yield (f, )
        f = functools.partial(self.step_along_mag_unit_vector_sensitivity_plots, direction='meridional')
        yield (f, )
                                    
    def test_geomag_efield_scalars_plots(self):
        import matplotlib.pyplot as plt
        import os
        
        p_lats, p_longs, p_alts = gen_plot_grid_fixed_alt(550.)  
        # data returned are the locations along each direction
        # the full range of points obtained by iterating over all
        # recasting alts into a more convenient form for later calculation
        p_alts = [p_alts[0]]*len(p_longs)       

        north_zonal = np.zeros((len(p_lats), len(p_longs)+1))
        north_mer = north_zonal.copy()
        south_zonal = north_zonal.copy()
        south_mer = north_zonal.copy()
        eq_zonal = north_zonal.copy()
        eq_mer = north_zonal.copy()
                
        date = datetime.datetime(2000,1,1)
        # set up multi
        if self.dc is not None:
            import itertools
            targets = itertools.cycle(dc.ids)
            pending = []
            for i,p_lat in enumerate(p_lats):
                # iterate through target cyclicly and run commands
                print (i, p_lat)
                dview.targets = targets.next()
                pending.append(dview.apply_async(pymv.scalars_for_mapping_ion_drifts,[p_lat]*len(p_longs), p_longs, 
                                                                        p_alts, [date]*len(p_longs), 
                                                                        e_field_scaling_only=True)) 
            for i,p_lat in enumerate(p_lats):
                print ('collecting ', i, p_lat)
                    # collect output 
                scalars = pending.pop(0).get()
                north_zonal[i,:-1] = scalars['north_mer_fields_scalar']
                north_mer[i,:-1] = scalars['north_zon_fields_scalar']
                south_zonal[i,:-1] = scalars['south_mer_fields_scalar']
                south_mer[i,:-1] = scalars['south_zon_fields_scalar']
                eq_zonal[i,:-1] = scalars['equator_mer_fields_scalar']
                eq_mer[i,:-1] = scalars['equator_zon_fields_scalar']
        else:
            for i,p_lat in enumerate(p_lats):
                print (i, p_lat)
                scalars = pymv.scalars_for_mapping_ion_drifts([p_lat]*len(p_longs), p_longs, 
                                                               p_alts, [date]*len(p_longs), e_field_scaling_only=True)
                north_zonal[i,:-1] = scalars['north_mer_fields_scalar']
                north_mer[i,:-1] = scalars['north_zon_fields_scalar']
                south_zonal[i,:-1] = scalars['south_mer_fields_scalar']
                south_mer[i,:-1] = scalars['south_zon_fields_scalar']
                eq_zonal[i,:-1] = scalars['equator_mer_fields_scalar']
                eq_mer[i,:-1] = scalars['equator_zon_fields_scalar']
        # account for periodicity
        north_zonal[:,-1] = north_zonal[:,0]
        north_mer[:,-1] = north_mer[:,0]
        south_zonal[:,-1] = south_zonal[:,0]
        south_mer[:,-1] = south_mer[:,0]
        eq_zonal[:,-1] = eq_zonal[:,0]
        eq_mer[:,-1] = eq_mer[:,0]
        
        xtickarr = np.array([0, 0.25, 0.5, 0.75, 1])*(len(p_lats)-1)
        ytickarr = np.array([0, 0.2, 0.4, 0.6, 0.8, 1])*len(p_longs)
        
        try:
            fig = plt.figure()
            plt.imshow(eq_zonal, origin='lower')#, vmin=0, vmax=2)
            plt.colorbar()
            plt.yticks(xtickarr, ['-50', '-25', '0', '25', '50'])
            plt.xticks(ytickarr, ['0', '72', '144', '216', '288', '360'])       
            plt.title('Meridional Electric Field Mapping to Magnetic Equator')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.savefig('eq_mer_field.pdf') 
            plt.close()
              
            fig = plt.figure()
            plt.imshow(eq_mer, origin='lower')#, vmin=0, vmax=1.)
            plt.colorbar()
            plt.yticks(xtickarr, ['-50', '-25', '0', '25', '50'])
            plt.xticks(ytickarr, ['0', '72', '144', '216', '288', '360'])       
            plt.title('Zonal Electric Field Mapping to Magnetic Equator')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.savefig('eq_zon_field.pdf') 
            plt.close()

            fig = plt.figure()
            plt.imshow(north_zonal, origin='lower')#, vmin=0, vmax=2)
            plt.colorbar()
            plt.yticks(xtickarr, ['-50', '-25', '0', '25', '50'])
            plt.xticks(ytickarr, ['0', '72', '144', '216', '288', '360'])       
            plt.title('Meridional Electric Field Mapping to Northern Footpoint')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.savefig('north_mer_field.pdf') 
            plt.close()

            fig = plt.figure()
            plt.imshow(north_mer, origin='lower')#, vmin=0, vmax=2)
            plt.colorbar()
            plt.yticks(xtickarr, ['-50', '-25', '0', '25', '50'])
            plt.xticks(ytickarr, ['0', '72', '144', '216', '288', '360'])       
            plt.title('Zonal Electric Field Mapping to Northern Footpoint')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.savefig('north_zon_field.pdf') 
            plt.close()

            fig = plt.figure()
            plt.imshow(south_zonal, origin='lower')#, vmin=0, vmax=2)
            plt.colorbar()
            plt.yticks(xtickarr, ['-50', '-25', '0', '25', '50'])
            plt.xticks(ytickarr, ['0', '72', '144', '216', '288', '360'])       
            plt.title('Meridional Electric Field Mapping to Southern Footpoint')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.savefig('south_mer_field.pdf') 
            plt.close()

            fig = plt.figure()
            plt.imshow(south_mer, origin='lower')#, vmin=0, vmax=2)
            plt.colorbar()
            plt.yticks(xtickarr, ['-50', '-25', '0', '25', '50'])
            plt.xticks(ytickarr, ['0', '72', '144', '216', '288', '360'])       
            plt.title('Zonal Electric Field Mapping to Southern Footpoint')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.savefig('south_zon_field.pdf') 
            plt.close()
        except:
            pass
                
    def test_geomag_ion_drift_scalars_plots(self):
        import matplotlib.pyplot as plt
        # from mpl_toolkits.mplot3d import Axes3D
        import os

        p_lats, p_longs, p_alts = gen_plot_grid_fixed_alt(550.)          
        # data returned are the locations along each direction
        # the full range of points obtained by iterating over all
        # recasting alts into a more convenient form for later calculation
        p_alts = [p_alts[0]]*len(p_longs)       

        north_zonal = np.zeros((len(p_lats), len(p_longs)+1))
        north_mer = north_zonal.copy()
        south_zonal = north_zonal.copy()
        south_mer = north_zonal.copy()
        eq_zonal = north_zonal.copy()
        eq_mer = north_zonal.copy()
                
        date = datetime.datetime(2000,1,1)
        # set up multi
        if self.dc is not None:
            import itertools
            targets = itertools.cycle(dc.ids)
            pending = []
            for i,p_lat in enumerate(p_lats):
                # iterate through target cyclicly and run commands
                dview.targets = targets.next()
                print ('Targeting ', dview.targets, i, p_lat)
                pending.append(dview.apply_async(pymv.scalars_for_mapping_ion_drifts, [p_lat]*len(p_longs), p_longs, 
                                                                        p_alts, [date]*len(p_longs))) 
            for i,p_lat in enumerate(p_lats):
                print ('collecting ', i, p_lat)
                # collect output 
                scalars = pending.pop(0).get()
                north_zonal[i,:-1] = scalars['north_zonal_drifts_scalar']
                north_mer[i,:-1] = scalars['north_mer_drifts_scalar']
                south_zonal[i,:-1] = scalars['south_zonal_drifts_scalar']
                south_mer[i,:-1] = scalars['south_mer_drifts_scalar']
                eq_zonal[i,:-1] = scalars['equator_zonal_drifts_scalar']
                eq_mer[i,:-1] = scalars['equator_mer_drifts_scalar']
        else:
            for i,p_lat in enumerate(p_lats):
                print (i, p_lat)
                scalars = pymv.scalars_for_mapping_ion_drifts([p_lat]*len(p_longs), p_longs, 
                                                                        p_alts, [date]*len(p_longs))
                north_zonal[i,:-1] = scalars['north_zonal_drifts_scalar']
                north_mer[i,:-1] = scalars['north_mer_drifts_scalar']
                south_zonal[i,:-1] = scalars['south_zonal_drifts_scalar']
                south_mer[i,:-1] = scalars['south_mer_drifts_scalar']
                eq_zonal[i,:-1] = scalars['equator_zonal_drifts_scalar']
                eq_mer[i,:-1] = scalars['equator_mer_drifts_scalar']
        # account for periodicity
        north_zonal[:,-1] = north_zonal[:,0]
        north_mer[:,-1] = north_mer[:,0]
        south_zonal[:,-1] = south_zonal[:,0]
        south_mer[:,-1] = south_mer[:,0]
        eq_zonal[:,-1] = eq_zonal[:,0]
        eq_mer[:,-1] = eq_mer[:,0]
        
        xtickvals = ['-25', '-12.5', '0', '12.5', '25']
        xtickarr = np.array([0, 0.25, 0.5, 0.75, 1])*(len(p_lats)-1)
        ytickarr = np.array([0, 0.2, 0.4, 0.6, 0.8, 1])*len(p_longs)
        
        try:
            fig = plt.figure()
            plt.imshow(np.log10(eq_zonal), origin='lower')#, vmin=0, vmax=2)
            plt.colorbar()
            plt.yticks(xtickarr, xtickvals)
            plt.xticks(ytickarr, ['0', '72', '144', '216', '288', '360'])       
            plt.title('Log Zonal Ion Drift Mapping to Magnetic Equator')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.savefig('eq_zonal_drift.pdf') 
            plt.close()
              
            fig = plt.figure()
            plt.imshow(np.log10(eq_mer), origin='lower')#, vmin=0, vmax=1.)
            plt.colorbar()
            plt.yticks(xtickarr, xtickvals)
            plt.xticks(ytickarr, ['0', '72', '144', '216', '288', '360'])       
            plt.title('Log Meridional Ion Drift Mapping to Magnetic Equator')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.savefig('eq_mer_drift.pdf') 
            plt.close()

            fig = plt.figure()
            plt.imshow(north_zonal, origin='lower')#, vmin=0, vmax=2)
            plt.colorbar()
            plt.yticks(xtickarr, xtickvals)
            plt.xticks(ytickarr, ['0', '72', '144', '216', '288', '360'])       
            plt.title('Zonal Ion Drift Mapping to Northern Footpoint')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.savefig('north_zonal_drift.pdf') 
            plt.close()

            fig = plt.figure()
            plt.imshow(north_mer, origin='lower')#, vmin=0, vmax=2)
            plt.colorbar()
            plt.yticks(xtickarr, xtickvals)
            plt.xticks(ytickarr, ['0', '72', '144', '216', '288', '360'])       
            plt.title('Meridional Ion Drift Mapping to Northern Footpoint')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.savefig('north_mer_drift.pdf') 
            plt.close()

            fig = plt.figure()
            plt.imshow(south_zonal, origin='lower')#, vmin=0, vmax=2)
            plt.colorbar()
            plt.yticks(xtickarr, xtickvals)
            plt.xticks(ytickarr, ['0', '72', '144', '216', '288', '360'])       
            plt.title('Zonal Ion Drift Mapping to Southern Footpoint')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.savefig('south_zonal_drift.pdf') 
            plt.close()

            fig = plt.figure()
            plt.imshow(south_mer, origin='lower')#, vmin=0, vmax=2)
            plt.colorbar()
            plt.yticks(xtickarr, xtickvals)
            plt.xticks(ytickarr, ['0', '72', '144', '216', '288', '360'])       
            plt.title('Meridional Ion Drift Mapping to Southern Footpoint')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.ylabel('Geodetic Latitude (Degrees)')
            plt.savefig('south_mer_drift.pdf') 
            plt.close()
        except:
            pass
        
    def test_basic_ecef_to_enu_rotations(self):
        # test basic transformations first
        # vector pointing along ecef y at 0, 0 is east
        ve, vn, vu = pymv.ecef_to_enu_vector(0., 1., 0., 0., 0.)
        # print ('{:9f}, {:9f}, {:9f}'.format(ve, vn, vu))
        asseq(ve, 1.0, 9)
        asseq(vn, 0, 9)
        asseq(vu, 0, 9)
        # vector pointing along ecef x at 0, 0 is up
        ve, vn, vu = pymv.ecef_to_enu_vector(1., 0., 0., 0., 0.)
        asseq(ve, 0.0, 9)
        asseq(vn, 0.0, 9)
        asseq(vu, 1.0, 9)
        # vector pointing along ecef z at 0, 0 is north
        ve, vn, vu = pymv.ecef_to_enu_vector(0., 0., 1., 0., 0.)
        asseq(ve, 0.0, 9)
        asseq(vn, 1.0, 9)
        asseq(vu, 0.0, 9)

        # vector pointing along ecef x at 0, 90 long is west
        ve, vn, vu = pymv.ecef_to_enu_vector(1., 0., 0., 0., 90.)
        # print ('{:9f}, {:9f}, {:9f}'.format(ve, vn, vu))
        asseq(ve, -1.0, 9)
        asseq(vn, 0.0, 9)
        asseq(vu, 0.0, 9)
        # vector pointing along ecef y at 0, 90 long is up
        ve, vn, vu = pymv.ecef_to_enu_vector(0., 1., 0., 0., 90.)
        asseq(ve, 0.0, 9)
        asseq(vn, 0.0, 9)
        asseq(vu, 1.0, 9)
        # vector pointing along ecef z at 0, 90 long is north
        ve, vn, vu = pymv.ecef_to_enu_vector(0., 0., 1., 0., 90.)
        asseq(ve, 0.0, 9)
        asseq(vn, 1.0, 9)
        asseq(vu, 0.0, 9)

        # vector pointing along ecef y at 0, 180 is west
        ve, vn, vu = pymv.ecef_to_enu_vector(0., 1., 0., 0., 180.)
        asseq(ve, -1.0, 9)
        asseq(vn, 0.0, 9)
        asseq(vu, 0.0, 9)
        # vector pointing along ecef x at 0, 180 is down
        ve, vn, vu = pymv.ecef_to_enu_vector(1., 0., 0., 0., 180.)
        asseq(ve, 0.0, 9)
        asseq(vn, 0.0, 9)
        asseq(vu, -1.0, 9)
        # vector pointing along ecef z at 0, 0 is north
        ve, vn, vu = pymv.ecef_to_enu_vector(0., 0., 1., 0., 180.)
        asseq(ve, 0.0, 9)
        asseq(vn, 1.0, 9)
        asseq(vu, 0.0, 9)

        ve, vn, vu = pymv.ecef_to_enu_vector(0., 1., 0., 45., 0.)
        # print ('{:9f}, {:9f}, {:9f}'.format(ve, vn, vu))
        asseq(ve, 1.0, 9)
        asseq(vn, 0, 9)
        asseq(vu, 0, 9)
        # vector pointing along ecef x at 0, 0 is south/up
        ve, vn, vu = pymv.ecef_to_enu_vector(1., 0., 0., 45., 0.)
        asseq(ve, 0.0, 9)
        asseq(vn, -np.cos(np.pi/4), 9)
        asseq(vu, np.cos(np.pi/4), 9)
        # vector pointing along ecef z at 45, 0 is north/up
        ve, vn, vu = pymv.ecef_to_enu_vector(0., 0., 1., 45., 0.)
        asseq(ve, 0.0, 9)
        asseq(vn, np.cos(np.pi/4), 9)
        asseq(vu, np.cos(np.pi/4), 9)
        

    def test_basic_enu_to_ecef_rotations(self):
        # test basic transformations first
        # vector pointing east at 0, 0 is along y
        vx, vy, vz = pymv.enu_to_ecef_vector(1., 0., 0., 0., 0.)
        # print ('{:9f}, {:9f}, {:9f}'.format(vx, vy, vz))
        asseq(vx, 0.0, 9)
        asseq(vy, 1.0, 9)
        asseq(vz, 0.0, 9)
        # vector pointing up at 0, 0 is along x
        vx, vy, vz = pymv.enu_to_ecef_vector(0., 0., 1., 0., 0.)
        asseq(vx, 1.0, 9)
        asseq(vy, 0.0, 9)
        asseq(vz, 0.0, 9)
        # vector pointing north at 0, 0 is along z
        vx, vy, vz = pymv.enu_to_ecef_vector(0., 1., 0., 0., 0.)
        asseq(vx, 0.0, 9)
        asseq(vy, 0.0, 9)
        asseq(vz, 1.0, 9)

        # east vector at 0, 90 long points along -x
        vx, vy, vz = pymv.enu_to_ecef_vector(1., 0., 0., 0., 90.)
        # print ('{:9f}, {:9f}, {:9f}'.format(vx, vy, vz))
        asseq(vx, -1.0, 9)
        asseq(vy, 0.0, 9)
        asseq(vz, 0.0, 9)
        # vector pointing up at 0, 90 is along y
        vx, vy, vz = pymv.enu_to_ecef_vector(0., 0., 1., 0., 90.)
        asseq(vx, 0.0, 9)
        asseq(vy, 1.0, 9)
        asseq(vz, 0.0, 9)
        # vector pointing north at 0, 90 is along z
        vx, vy, vz = pymv.enu_to_ecef_vector(0., 1., 0., 0., 90.)
        asseq(vx, 0.0, 9)
        asseq(vy, 0.0, 9)
        asseq(vz, 1.0, 9)

        # vector pointing east at 0, 0 is along y
        vx, vy, vz = pymv.enu_to_ecef_vector(1., 0., 0., 0., 180.)
        # print ('{:9f}, {:9f}, {:9f}'.format(vx, vy, vz))
        asseq(vx, 0.0, 9)
        asseq(vy, -1.0, 9)
        asseq(vz, 0.0, 9)
        # vector pointing up at 0, 180 is along -x
        vx, vy, vz = pymv.enu_to_ecef_vector(0., 0., 1., 0., 180.)
        asseq(vx, -1.0, 9)
        asseq(vy, 0.0, 9)
        asseq(vz, 0.0, 9)
        # vector pointing north at 0, 180 is along z
        vx, vy, vz = pymv.enu_to_ecef_vector(0., 1., 0., 0., 180.)
        asseq(vx, 0.0, 9)
        asseq(vy, 0.0, 9)
        asseq(vz, 1.0, 9)

    def test_ecef_to_enu_back_to_ecef(self):
        
        vx = 0.9
        vy = 0.1
        vz = np.sqrt(1. - vx**2+vy**2)
        lats, longs, alts = gen_data_fixed_alt(550.)        
        for lat, lon, alt in zip(lats, longs, alts):
            vxx, vyy, vzz = pymv.ecef_to_enu_vector(vx, vy, vz, lat, lon)
            vxx, vyy, vzz = pymv.enu_to_ecef_vector(vxx, vyy, vzz, lat, lon)
            asseq(vx, vxx, 9)
            asseq(vy, vyy, 9)
            asseq(vz, vzz, 9)
        
    def test_enu_to_ecef_back_to_enu(self):
        
        vx = 0.9
        vy = 0.1
        vz = np.sqrt(1. - vx**2+vy**2)
        lats, longs, alts = gen_data_fixed_alt(550.)        
        for lat, lon, alt in zip(lats, longs, alts):
            vxx, vyy, vzz = pymv.enu_to_ecef_vector(vx, vy, vz, lat, lon)
            vxx, vyy, vzz = pymv.ecef_to_enu_vector(vxx, vyy, vzz, lat, lon)
            asseq(vx, vxx, 9)
            asseq(vy, vyy, 9)
            asseq(vz, vzz, 9)
            
    def test_igrf_end_to_ecef_back_to_end(self):
        #import pdb    
        vx = 0.9
        vy = 0.1
        vz = np.sqrt(1. - vx**2+vy**2)
        vz = -vz
        lats, longs, alts = gen_data_fixed_alt(550.)        
        for lat, lon, alt in zip(lats, longs, alts):
            #print(vx, vy, vz, lat, lon)
            #pdb.set_trace()
            # input here is co-latitude, not latitude
            # inputs to fortran are in radians
            vxx, vyy, vzz = igrf.end_vector_to_ecef(vx, vy, vz, np.deg2rad(90.-lat), np.deg2rad(lon))
            vx2, vy2, vz2 = pymv.enu_to_ecef_vector(vx, vy, -vz, lat, lon)
            # print ('end check ', vxx, vyy, vzz, vx2, vy2, vz2)
            asseq(vxx, vx2, 9)
            asseq(vyy, vy2, 9)
            asseq(vzz, vz2, 9)
            
            vxx, vyy, vzz = pymv.ecef_to_enu_vector(vxx, vyy, vzz, lat, lon)
            # convert upward component back to down
            vzz = -vzz
            # compare original inputs to outputs
            asseq(vx, vxx, 9)
            asseq(vy, vyy, 9)
            asseq(vz, vzz, 9)


    def test_igrf_ecef_to_geodetic_back_to_ecef(self):
        lats, longs, alts = gen_data_fixed_alt(550.)        
        ecf_x,ecf_y,ecf_z = pymv.geodetic_to_ecef(lats, 
                                                  longs,
                                                  alts)        
        for ecef_x, ecef_y, ecef_z, geo_lat, geo_lon, geo_alt in zip(ecf_x,ecf_y,
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
        lats, longs, alts = gen_data_fixed_alt(550.)        
        ecf_x,ecf_y,ecf_z = pymv.geodetic_to_ecef(lats, 
                                                  longs,
                                                  alts)        
        for ecef_x, ecef_y, ecef_z, geo_lat, geo_lon, geo_alt in zip(ecf_x,ecf_y,
                           ecf_z, lats, longs, alts):
            pos = np.array([ecef_x, ecef_y, ecef_z])
            
            colat, lon, r = igrf.ecef_to_colat_long_r(pos)
            # results are returned in radians
            lat = 90. - np.rad2deg(colat) 
            lon = np.rad2deg(lon)
            
            lat2, lon2, h2 = pymv.ecef_to_geocentric(*pos, ref_height=0)
        
            # print(lat, lon, r, lat2, lon2, h2)
            asseq(r, h2, 9)
            asseq(lat, lat2, 9)
            asseq(lon, lon2, 9)
