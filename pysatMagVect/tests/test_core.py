from nose.tools import assert_raises, raises
import nose.tools
from nose.tools import assert_almost_equals as asseq


import numpy as np
import matplotlib.pyplot as plt
import pandas as pds
import datetime

import pysatMagVect as pymv

import pysat

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


class TestCore():

    def __init__(self):
        # placeholder for data management features
        self.inst = pysat.Instrument('pysat', 'testing')
        self.inst.yr = 2010.
        self.inst.doy = 1.

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
            plt.savefig('magnetic_footpoint_comparison.png')
        except:
            pass
        
        # better than 0.5 km accuracy expected for settings above
        assert np.all(np.std(diff_n_lat) < .5)
        assert np.all(np.std(diff_n_lon) < .5)
        assert np.all(np.std(diff_s_lat) < .5)
        assert np.all(np.std(diff_s_lon) < .5)

        
                
    def test_geodetic_to_ecef_to_geodetic(self):
            
        ecf_x,ecf_y,ecf_z = pymv.geodetic_to_ecef(omni['p_lat'], 
                                                  omni['p_long'],
                                                  omni['p_alt'])
        lat, elong, alt = pymv.ecef_to_geodetic(ecf_x, ecf_y, ecf_z)
        
        idx, = np.where(elong < 0)
        elong[idx] += 360.

        d_lat = lat - omni['p_lat']
        d_long = elong - omni['p_long']
        d_alt = alt - omni['p_alt']
        
        assert np.all(np.abs(d_lat) < 1.E-5)
        assert np.all(np.abs(d_long) < 1.E-5)
        assert np.all(np.abs(d_alt) < 1.E-5)

    def test_geodetic_to_ecef_to_geodetic_via_different_methods(self):
            
        ecf_x,ecf_y,ecf_z = pymv.geodetic_to_ecef(omni['p_lat'], 
                                                  omni['p_long'],
                                                  omni['p_alt'])
        methods = ['closed', 'iterative']
        flags = []
        for method in methods:
            lat, elong, alt = pymv.ecef_to_geodetic(ecf_x, ecf_y, ecf_z,
                                                  method=method)
            
            idx, = np.where(elong < 0)
            elong[idx] += 360.
    
            d_lat = lat - omni['p_lat']
            d_long = elong - omni['p_long']
            d_alt = alt - omni['p_alt']
            
            flag1 = np.all(np.abs(d_lat) < 1.E-5)
            flag2 = np.all(np.abs(d_long) < 1.E-5)
            flag3 = np.all(np.abs(d_alt) < 1.E-5)
            flags.extend([flag1, flag2, flag3])
            
        assert np.all(flags)


    def test_geodetic_to_ecef_to_geocentric_to_ecef_to_geodetic(self):
            
        ecf_x,ecf_y,ecf_z = pymv.geodetic_to_ecef(omni['p_lat'], 
                                                  omni['p_long'],
                                                  omni['p_alt'])
        geo_lat, geo_long, geo_alt = pymv.ecef_to_geocentric(ecf_x, ecf_y, ecf_z)

        ecf_x,ecf_y,ecf_z = pymv.geocentric_to_ecef(geo_lat, geo_long, geo_alt)

        lat, elong, alt = pymv.ecef_to_geodetic(ecf_x, ecf_y, ecf_z)
        
        idx, = np.where(elong < 0)
        elong[idx] += 360.

        d_lat = lat - omni['p_lat']
        d_long = elong - omni['p_long']
        d_alt = alt - omni['p_alt']
        
        flag1 = np.all(np.abs(d_lat) < 1.E-5)
        flag2 = np.all(np.abs(d_long) < 1.E-5)
        flag3 = np.all(np.abs(d_alt) < 1.E-5)
        assert flag1 & flag2 & flag3


    def test_geocentric_to_ecef_to_geocentric(self):
            
        ecf_x,ecf_y,ecf_z = pymv.geocentric_to_ecef(omni['p_lat'], 
                                                  omni['p_long'],
                                                  omni['p_alt'])
        lat, elong, alt = pymv.ecef_to_geocentric(ecf_x, ecf_y, ecf_z)

        idx, = np.where(elong < 0)
        elong[idx] += 360.
        
        d_lat = lat - omni['p_lat']
        d_long = elong - omni['p_long']
        d_alt = alt - omni['p_alt']
        
        flag1 = np.all(np.abs(d_lat) < 1.E-5)
        flag2 = np.all(np.abs(d_long) < 1.E-5)
        flag3 = np.all(np.abs(d_alt) < 1.E-5)
        assert flag1 & flag2 & flag3


    def test_tracing_accuracy(self):
        x,y,z = pymv.geocentric_to_ecef(np.array([20.]), np.array([0.]), np.array([550.]))
        
        steps_goal = np.array([1., 3.3E-1, 1.E-1, 3.3E-2,  1.E-2, 3.3E-3, 1.E-3, 3.3E-4, 
                               1.E-4])*6371. #, 3.3E-5, 1.E-5, 3.3E-6, 1.E-6, 3.3E-7])
        max_steps_goal = 1.E2*6371./steps_goal

        out = []
        date = datetime.datetime(2000, 1, 1)
        for steps, max_steps in zip(steps_goal, max_steps_goal):
            #print ' '
            #print steps, max_steps
            trace_n = pymv.field_line_trace(np.array([x[0],y[0],z[0]]), date, 1., 0., 
                                               step_size=steps, 
                                               max_steps=max_steps)  

            #print trace_n
            pt = trace_n[1,:]
            out.append(pt)
            #out.append(pymv.ecef_to_geocentric(pt[0], pt[1], pt[2]))

        final_pt = pds.DataFrame(out, columns = ['x', 'y', 'z'])
        x = np.log10(np.abs(final_pt.ix[1:, 'x'].values - final_pt.ix[:,'x'].values[:-1]))
        y = np.log10(np.abs(final_pt.ix[1:, 'y'].values - final_pt.ix[:,'y'].values[:-1]))
        z = np.log10(np.abs(final_pt.ix[1:, 'z'].values- final_pt.ix[:,'z'].values[:-1]))
        
        try:
            plt.figure()
            plt.plot(np.log10(steps_goal[1:]), x)
            plt.plot(np.log10(steps_goal[1:]), y)
            plt.plot(np.log10(steps_goal[1:]), z)
            plt.xlabel('Log Step Size (km)')
            plt.ylabel('Log Change in Foot Point Position (km)')
            plt.savefig('Footpoint_position_vs_step_size.png' )
        except:
            pass
                           
                                                                  
    def test_unit_vector_plots(self):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        import os
        on_travis = os.environ.get('ONTRAVIS') == 'True'
        
        # convert OMNI position to ECEF
        p_long = np.arange(0.,360.,12.)
        p_lat = 0.*p_long
        #p_lat = np.hstack((p_lat+5., p_lat+20.))
        #p_long = np.hstack((p_long, p_long))
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
                trace_n = pymv.field_line_trace(np.array([x,y,z]), date, 1., 0., step_size=0.5, max_steps=1.E6)
                trace_s = pymv.field_line_trace(np.array([x,y,z]), date, -1., 0., step_size=0.5, max_steps=1.E6)
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
                pymv.add_mag_drift_unit_vectors(self.inst)
                
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
                mag1 = np.all(np.sqrt(self.inst['unit_zon_x']**2 + self.inst['unit_zon_y']**2 + self.inst['unit_zon_z']**2) > 0.999999)
                mag2 = np.all(np.sqrt(self.inst['unit_fa_x']**2 + self.inst['unit_fa_y']**2 + self.inst['unit_fa_z']**2) > 0.999999)
                mag3 = np.all(np.sqrt(self.inst['unit_mer_x']**2 + self.inst['unit_mer_y']**2 + self.inst['unit_mer_z']**2) > 0.999999)
                # confirm vectors are mutually orthogonal
                dot1 =  self.inst['unit_zon_x']*self.inst['unit_fa_x'] + self.inst['unit_zon_y']*self.inst['unit_fa_y']  + self.inst['unit_zon_z']*self.inst['unit_fa_z']
                dot2 =  self.inst['unit_zon_x']*self.inst['unit_mer_x'] + self.inst['unit_zon_y']*self.inst['unit_mer_y']  + self.inst['unit_zon_z']*self.inst['unit_mer_z']
                dot3 =  self.inst['unit_fa_x']*self.inst['unit_mer_x'] + self.inst['unit_fa_y']*self.inst['unit_mer_y']  + self.inst['unit_fa_z']*self.inst['unit_mer_z']
                flag1 = np.all(np.abs(dot1) < 1.E-3)
                flag2 = np.all(np.abs(dot2) < 1.E-3)
                flag3 = np.all(np.abs(dot3) < 1.E-3)
                #print mag1, mag2, mag3, flag1, flag2, flag3
                #print np.sqrt(self.inst['unit_mer_x']**2 + self.inst['unit_mer_y']**2 + self.inst['unit_mer_z']**2)#, dot2, dot3
                truthiness.append(flag1 & flag2 & flag3 & mag1 & mag2 & mag3)
        
            if not on_travis:
                plt.savefig(''.join(('magnetic_unit_vectors_',str(int(p_lat)),'.png')))
        ## plot Earth
        #u = np.linspace(0, 2 * np.pi, 100)
        #v = np.linspace(60.*np.pi/180., 120.*np.pi/180., 100)
        #xx = 6371. * np.outer(np.cos(u), np.sin(v))
        #yy = 6371. * np.outer(np.sin(u), np.sin(v))
        #zz = 6371. * np.outer(np.ones(np.size(u)), np.cos(v))
        #ax.plot_surface(xx, yy, zz, rstride=4, cstride=4, color='darkgray')
        #plt.savefig('magnetic_unit_vectors_w_globe.png')
        #print truthiness
        assert np.all(truthiness)
        
        
    def test_basic_ecef_to_enu_rotations(self):
        # test basic transformations first
        # vector pointing along ecef y at 0, 0 is east
        ve, vn, vu = pymv.ecef_to_enu_vector(0., 1., 0., 0., 0.)
        # print ('{:9f}, {:9f}, {:9f}'.format(ve, vn, vu))
        asseq(ve, 1.0, 6)
        asseq(vn, 0, 6)
        asseq(vu, 0, 6)
        # vector pointing along ecef x at 0, 0 is up
        ve, vn, vu = pymv.ecef_to_enu_vector(1., 0., 0., 0., 0.)
        asseq(ve, 0.0, 6)
        asseq(vn, 0.0, 6)
        asseq(vu, 1.0, 6)
        # vector pointing along ecef z at 0, 0 is north
        ve, vn, vu = pymv.ecef_to_enu_vector(0., 0., 1., 0., 0.)
        asseq(ve, 0.0, 6)
        asseq(vn, 1.0, 6)
        asseq(vu, 0.0, 6)

        # vector pointing along ecef x at 0, 90 long is west
        ve, vn, vu = pymv.ecef_to_enu_vector(1., 0., 0., 0., 90.)
        # print ('{:9f}, {:9f}, {:9f}'.format(ve, vn, vu))
        asseq(ve, -1.0, 6)
        asseq(vn, 0.0, 6)
        asseq(vu, 0.0, 6)
        # vector pointing along ecef y at 0, 90 long is up
        ve, vn, vu = pymv.ecef_to_enu_vector(0., 1., 0., 0., 90.)
        asseq(ve, 0.0, 6)
        asseq(vn, 0.0, 6)
        asseq(vu, 1.0, 6)
        # vector pointing along ecef z at 0, 90 long is north
        ve, vn, vu = pymv.ecef_to_enu_vector(0., 0., 1., 0., 90.)
        asseq(ve, 0.0, 6)
        asseq(vn, 1.0, 6)
        asseq(vu, 0.0, 6)

        # vector pointing along ecef y at 0, 180 is west
        ve, vn, vu = pymv.ecef_to_enu_vector(0., 1., 0., 0., 180.)
        asseq(ve, -1.0, 6)
        asseq(vn, 0.0, 6)
        asseq(vu, 0.0, 6)
        # vector pointing along ecef x at 0, 180 is down
        ve, vn, vu = pymv.ecef_to_enu_vector(1., 0., 0., 0., 180.)
        asseq(ve, 0.0, 6)
        asseq(vn, 0.0, 6)
        asseq(vu, -1.0, 6)
        # vector pointing along ecef z at 0, 0 is north
        ve, vn, vu = pymv.ecef_to_enu_vector(0., 0., 1., 0., 180.)
        asseq(ve, 0.0, 6)
        asseq(vn, 1.0, 6)
        asseq(vu, 0.0, 6)

        ve, vn, vu = pymv.ecef_to_enu_vector(0., 1., 0., 45., 0.)
        # print ('{:9f}, {:9f}, {:9f}'.format(ve, vn, vu))
        asseq(ve, 1.0, 6)
        asseq(vn, 0, 6)
        asseq(vu, 0, 6)
        # vector pointing along ecef x at 0, 0 is south/up
        ve, vn, vu = pymv.ecef_to_enu_vector(1., 0., 0., 45., 0.)
        asseq(ve, 0.0, 6)
        asseq(vn, -np.cos(np.pi/4), 6)
        asseq(vu, np.cos(np.pi/4), 6)
        # vector pointing along ecef z at 45, 0 is north/up
        ve, vn, vu = pymv.ecef_to_enu_vector(0., 0., 1., 45., 0.)
        asseq(ve, 0.0, 6)
        asseq(vn, np.cos(np.pi/4), 6)
        asseq(vu, np.cos(np.pi/4), 6)


    def test_basic_enu_to_ecef_rotations(self):
        # test basic transformations first
        # vector pointing east at 0, 0 is along y
        vx, vy, vz = pymv.enu_to_ecef_vector(1., 0., 0., 0., 0.)
        # print ('{:9f}, {:9f}, {:9f}'.format(vx, vy, vz))
        asseq(vx, 0.0, 6)
        asseq(vy, 1.0, 6)
        asseq(vz, 0.0, 6)
        # vector pointing up at 0, 0 is along x
        vx, vy, vz = pymv.enu_to_ecef_vector(0., 0., 1., 0., 0.)
        asseq(vx, 1.0, 6)
        asseq(vy, 0.0, 6)
        asseq(vz, 0.0, 6)
        # vector pointing north at 0, 0 is along z
        vx, vy, vz = pymv.enu_to_ecef_vector(0., 1., 0., 0., 0.)
        asseq(vx, 0.0, 6)
        asseq(vy, 0.0, 6)
        asseq(vz, 1.0, 6)

        # east vector at 0, 90 long points along -x
        vx, vy, vz = pymv.enu_to_ecef_vector(1., 0., 0., 0., 90.)
        # print ('{:9f}, {:9f}, {:9f}'.format(vx, vy, vz))
        asseq(vx, -1.0, 6)
        asseq(vy, 0.0, 6)
        asseq(vz, 0.0, 6)
        # vector pointing up at 0, 90 is along y
        vx, vy, vz = pymv.enu_to_ecef_vector(0., 0., 1., 0., 90.)
        asseq(vx, 0.0, 6)
        asseq(vy, 1.0, 6)
        asseq(vz, 0.0, 6)
        # vector pointing north at 0, 90 is along z
        vx, vy, vz = pymv.enu_to_ecef_vector(0., 1., 0., 0., 90.)
        asseq(vx, 0.0, 6)
        asseq(vy, 0.0, 6)
        asseq(vz, 1.0, 6)

        # vector pointing east at 0, 0 is along y
        vx, vy, vz = pymv.enu_to_ecef_vector(1., 0., 0., 0., 180.)
        # print ('{:9f}, {:9f}, {:9f}'.format(vx, vy, vz))
        asseq(vx, 0.0, 6)
        asseq(vy, -1.0, 6)
        asseq(vz, 0.0, 6)
        # vector pointing up at 0, 180 is along -x
        vx, vy, vz = pymv.enu_to_ecef_vector(0., 0., 1., 0., 180.)
        asseq(vx, -1.0, 6)
        asseq(vy, 0.0, 6)
        asseq(vz, 0.0, 6)
        # vector pointing north at 0, 180 is along z
        vx, vy, vz = pymv.enu_to_ecef_vector(0., 1., 0., 0., 180.)
        asseq(vx, 0.0, 6)
        asseq(vy, 0.0, 6)
        asseq(vz, 1.0, 6)

    def test_ecef_to_enu_back_to_ecef(self):
        
        vx = 0.9
        vy = 0.1
        vz = np.sqrt(1. - vx**2+vy**2)

        for lat, lon, alt in zip(omni['p_lat'], omni['p_long'], omni['p_alt']):
            vxx, vyy, vzz = pymv.ecef_to_enu_vector(vx, vy, vz, lat, lon)
            vxx, vyy, vzz = pymv.enu_to_ecef_vector(vxx, vyy, vzz, lat, lon)
            asseq(vx, vxx, 6)
            asseq(vy, vyy, 6)
            asseq(vz, vzz, 6)
        
    def test_enu_to_ecef_back_to_enu(self):
        
        vx = 0.9
        vy = 0.1
        vz = np.sqrt(1. - vx**2+vy**2)

        for lat, lon, alt in zip(omni['p_lat'], omni['p_long'], omni['p_alt']):
            vxx, vyy, vzz = pymv.enu_to_ecef_vector(vx, vy, vz, lat, lon)
            vxx, vyy, vzz = pymv.ecef_to_enu_vector(vxx, vyy, vzz, lat, lon)
            asseq(vx, vxx, 6)
            asseq(vy, vyy, 6)
            asseq(vz, vzz, 6)
