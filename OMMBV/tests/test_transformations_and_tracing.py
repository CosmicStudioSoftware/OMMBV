
import datetime as dt
import numpy as np
import pandas as pds

import OMMBV
import OMMBV.trans
import OMMBV.vector
import pysat

# Results from omniweb calculator
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
omni = pds.DataFrame(omni_list, columns=['p_alt', 'p_lat', 'p_long', 'n_lat',
                                         'n_long', 's_lat', 's_long'])


class TestTracing():

    def __init__(self):
        # placeholder for data management features
        self.inst = pysat.Instrument('pysat', 'testing')
        self.inst.yr = 2010.
        self.inst.doy = 1.

        return

    def test_field_line_tracing_against_vitmo(self):
        """Compare model to http://omniweb.gsfc.nasa.gov/vitmo/cgm_vitmo.html"""

        # convert position to ECEF
        ecf_x, ecf_y, ecf_z = OMMBV.trans.geocentric_to_ecef(omni['p_lat'],
                                                             omni['p_long'],
                                                             omni['p_alt'])
        trace_n = []
        trace_s = []
        date = dt.datetime(2000, 1, 1)
        for x, y, z in zip(ecf_x, ecf_y, ecf_z):
            # Trace north and south, take last points
            trace_n.append(
                OMMBV.field_line_trace(np.array([x, y, z]), date, 1., 0.,
                                       step_size=0.5, max_steps=1.E6)[-1, :])
            trace_s.append(
                OMMBV.field_line_trace(np.array([x, y, z]), date, -1., 0.,
                                       step_size=0.5, max_steps=1.E6)[-1, :])
        trace_n = pds.DataFrame(trace_n, columns=['x', 'y', 'z'])
        (trace_n['lat'],
         trace_n['long'],
         trace_n['altitude']) = OMMBV.trans.ecef_to_geocentric(trace_n['x'],
                                                               trace_n['y'],
                                                               trace_n['z'])

        trace_s = pds.DataFrame(trace_s, columns=['x', 'y', 'z'])
        (trace_s['lat'],
         trace_s['long'],
         trace_s['altitude']) = OMMBV.trans.ecef_to_geocentric(trace_s['x'],
                                                               trace_s['y'],
                                                               trace_s['z'])
        # Ensure longitudes are all 0-360
        omni.loc[omni['n_long'] < 0, 'n_long'] += 360.
        omni.loc[omni['s_long'] < 0, 's_long'] += 360.
        trace_n.loc[trace_n['long'] < 0, 'long'] += 360.
        trace_s.loc[trace_s['long'] < 0, 'long'] += 360.

        # Compute difference between OMNI and local calculation.
        # There is a known difference near 0 longitude, ignore this area,
        diff_n_lat = (omni['n_lat'] - trace_n['lat'])[4:-4]
        diff_n_lon = (omni['n_long'] - trace_n['long'])[4:-4]
        diff_s_lat = (omni['s_lat'] - trace_s['lat'])[4:-4]
        diff_s_lon = (omni['s_long'] - trace_s['long'])[4:-4]

        # Better than 0.5 km accuracy expected for settings above
        assert np.all(np.nanstd(diff_n_lat) < .5)
        assert np.all(np.nanstd(diff_n_lon) < .5)
        assert np.all(np.nanstd(diff_s_lat) < .5)
        assert np.all(np.nanstd(diff_s_lon) < .5)

        return


    def test_unit_vector_properties(self):
        """Test basic vector properties along field lines."""

        # Start with a set of locations
        p_long = np.arange(0., 360., 12.)
        p_alt = 0 * p_long + 550.
        p_lats = [5., 10., 15., 20., 25., 30.]

        truthiness = []
        for i, p_lat in enumerate(p_lats):

            date = dt.datetime(2000, 1, 1)
            ecef_x, ecef_y, ecef_z = OMMBV.trans.geocentric_to_ecef(p_lat,
                                                                    p_long,
                                                                    p_alt)

            for j, (x, y, z) in enumerate(zip(ecef_x, ecef_y, ecef_z)):
                # Perform field line traces
                trace_n = OMMBV.field_line_trace(np.array([x, y, z]), date, 1.,
                                                 0., step_size=.5,
                                                 max_steps=1.E6)
                trace_s = OMMBV.field_line_trace(np.array([x, y, z]), date, -1.,
                                                 0., step_size=.5,
                                                 max_steps=1.E6)

                # Combine together, S/C position is first for both
                # Reverse first array and join so plotting makes sense
                trace = np.vstack((trace_n[::-1], trace_s))
                trace = pds.DataFrame(trace, columns=['x', 'y', 'z'])

                # clear stored data
                self.inst.data = pds.DataFrame()
                # downselect, reduce number of points
                trace = trace.loc[::1000, :]

                # compute magnetic field vectors
                # need to provide alt, latitude, and longitude in geodetic coords
                (latitude,
                 longitude,
                 altitude) = OMMBV.ecef_to_geodetic(trace['x'], trace['y'],
                                                    trace['z'])

                self.inst[:, 'latitude'] = latitude
                self.inst[:, 'longitude'] = longitude
                self.inst[:, 'altitude'] = altitude

                # Store values for plotting locations for vectors
                self.inst[:, 'x'] = trace['x'].values
                self.inst[:, 'y'] = trace['y'].values
                self.inst[:, 'z'] = trace['z'].values
                idx, = np.where(self.inst['altitude'] > 250.)
                self.inst.data = self.inst[idx, :]

                # also need to provide transformation from ECEF to S/C
                # going to leave that a null transformation so we can plot in ECF
                (self.inst[:, 'sc_xhat_x'],
                 self.inst[:, 'sc_xhat_y'],
                 self.inst[:, 'sc_xhat_z']) = 1., 0., 0.
                (self.inst[:, 'sc_yhat_x'],
                 self.inst[:, 'sc_yhat_y'],
                 self.inst[:, 'sc_yhat_z']) = 0., 1., 0.
                (self.inst[:, 'sc_zhat_x'],
                 self.inst[:, 'sc_zhat_y'],
                 self.inst[:, 'sc_zhat_z']) = 0., 0., 1.
                self.inst.data.index = pysat.utils.time.create_date_range(dt.datetime(2000, 1, 1),
                                                                          dt.datetime(2000, 1, 1) +
                                                                          pds.DateOffset(
                                                                              seconds=len(self.inst.data) - 1),
                                                                          freq='S')
                OMMBV.satellite.add_mag_drift_unit_vectors(self.inst)

                # Check that vectors norm to 1
                assert np.all(np.sqrt(self.inst['unit_zon_x'] ** 2 +
                                      self.inst['unit_zon_y'] ** 2 +
                                      self.inst['unit_zon_z'] ** 2) > 0.999999)
                assert np.all(np.sqrt(self.inst['unit_fa_x'] ** 2 +
                                      self.inst['unit_fa_y'] ** 2 +
                                      self.inst['unit_fa_z'] ** 2) > 0.999999)
                assert np.all(np.sqrt(self.inst['unit_mer_x'] ** 2 +
                                      self.inst['unit_mer_y'] ** 2 +
                                      self.inst['unit_mer_z'] ** 2) > 0.999999)

                # Confirm vectors are mutually orthogonal
                dot1 = self.inst['unit_zon_x'] * self.inst['unit_fa_x'] \
                       + self.inst['unit_zon_y'] * self.inst['unit_fa_y'] \
                       + self.inst['unit_zon_z'] * self.inst['unit_fa_z']
                dot2 = self.inst['unit_zon_x'] * self.inst['unit_mer_x']\
                       + self.inst['unit_zon_y'] * self.inst['unit_mer_y'] \
                       + self.inst['unit_zon_z'] * self.inst['unit_mer_z']
                dot3 = self.inst['unit_fa_x'] * self.inst['unit_mer_x']\
                       + self.inst['unit_fa_y'] * self.inst['unit_mer_y']\
                       + self.inst['unit_fa_z'] * self.inst['unit_mer_z']
                assert np.all(np.abs(dot1) < 1.E-6)
                assert np.all(np.abs(dot2) < 1.E-6)
                assert np.all(np.abs(dot3) < 1.E-6)

                # ensure that zonal vector is generally eastward
                ones = np.ones(len(self.inst.data.index))
                zeros = np.zeros(len(self.inst.data.index))
                ex, ey, ez = OMMBV.vector.enu_to_ecef(ones, zeros, zeros,
                                                      self.inst['latitude'],
                                                      self.inst['longitude'])
                nx, ny, nz = OMMBV.vector.enu_to_ecef(zeros, ones, zeros,
                                                      self.inst['latitude'],
                                                      self.inst['longitude'])
                ux, uy, uz = OMMBV.vector.enu_to_ecef(zeros, zeros, ones,
                                                      self.inst['latitude'],
                                                      self.inst['longitude'])

                dot1 = self.inst['unit_zon_x'] * ex\
                       + self.inst['unit_zon_y'] * ey\
                       + self.inst['unit_zon_z'] * ez
                assert np.all(dot1 > 0.)

                dot1 = self.inst['unit_fa_x'] * nx\
                       + self.inst['unit_fa_y'] * ny\
                       + self.inst['unit_fa_z'] * nz
                assert np.all(dot1 > 0.)

                dot1 = self.inst['unit_mer_x'] * ux\
                       + self.inst['unit_mer_y'] * uy\
                       + self.inst['unit_mer_z'] * uz
                assert np.all(dot1 > 0.)

        assert np.all(truthiness)
