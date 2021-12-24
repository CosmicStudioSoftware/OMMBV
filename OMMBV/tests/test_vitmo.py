
import datetime as dt
import numpy as np
import pandas as pds

import OMMBV
import OMMBV.trace
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


class TestTracingVitmo():
    """Test OMMBV tracing against public results hosted by NASA."""

    def setup(self):
        """Setup test environment before each function."""

        self.inst = pysat.Instrument('pysat', 'testing')
        self.inst.yr = 2010.
        self.inst.doy = 1.

        return

    def teardown(self):
        """Clean up test environment after each function."""

        del self.inst
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
            trace_n.append(OMMBV.trace.field_line_trace(np.array([x, y, z]),
                                                        date, 1., 0.,
                                                        step_size=0.5,
                                                        max_steps=1.E6)[-1, :])
            trace_s.append(OMMBV.trace.field_line_trace(np.array([x, y, z]),
                                                        date, -1., 0.,
                                                        step_size=0.5,
                                                        max_steps=1.E6)[-1, :])

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
