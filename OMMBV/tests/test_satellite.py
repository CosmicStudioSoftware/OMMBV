import OMMBV
import pysat


class TestSatellite():

    def setup(self):

        self.inst = pysat.Instrument('pysat', 'testing', num_samples=32)
        return

    def test_application_add_unit_vectors(self):
        """Check application of unit_vectors to satellite data"""
        self.inst.load(2010, 365)
        self.inst['altitude'] = 550.
        print(self.inst)
        OMMBV.satellite.add_mag_drift_unit_vectors_ecef(self.inst)
        items = ['unit_zon_ecef_x', 'unit_zon_ecef_y', 'unit_zon_ecef_z',
                 'unit_fa_ecef_x', 'unit_fa_ecef_y', 'unit_fa_ecef_z',
                 'unit_mer_ecef_x', 'unit_mer_ecef_y', 'unit_mer_ecef_z']
        for item in items:
            assert item in self.inst.data

    def test_application_add_mag_drifts(self):
        """Check application of unit vectors to drift measurements"""
        self.inst.load(2010, 365)
        self.inst['altitude'] = 550.
        # create false orientation signal
        self.inst['sc_xhat_x'] = 1.
        self.inst['sc_xhat_y'] = 0.
        self.inst['sc_xhat_z'] = 0.
        self.inst['sc_yhat_x'] = 0.
        self.inst['sc_yhat_y'] = 1.
        self.inst['sc_yhat_z'] = 0.
        self.inst['sc_zhat_x'] = 0.
        self.inst['sc_zhat_y'] = 0.
        self.inst['sc_zhat_z'] = 1.

        OMMBV.satellite.add_mag_drift_unit_vectors(self.inst)
        items = ['unit_zon_x', 'unit_zon_y', 'unit_zon_z',
                 'unit_fa_x', 'unit_fa_y', 'unit_fa_z',
                 'unit_mer_x', 'unit_mer_y', 'unit_mer_z']
        for item in items:
            assert item in self.inst.data

        # check adding drifts now
        self.inst['iv_x'] = 150.
        self.inst['iv_y'] = 50.
        self.inst['iv_z'] = -50.
        OMMBV.satellite.add_mag_drifts(self.inst)
        items = ['iv_zon', 'iv_fa', 'iv_mer']
        for item in items:
            assert item in self.inst.data

        # check scaling to footpoints and equator
        self.inst['equ_mer_drifts_scalar'] = 1.
        self.inst['equ_zon_drifts_scalar'] = 1.
        self.inst['north_footpoint_mer_drifts_scalar'] = 1.
        self.inst['north_footpoint_zon_drifts_scalar'] = 1.
        self.inst['south_footpoint_mer_drifts_scalar'] = 1.
        self.inst['south_footpoint_zon_drifts_scalar'] = 1.
        OMMBV.satellite.add_footpoint_and_equatorial_drifts(self.inst)
        items = ['equ_mer_drifts_scalar', 'equ_zon_drifts_scalar',
                 'north_footpoint_mer_drifts_scalar', 'north_footpoint_zon_drifts_scalar',
                 'south_footpoint_mer_drifts_scalar', 'south_footpoint_zon_drifts_scalar']
        for item in items:
            assert item in self.inst.data
