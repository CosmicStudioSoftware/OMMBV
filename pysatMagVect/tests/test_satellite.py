import pysatMagVect
import pysatMagVect as pymv

import pysat

class TestSatellite():

    def setup(self):

        self.inst = pysat.Instrument('pysat', 'testing', sat_id='32')


        return


    def test_application_add_unit_vectors(self):
        self.inst.load(2019, 365)
        pymv.satellite.add_mag_drift_unit_vectors_ecef(self.inst)
        items = ['unit_zon_ecef_x', 'unit_zon_ecef_y', 'unit_zon_ecef_z',
                 'unit_fa_ecef_x', 'unit_fa_ecef_y', 'unit_fa_ecef_z',
                 'unit_mer_ecef_x', 'unit_mer_ecef_y', 'unit_mer_ecef_z']
        for item in items:
            assert item in self.inst.data
