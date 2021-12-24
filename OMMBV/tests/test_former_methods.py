import datetime as dt
import numpy as np
import pytest

import OMMBV
import OMMBV.heritage
from OMMBV.tests.test_core import gen_plot_grid_fixed_alt
import OMMBV.vector


class TestIntegratedMethods(object):

    def setup(self):
        """Setup test environment before each function."""
        self.lats, self.longs, self.alts = gen_plot_grid_fixed_alt(550.)
        self.date = dt.datetime(2000, 1, 1)
        return

    def teardown(self):
        """Clean up test environment after each function."""
        del self.lats, self.longs, self.alts

    def test_integrated_unit_vector_components(self):
        """Test Field-Line Integrated Unit Vectors"""

        p_lats, p_longs, p_alts = gen_plot_grid_fixed_alt(550.)
        # data returned are the locations along each direction
        # the full range of points obtained by iterating over all
        # recasting alts into a more convenient form for later calculation
        p_alts = [p_alts[0]]*len(p_longs)
        zvx = np.zeros((len(p_lats), len(p_longs)))
        zvy = zvx.copy()
        zvz = zvx.copy()
        mx = zvx.copy()
        my = zvx.copy()
        mz = zvx.copy()
        bx = zvx.copy()
        by = zvx.copy()
        bz = zvx.copy()
        date = dt.datetime(2000, 1, 1)

        fcn = OMMBV.heritage.calculate_integrated_mag_drift_unit_vectors_ecef
        for i, p_lat in enumerate(p_lats):
            (tzx, tzy, tzz,
             tbx, tby, tbz,
             tmx, tmy, tmz
             ) = fcn([p_lat]*len(p_longs), p_longs, p_alts, [date]*len(p_longs),
                     steps=None, max_steps=10000, step_size=10.,
                     ref_height=120.)
            (zvx[i, :], zvy[i, :],
             zvz[i, :]) = OMMBV.vector.ecef_to_enu(tzx, tzy, tzz,
                                                   [p_lat] * len(p_longs),
                                                   p_longs)
            (bx[i, :], by[i, :],
             bz[i, :])= OMMBV.vector.ecef_to_enu(tbx, tby, tbz,
                                                 [p_lat] * len(p_longs),
                                                 p_longs)
            (mx[i, :], my[i, :],
             mz[i, :]) = OMMBV.vector.ecef_to_enu(tmx, tmy, tmz,
                                                  [p_lat] * len(p_longs),
                                                  p_longs)

        # Zonal generally eastward
        assert np.all(zvx > 0.7)

        # Meridional generally not eastward
        assert np.all(mx < 0.4)

        return

    def test_integrated_unit_vector_component_stepsize_sensitivity(self):
        """Test field-line integrated vector sensitivity."""
        p_lats, p_longs, p_alts = gen_plot_grid_fixed_alt(550.)
        # data returned are the locations along each direction
        # the full range of points obtained by iterating over all
        # recasting alts into a more convenient form for later calculation
        p_alts = [p_alts[0]]*len(p_longs)

        # zonal vector components
        # +1 on length of longitude array supports repeating first element
        # shows nice periodicity on the plots
        zvx = np.zeros((len(p_lats), len(p_longs)))
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
        fcn = OMMBV.heritage.calculate_integrated_mag_drift_unit_vectors_ecef
        for i, p_lat in enumerate(p_lats):
            (tzx, tzy, tzz, tbx, tby, tbz, tmx, tmy, tmz
             ) = fcn([p_lat]*len(p_longs), p_longs, p_alts, [date]*len(p_longs),
                     steps=None, max_steps=10000, step_size=10.,
                     ref_height=120.)
            (zvx[i, :], zvy[i, :],
             zvz[i, :]) = OMMBV.vector.ecef_to_enu(tzx, tzy, tzz,
                                                   [p_lat] * len(p_longs),
                                                   p_longs)
            (bx[i, :], by[i, :],
             bz[i, :]) = OMMBV.vector.ecef_to_enu(tbx, tby, tbz,
                                                  [p_lat] * len(p_longs),
                                                  p_longs)
            (mx[i, :], my[i, :],
             mz[i, :]) = OMMBV.vector.ecef_to_enu(tmx, tmy, tmz,
                                                  [p_lat] * len(p_longs),
                                                  p_longs)

            # second run
            (tzx, tzy, tzz, tbx, tby, tbz, tmx, tmy, tmz
             ) = fcn([p_lat]*len(p_longs), p_longs, p_alts, [date]*len(p_longs),
                     steps=None, max_steps=1000, step_size=100.,
                     ref_height=120.)
            _a, _b, _c = OMMBV.vector.ecef_to_enu(tzx, tzy, tzz,
                                                  [p_lat] * len(p_longs),
                                                  p_longs)
            # take difference with first run
            zvx[i, :] = (zvx[i, :] - _a) / zvx[i, :]
            zvy[i, :] = (zvy[i, :] - _b) / zvy[i, :]
            zvz[i, :] = (zvz[i, :] - _c) / zvz[i, :]

            _a, _b, _c = OMMBV.vector.ecef_to_enu(tbx, tby, tbz,
                                                  [p_lat] * len(p_longs),
                                                  p_longs)
            # take difference with first run
            bx[i, :] = (bx[i, :] - _a) / bx[i, :]
            by[i, :] = (by[i, :] - _b) / by[i, :]
            bz[i, :] = (bz[i, :] - _c) / bz[i, :]

            _a, _b, _c = OMMBV.vector.ecef_to_enu(tmx, tmy, tmz,
                                                  [p_lat] * len(p_longs),
                                                  p_longs)
            # take difference with first run
            mx[i, :] = (mx[i, :] - _a) / mx[i, :]
            my[i, :] = (my[i, :] - _b) / my[i, :]
            mz[i, :] = (mz[i, :] - _c) / mz[i, :]

        items = [zvx, zvy, zvz, bx, by, bz, mx, my, mz]
        for item in items:
            assert np.all(item < 1.E-2)

        return

    @pytest.mark.parametrize("e_field_scaling", (True, False))
    def test_heritage_geomag_efield_scalars(self, e_field_scaling):
        """Test heritage code path for scaling electric fields/ion drifts."""

        data = {}
        fcn = OMMBV.heritage.heritage_scalars_for_mapping_ion_drifts
        for i, p_lat in enumerate(self.lats):
            templ = [p_lat] * len(self.longs)
            tempd = [self.date] * len(self.longs)
            scalars = fcn(templ, self.longs, self.alts, tempd,
                          e_field_scaling_only=e_field_scaling)
            for scalar in scalars:
                if scalar not in data:
                    data[scalar] = np.full((len(self.lats), len(self.longs)),
                                           np.nan)

                data[scalar][i, :] = scalars[scalar]

        if e_field_scaling:
            assert len(scalars.keys()) == 6
        else:
            assert len(scalars.keys()) == 12

        for scalar in scalars:
            assert np.all(np.isfinite(scalars[scalar]))

        return

    @pytest.mark.parametrize("direction", ('zonal', 'meridional', 'aligned'))
    def test_apex_distance_local_step(self, direction):
        """Test heritage code path for apex distance after local step."""

        fcn = OMMBV.heritage.apex_distance_after_local_step
        for i, p_lat in enumerate(self.lats):
            templ = [p_lat] * len(self.longs)
            tempd = [self.date] * len(self.longs)
            dist = fcn(templ, self.longs, self.alts, tempd,
                       direction, ecef_input=False,
                       return_geodetic=True)
            assert np.all(np.isfinite(dist))

            if direction == 'aligned':
                np.testing.assert_allclose(dist, 0., atol=1.E-2)

        return
