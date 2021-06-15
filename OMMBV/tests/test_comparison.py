import datetime as dt
import numpy as np
import matplotlib.pyplot as plt

import OMMBV

import pysat
import apexpy


class TestComparison():

    def __init__(self):
        # placeholder for data management features
        self.inst = pysat.Instrument('pysat', 'testing')
        self.inst.yr = 2019
        self.inst.doy = 365
        self.inst.date = dt.datetime(2019, 12, 31)

        return

    def test_apexpy_v_OMMBV(self):
        """Comparison with apexpy along magnetic equator"""
        # generate test values
        glongs = np.arange(99)*3.6 - 180.
        glats = np.zeros(99) + 10.
        alts = np.zeros(99) + 450.
        # date of run
        date = self.inst.date
        # map to the magnetic equator
        ecef_x, ecef_y, ecef_z, eq_lat, eq_long, eq_z = OMMBV.apex_location_info(
            glats, glongs, alts,
            [date]*len(alts),
            return_geodetic=True)
        idx = np.argsort(eq_long)
        eq_long = eq_long[idx]
        eq_lat = eq_lat[idx]
        eq_z = eq_z[idx]
        # get apex basis vectors
        apex = apexpy.Apex(date=self.inst.date)
        apex_vecs = apex.basevectors_apex(eq_lat, eq_long, eq_z, coords='geo')
        apex_fa = apex_vecs[8]
        apex_mer = apex_vecs[7]
        apex_zon = apex_vecs[6]

        # normalize into unit vectors
        apex_mer[0, :], apex_mer[1, :], apex_mer[2, :] = OMMBV.normalize_vector(-apex_mer[0, :],
                                                                                       -apex_mer[1, :],
                                                                                       -apex_mer[2, :])

        apex_zon[0, :], apex_zon[1, :], apex_zon[2, :] = OMMBV.normalize_vector(apex_zon[0, :],
                                                                                       apex_zon[1, :],
                                                                                       apex_zon[2, :])
        apex_fa[0, :], apex_fa[1, :], apex_fa[2, :] = OMMBV.normalize_vector(apex_fa[0, :],
                                                                                    apex_fa[1, :],
                                                                                    apex_fa[2, :])

        # calculate mag unit vectors in ECEF coordinates
        out = OMMBV.calculate_mag_drift_unit_vectors_ecef(eq_lat, eq_long, eq_z,
                                                                 [self.inst.date]*len(eq_z))
        zx, zy, zz, fx, fy, fz, mx, my, mz = out

        # convert into north, east, and up system
        ze, zn, zu = OMMBV.ecef_to_enu_vector(zx, zy, zz, eq_lat, eq_long)
        fe, fn, fu = OMMBV.ecef_to_enu_vector(fx, fy, fz, eq_lat, eq_long)
        me, mn, mu = OMMBV.ecef_to_enu_vector(mx, my, mz, eq_lat, eq_long)

        # create inputs straight from IGRF
        igrf_n = []
        igrf_e = []
        igrf_u = []
        for lat, lon, alt in zip(eq_lat, eq_long, eq_z):
            out = OMMBV.igrf.igrf13syn(0, date.year, 1, alt,
                                              np.deg2rad(90 - lat), np.deg2rad(lon))
            out = np.array(out)
            # normalize
            out /= out[-1]
            igrf_n.append(out[0])
            igrf_e.append(out[1])
            igrf_u.append(-out[2])

        # dot product of zonal and field aligned
        dot_apex_zonal = apex_fa[0, :]*apex_zon[0, :] + apex_fa[1, :]*apex_zon[1, :] + apex_fa[2, :]*apex_zon[2, :]
        dot_apex_mer = apex_fa[0, :]*apex_mer[0, :] + apex_fa[1, :]*apex_mer[1, :] + apex_fa[2, :]*apex_mer[2, :]

        dotmagvect_zonal = ze*fe + zn*fn + zu*fu
        dotmagvect_mer = me*fe + mn*fn + mu*fu

        assert np.all(dotmagvect_mer < 1.E-8)
        assert np.all(dotmagvect_zonal < 1.E-8)

        try:
            plt.figure()
            plt.plot(eq_long, np.log10(np.abs(dot_apex_zonal)), label='apex_zonal')
            plt.plot(eq_long, np.log10(np.abs(dot_apex_mer)), label='apex_mer')
            plt.plot(eq_long, np.log10(np.abs(dotmagvect_zonal)), label='magv_zonal')
            plt.plot(eq_long, np.log10(np.abs(dotmagvect_mer)), label='magv_mer')
            plt.ylabel('Log Dot Product Magnitude')
            plt.xlabel('Geodetic Longitude (Degrees)')
            plt.legend()
            plt.savefig('dot_product.pdf')
            plt.close()

            plt.figure()
            plt.plot(eq_long, fe, label='fa_e', color='r')
            plt.plot(eq_long, fn, label='fa_n', color='k')
            plt.plot(eq_long, fu, label='fa_u', color='b')

            plt.plot(eq_long, apex_fa[0, :], label='apexpy_e', color='r', linestyle='dotted')
            plt.plot(eq_long, apex_fa[1, :], label='apexpy_n', color='k', linestyle='dotted')
            plt.plot(eq_long, apex_fa[2, :], label='apexpy_u', color='b', linestyle='dotted')

            plt.plot(eq_long, igrf_e, label='igrf_e', color='r', linestyle='-.')
            plt.plot(eq_long, igrf_n, label='igrf_n', color='k', linestyle='-.')
            plt.plot(eq_long, igrf_u, label='igrf_u', color='b', linestyle='-.')

            plt.legend()
            plt.xlabel('Longitude')
            plt.ylabel('Vector Component')
            plt.title('Field Aligned Vector')
            plt.savefig('comparison_field_aligned.pdf')
            plt.close()

            f = plt.figure()
            plt.plot(eq_long, ze, label='zonal_e', color='r')
            plt.plot(eq_long, zn, label='zonal_n', color='k')
            plt.plot(eq_long, zu, label='zonal_u', color='b')

            plt.plot(eq_long, apex_zon[0, :], label='apexpy_e', color='r', linestyle='dotted')
            plt.plot(eq_long, apex_zon[1, :], label='apexpy_n', color='k', linestyle='dotted')
            plt.plot(eq_long, apex_zon[2, :], label='apexpy_u', color='b', linestyle='dotted')

            plt.legend()
            plt.xlabel('Longitude')
            plt.ylabel('Vector Component')
            plt.title('Zonal Vector')
            plt.savefig('comparison_zonal.pdf')
            plt.close()

            f = plt.figure()
            plt.plot(eq_long, me, label='mer_e', color='r')
            plt.plot(eq_long, mn, label='mer_n', color='k')
            plt.plot(eq_long, mu, label='mer_u', color='b')

            plt.plot(eq_long, apex_mer[0, :], label='apexpy_e', color='r', linestyle='dotted')
            plt.plot(eq_long, apex_mer[1, :], label='apexpy_n', color='k', linestyle='dotted')
            plt.plot(eq_long, apex_mer[2, :], label='apexpy_u', color='b', linestyle='dotted')

            plt.legend()
            plt.xlabel('Longitude')
            plt.ylabel('Vector Component')
            plt.title('Meridional Vector')
            plt.savefig('comparison_meridional.pdf')
            plt.close()
        except:
            pass
