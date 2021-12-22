"""Test deprecated functions and keywords."""

import datetime as dt
import numpy as np
import pytest
import warnings

import OMMBV


def eval_warnings(warns, check_msgs, warn_type=DeprecationWarning):
    """Evaluate warnings by category and message.

    Borrowed from pysat `develop` branch. Will be in the release
    after v3.0.1.

    Parameters
    ----------
    warns : list
        List of warnings.WarningMessage objects
    check_msgs : list
        List of strings containing the expected warning messages
    warn_type : type
        Type for the warning messages (default=DeprecationWarning)

    Raises
    ------
    AssertionError
        If warning category doesn't match type or an expected message is missing

    """

    # Initialize the output
    found_msgs = [False for msg in check_msgs]

    # Test the warning messages, ensuring each attribute is present
    for iwar in warns:
        for i, msg in enumerate(check_msgs):
            if str(iwar.message).find(msg) >= 0:
                assert iwar.category == warn_type, \
                    "bad warning type for message: {:}".format(msg)
                found_msgs[i] = True

    assert np.all(found_msgs), "did not find {:d} expected {:}".format(
        len(found_msgs) - np.sum(found_msgs), repr(warn_type))

    return


class TestDeprecations(object):
    """Unit tests for deprecated inputs and functions."""

    def setup(self):
        """Setup test environment before each function."""
        warnings.simplefilter("always", DeprecationWarning)

        self.date = dt.datetime(2020, 1, 1)
        self.warn_msgs = ''
        return

    def teardown(self):
        """Clean up test environment after each function."""
        del self.warn_msgs, self.date
        return

    @pytest.mark.parametrize("param", ('max_steps', 'ref_height', 'steps',
                                       'scalar', 'edge_steps'))
    def test_core_geomagnetic_ecef(self, param):
        """Test deprecated inputs, `calculate_mag_drift_unit_vectors_ecef`."""

        self.warn_msgs = [" ".join([param, "is deprecated, non-functional,",
                                    "and will be removed after OMMBV v1.0.0."])]
        self.warn_msgs = np.array(self.warn_msgs)

        # Prep input
        kwargs = {param: 1}

        # Catch the warnings.
        with warnings.catch_warnings(record=True) as war:
            OMMBV.calculate_mag_drift_unit_vectors_ecef([0.], [0.], [550.],
                                                        [self.date], **kwargs)

        # Ensure the minimum number of warnings were raised.
        assert len(war) >= len(self.warn_msgs)

        # Test the warning messages, ensuring each attribute is present.
        eval_warnings(war, self.warn_msgs)

        return

    @pytest.mark.parametrize("param", ('e_field_scaling_only', 'max_steps',
                                       'edge_length', 'edge_steps'))
    def test_core_scalars_for_mapping(self, param):
        """Test deprecated inputs, `scalars_for_mapping_ion_drifts`."""

        self.warn_msgs = [" ".join([param, "is deprecated, non-functional,",
                                    "and will be removed after OMMBV v1.0.0."])]
        self.warn_msgs = np.array(self.warn_msgs)

        # Prep input
        kwargs = {param: 1}

        # Catch the warnings.
        with warnings.catch_warnings(record=True) as war:
            OMMBV.scalars_for_mapping_ion_drifts([0.], [0.], [550.], [self.date],
                                                 **kwargs)

        # Ensure the minimum number of warnings were raised.
        assert len(war) >= len(self.warn_msgs)

        # Test the warning messages, ensuring each attribute is present.
        eval_warnings(war, self.warn_msgs)

        return


class TestCoreDeprecations(object):
    """Unit tests for deprecated core functions."""

    def setup(self):
        """Setup test environment before each function."""
        warnings.simplefilter("always", DeprecationWarning)

        self.date = dt.datetime(2020, 1, 1)
        self.warn_msgs = ''

        return

    def teardown(self):
        """Clean up test environment after each function."""
        del self.warn_msgs, self.date
        return

    @pytest.mark.parametrize("func", (OMMBV.geocentric_to_ecef,
                                      OMMBV.ecef_to_geocentric,
                                      OMMBV.ecef_to_geodetic,
                                      OMMBV.geodetic_to_ecef,
                                      OMMBV.python_ecef_to_geodetic))
    def test_core_trans_functions_deprecated(self, func):
        """Test deprecated wrapped functions."""

        self.warn_msgs = ["".join(["Function moved to `OMMBV.trans`, "
                                   "deprecated wrapper will be removed after ",
                                   "OMMBV v1.0.0."])]
        self.warn_msgs = np.array(self.warn_msgs)

        # Catch the warnings.
        with warnings.catch_warnings(record=True) as war:
            func(np.array([0.]), np.array([0.]), np.array([550.]))

        # Ensure the minimum number of warnings were raised.
        assert len(war) >= len(self.warn_msgs)

        # Test the warning messages, ensuring each attribute is present.
        eval_warnings(war, self.warn_msgs)

        return

    def test_core_trans_kwargs_deprecated(self):
        """Test deprecated keywords in wrapped transformation functions."""

        self.warn_msgs = ["".join(["`method` must be a string value in ",
                                   "v1.0.0+. Setting to function default."])]
        self.warn_msgs = np.array(self.warn_msgs)

        # Catch the warnings.
        with warnings.catch_warnings(record=True) as war:
            OMMBV.python_ecef_to_geodetic(np.array([0.]), np.array([0.]),
                                          np.array([550.]), method=None)

        # Ensure the minimum number of warnings were raised.
        assert len(war) >= len(self.warn_msgs)

        # Test the warning messages, ensuring each attribute is present.
        eval_warnings(war, self.warn_msgs)

        return

    @pytest.mark.parametrize("func,args", ([OMMBV.enu_to_ecef_vector,
                                            (1., 0., 0., 0., 0.)],
                                           [OMMBV.ecef_to_enu_vector,
                                            (1., 0., 0., 0., 0.)],
                                           [OMMBV.project_ECEF_vector_onto_basis,
                                            (1., 0., 0., 1., 0., 0., 0., 1.,
                                             0., 0., 0., 1.)],
                                           [OMMBV.normalize_vector,
                                            (1., 0., 0.)],
                                           [OMMBV.cross_product,
                                            (0., 0., 1., 1., 0., 0.)]))
    def test_core_vect_functions_deprecated(self, func, args):
        """Test deprecated wrapped functions."""

        self.warn_msgs = ["".join(["Function moved to `OMMBV.vector`, ",
                                   "deprecated wrapper will be removed after ",
                                   "OMMBV v1.0.0."])]
        self.warn_msgs = np.array(self.warn_msgs)

        # Catch the warnings.
        with warnings.catch_warnings(record=True) as war:
            func(*args)

        # Ensure the minimum number of warnings were raised.
        assert len(war) >= len(self.warn_msgs)

        # Test the warning messages, ensuring each attribute is present.
        eval_warnings(war, self.warn_msgs)

        return

    def test_heritage_functions_deprecated(self):
        """Test deprecated wrapped `OMMBV.heritage` functions."""

        self.warn_msgs = [''.join(['This method now called `apex_distance_',
                                   'after_footpoint_step`. Wrapper will be ',
                                   'removed after OMMBV v1.0.0.'])]
        self.warn_msgs = np.array(self.warn_msgs)

        # Catch the warnings.
        with warnings.catch_warnings(record=True) as war:
            OMMBV.heritage.apex_edge_lengths_via_footpoint(np.array([0.]),
                                                           np.array([0.]),
                                                           np.array([550.]),
                                                           [self.date],
                                                           'north',
                                                           'zonal')

        # Ensure the minimum number of warnings were raised.
        assert len(war) >= len(self.warn_msgs)

        # Test the warning messages, ensuring each attribute is present.
        eval_warnings(war, self.warn_msgs)

        return
