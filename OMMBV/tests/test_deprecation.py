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
