"""Test deprecated functions and keywords."""

import datetime as dt
import numpy as np
import pytest
import warnings

import pysat
from pysat.utils import testing

import OMMBV


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
        """Test deprecated inputs."""

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
        testing.eval_warnings(war, self.warn_msgs)

        return

    @pytest.mark.parametrize("param", ('e_field_scaling_only', 'max_steps',
                                       'edge_length', 'edge_steps'))
    def test_core_scalars_for_mapping(self, param):
        """Test deprecated inputs."""

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
        testing.eval_warnings(war, self.warn_msgs)

        return
