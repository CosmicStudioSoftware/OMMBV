"""Unit tests for `OMMBV.utils`."""

import datetime as dt
import numpy as np

import pysat

import OMMBV.utils as utils


class TestUtils(object):
    """Unit tests for utils."""

    def setup(self):
        """Setup test environment before each function."""

        self.start_date = dt.datetime(2000, 1, 1)
        self.end_date = dt.datetime(2020, 12, 31)
        self.dates = pysat.utils.time.create_date_range(self.start_date,
                                                        self.end_date)
        return

    def teardown(self):
        """Clean up test environment after each function."""

        del self.start_date, self.end_date, self.dates
        return

    def test_datetimes_to_doubles_year(self):
        """Test year output from `datetimes_to_doubles`."""

        out_dates = utils.datetimes_to_doubles(self.dates)

        # Check year for accuracy
        target_year = [date.year for date in self.dates]
        calc_year = [int(date) for date in out_dates]

        assert np.all(calc_year == target_year)
        return

    def test_datetimes_to_doubles_day(self):
        """
        Test day from `datetimes_to_doubles` for uniqueness and monotonicity.
        """

        out_dates = utils.datetimes_to_doubles(self.dates)

        # Get official unique years from datetimes
        target_year = [date.year for date in self.dates]
        u_years = np.unique(target_year)

        # Identify outputs from each unique year and test
        for year in u_years:
            idx, = np.where(target_year == year)
            sub_dates = out_dates[idx]

            # Unique test
            assert len(np.unique(sub_dates)) == len(sub_dates)

            # Monotonic increasing test
            assert np.all(np.diff(sub_dates) > 0)

        return

    def test_datetimes_to_doubles_output_type(self):
        """
        Test np.array of type np.float64 output from `datetimes_to_doubles`.
        """

        out_dates = utils.datetimes_to_doubles(self.dates)

        # Check type
        assert out_dates.dtype == np.float64
        return
