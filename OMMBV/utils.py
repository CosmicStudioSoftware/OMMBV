"""General utilities needed to support OMMBV."""

import datetime as dt
import numpy as np


def datetimes_to_doubles(dates):
    """Convert input datetimes to normalized double for IGRF.

    Parameters
    ----------
    dates : list-like of dt.datetimes

    Returns
    -------
    ddates : np.array, dtype=np.float64
      Dates cast as year.fractional_year

    """

    # Need a double variable for time.
    # First, get day of year as well as the year
    doy = np.array([(time - dt.datetime(time.year, 1, 1)).days
                    for time in dates], dtype=np.float64)
    years = np.array([time.year for time in dates], dtype=np.float64)

    # Number of days in year
    num_doy_year = np.array([(dt.datetime(time.year + 1, 1, 1)
                              - dt.datetime(time.year, 1, 1)).days
                             for time in dates], dtype=np.float64)

    # Time in hours, relative to midnight
    time = np.array([(time.hour + time.minute / 60. + time.second / 3600.) / 24.
                     for time in dates], dtype=np.float64)

    # Create double variable for time
    ddates = years + (doy + time) / (num_doy_year + 1.)

    return ddates
