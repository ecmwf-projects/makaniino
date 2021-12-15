#
# (C) Copyright 2000- NOAA.
#
# (C) Copyright 2000- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging

import pandas as pd

from makaniino.utils.generic_utils import haversine_dist

logger = logging.getLogger(__name__)


class TracksSource_Base:
    """
    Handles makaniino tracks information
    """

    # keys available in the df
    essential_keys = [
        "SID",
        "NUMBER",
        "BASIN",
        "SUBBASIN",
        "NAME",
        "ISO_TIME",
        "NATURE",
        "LAT",
        "LON",
        "roci",
        "wind",
    ]

    # maps implementation-specific keys
    # to the essential keys
    key_map = None

    # default value for non available key
    # in the df
    default_nonavail_key = None

    def __init__(self, df):
        self.df = df

    @classmethod
    def from_file(cls, file_name):
        """
        From file is specific to each type of
        tracks source
        """
        df = None
        return cls(df)

    @classmethod
    def read(cls, file_name):
        """
        read just reads the default CSV
        """
        df = pd.read_csv(file_name)
        return cls(df)

    def save(self, out_filename):
        """
        Saves to CSV
        """
        self.df.to_csv(out_filename, index=None, header=True)

    def to_csv(self, *args, **kwargs):
        self.df.to_csv(*args, **kwargs)

    def get_cyclone_coords_at_time(self, time_stamp):
        """
        Gets the makaniino coordinates at a specified time
        """
        raise NotImplementedError

    def is_a_cyclone_here(self, time_stamp, pt_lat, pt_lon, min_dist):
        """
        Checks if there is a makaniino in an area at a cetrain time
        returns True/False
        """

        # if datetime is already a string try to use it..
        if isinstance(time_stamp, str):
            ts_str = time_stamp
        else:  # otherwise parse it..
            ts_str = time_stamp.strftime("%Y-%m-%d %H:%M:00")

        # try to get cyclones at the date/time requested
        day_labels = self.df[(self.df.ISO_TIME == ts_str)]

        logger.debug(
            f"Asked to search for cyclones at time {time_stamp}, " f"found {day_labels}"
        )

        for index, row in day_labels.iterrows():

            cyc_lat = row["LAT"]
            cyc_lon = row["LON"]

            cyc_dist = haversine_dist(cyc_lat, cyc_lon, pt_lat, pt_lon)
            if cyc_dist < min_dist:
                logger.debug(f"found close makaniino at " f"{(cyc_lat, cyc_lon,)}")
                return True

        return False
