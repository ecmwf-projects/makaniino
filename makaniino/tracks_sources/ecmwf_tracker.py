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

from makaniino.tracks_sources.base import TracksSource_Base

logger = logging.getLogger(__name__)


class TracksSource_ECMWFTracks(TracksSource_Base):

    raw_indexes = [
        ("date", "DATE"),
        ("time", "TIME"),
        ("id", "SID"),
        ("age",),
        ("latF", "LAT"),
        ("lonF", "LON"),
        ("mslpF",),
        ("speedF",),
        ("dpF",),
        ("lat_wF",),
        ("lon_wF",),
        ("maxwindF", "wind"),
        ("latO",),
        ("lonO",),
        ("mslpO",),
        ("speedO",),
        ("dpO",),
        ("distFO",),
        ("dist_mw",),
    ]

    # time interval between records
    delta_hours = 12

    @classmethod
    def from_file(cls, file_name):

        df = pd.read_csv(
            file_name,
            sep=" ",
            header=0,
            index_col=False,
            names=[f[-1] for f in cls.raw_indexes],
        )

        # merge the first 2 column and rename the
        # column according to a map
        df["ISO_TIME"] = df["DATE"].str.replace("/", "-") + " " + df["TIME"] + ":00"

        # sort value according to time
        df.sort_values(by=["ISO_TIME"])

        return cls(df)

    def get_cyclone_coords_at_time(self, time_stamp):
        """
        Gets the makaniino coordinates at a specified time
        """

        ts_str = time_stamp.strftime("%Y-%m-%d %H:%M:00")
        logger.debug("looking for: %s", time_stamp)

        day_labels = self.df[(self.df.ISO_TIME == ts_str)]
        # logger.debug(day_labels)

        points = []
        for index, row in day_labels.iterrows():
            lat = row["LAT"]
            lon = row["LON"]

            points.append([lat, lon])

        logger.info("found : %s points", len(points))
        return points
