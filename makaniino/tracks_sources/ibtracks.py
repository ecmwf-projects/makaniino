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
import math
from datetime import timedelta

import numpy as np
import pandas as pd

from makaniino.tracks_sources.base import TracksSource_Base

logger = logging.getLogger(__name__)


class TracksSource_IBTracks(TracksSource_Base):

    winds = [
        "WMO_WIND",
        "USA_WIND",
        "TOKYO_WIND",
        "CMA_WIND",
        "NEWDELHI_WIND",
        "REUNION_WIND",
        "BOM_WIND",
        "WELLINGTON_WIND",
        "TD9636_WIND",
        "TD9635_WIND",
        "NEUMANN_WIND",
        "MLC_WIND",
    ]

    # time interval between records
    delta_hours = 3

    @classmethod
    def from_file(cls, file_name):
        df = cls.readIBTracs(file_name)
        return cls(df)

    def get_cyclone_coords_at_time(self, time_stamp):
        """
        Gets the makaniino coordinates at a specified time
        """
        delta_min = self.delta_hours * 60
        delta_min_double = 2 * delta_min
        if time_stamp.hour % 6 == self.delta_hours:
            logger.debug("Determining intermediate points")
            _ts = time_stamp + timedelta(minutes=delta_min)
            points = self.getPointsForIntermediateTime(_ts, time_diff=delta_min_double)
        else:
            ts_str = time_stamp.strftime("%Y-%m-%d %H:%M:00")
            points = self.getPointsForTime(ts_str)

        return points

    @classmethod
    def readIBTracs(cls, file):
        """
        Read ibtracks from CSV file
        """

        dforig = pd.read_csv(file, sep=",", header=0, skiprows=[1, 2])
        df = dforig.replace(-999.0, np.nan)

        # extract only centers between values
        # df = df.query('Longitude >= 143 | Longitude <= -54')

        # consolidate to a single radius observation
        df[["roci"]] = df.apply(cls.bestRadius, axis=1)
        df[["wind"]] = df.apply(cls.bestWind, axis=1)

        # extract only needed columns
        df = df[
            [
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
        ]

        # extract rows where roci is defined
        # df = df[np.isfinite(df['roci'])]

        # ************** filtering of cyclones *************
        # extract where wind exceeds certain value
        # wind speed is in knots
        # TS > 34
        # 1 > 64
        # 2 > 82
        # 3 > 96
        # 4 > 113
        # 5 > 137
        df = df[df["wind"] > 34]
        # **************************************************

        # sort fields by time
        df.sort_values(by=["ISO_TIME"])
        return df

    def getPointsForTime(self, time_step):

        logger.debug("looking for: %s", time_step)

        day_labels = self.df[(self.df.ISO_TIME == time_step)]
        logger.debug(day_labels)

        points = []
        for index, row in day_labels.iterrows():
            lat = row["LAT"]
            lon = row["LON"]
            # roci = row['roci']
            # name = row['NAME']
            # wind = row['wind']

            points.append([lat, lon])

        logger.info("found : %s points", len(points))
        return points

    def getPointsForIntermediateTime(self, valid_time, time_diff=360):

        timestamp = valid_time.strftime("%Y-%m-%d %H:%M:00")
        logger.info("looking for : %s", timestamp)

        day_labels = self.df[(self.df.ISO_TIME == timestamp)]
        previous_time = valid_time - timedelta(minutes=time_diff)
        logger.info("previous time : %s", previous_time)

        previous_stamp = previous_time.strftime("%Y-%m-%d %H:%M:00")
        previous_labels = self.df[(self.df.ISO_TIME == previous_stamp)]

        points = []

        for index, row in day_labels.iterrows():

            sn = row["SID"]
            prev = previous_labels.loc[previous_labels.SID == sn]

            if len(prev) > 0:
                prev_row = prev.iloc[0]

                cur_lat = row["LAT"]
                cur_lon = row["LON"]

                prev_lat = prev_row["LAT"]
                prev_lon = prev_row["LON"]

                lat, lon = self.midpoint(prev_lat, prev_lon, cur_lat, cur_lon)
                # print (lat, " :: ", lon)
                points.append([lat, lon])

        logger.debug("found : %s points", len(points))
        return points

    @classmethod
    def has_numbers(cls, input_string):
        return any(char.isdigit() for char in input_string)

    @classmethod
    def bestRadius(cls, row):
        """
        Determine radius, could be from three
        different columns depending on who is reporting
        """
        r1 = row["TD9635_ROCI"]
        r2 = row["BOM_ROCI"]
        r3 = row["USA_ROCI"]

        if cls.has_numbers(r1):
            r1 = int(r1)
        else:
            r1 = np.nan

        if cls.has_numbers(r2):
            r2 = int(r2)
        else:
            r2 = np.nan

        if cls.has_numbers(r3):
            r3 = int(r3)
        else:
            r3 = np.nan

        result = r1

        if not math.isnan(r2):
            result = r2
        elif not math.isnan(r3):
            result = r3

        return pd.Series(dict(roci=result))

    @classmethod
    def bestWind(cls, row):
        """
        Determine wind, could be from three
        different columns depending on who is reporting
        """

        result = np.nan

        for r in cls.winds:
            value = row[r]
            if cls.has_numbers(value):
                result = int(value)
                break

        return pd.Series(dict(wind=result))

    @classmethod
    def midpoint(cls, x1, x2, y1, y2):
        """
        Input values as degrees
        """

        # Convert to radians
        lat1 = math.radians(x1)
        lon1 = math.radians(x2)
        lat2 = math.radians(y1)
        lon2 = math.radians(y2)

        bx = math.cos(lat2) * math.cos(lon2 - lon1)
        by = math.cos(lat2) * math.sin(lon2 - lon1)

        lat3 = math.atan2(
            math.sin(lat1) + math.sin(lat2),
            math.sqrt((math.cos(lat1) + bx) * (math.cos(lat1) + bx) + by ** 2),
        )

        lon3 = lon1 + math.atan2(by, math.cos(lat1) + bx)

        return [round(math.degrees(lat3), 2), round(math.degrees(lon3), 2)]


class TracksSource_IBTracks_All(TracksSource_IBTracks):
    """
    This class is similar to the default ibtracks class
    but it does not filter out cyclones whose wind speed is < 34 kts
    """

    @classmethod
    def readIBTracs(cls, file):
        """
        Read ibtracks from CSV file
        """

        dforig = pd.read_csv(file, sep=",", header=0, skiprows=[1, 2])
        df = dforig.replace(-999.0, np.nan)

        # extract only centers between values
        # df = df.query('Longitude >= 143 | Longitude <= -54')

        # consolidate to a single radius observation
        df[["roci"]] = df.apply(cls.bestRadius, axis=1)
        df[["wind"]] = df.apply(cls.bestWind, axis=1)

        # extract only needed columns
        df = df[
            [
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
        ]

        # extract rows where roci is defined
        # df = df[np.isfinite(df['roci'])]

        # sort fields by time
        df.sort_values(by=["ISO_TIME"])

        return df
