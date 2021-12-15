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
import os
import sys
from datetime import timedelta

import numpy as np

from makaniino.data_handling.datasets.base import DatasetTypes, EntrySchema, RecordSchema
from makaniino.data_preprocess.base import PreProcessorBase
from makaniino.label_data import createLabeledData, pointsToDataPoints
from makaniino.utils.generic_utils import numpy_to_npstring, time_to_numpy

logger = logging.getLogger(__name__)


class PreProcessorTCClasses(PreProcessorBase):
    """
    Default Pre-processor of the downloaded data
     - it write a dataset with:
       - input (stack of 3 single-var fields)
       - ground truth (1 field of labelled image)
       - time (string of date/time of the most recent sample)
       - tc_class (int, makaniino class 1 to 6)
       - makaniino points (string of makaniino coords)
    """

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        # size of input tensor
        input_size = [self.field_size[0], self.field_size[1], 3]

        # ground truth size
        ground_truth_size = [self.field_size[0], self.field_size[1], 1]

        # Schema of the database to be created
        # b this pre-processor
        self.record_schema = RecordSchema(
            [
                EntrySchema(
                    {
                        "name": "train",
                        "shape": (0, input_size[0], input_size[1], input_size[2]),
                        "chunks": (1, input_size[0], input_size[1], input_size[2]),
                        "dtype": DatasetTypes.NP_FLOAT(),
                    }
                ),
                EntrySchema(
                    {
                        "name": "test",
                        "shape": (
                            0,
                            ground_truth_size[0],
                            ground_truth_size[1],
                            ground_truth_size[2],
                        ),
                        "chunks": (
                            1,
                            ground_truth_size[0],
                            ground_truth_size[1],
                            ground_truth_size[2],
                        ),
                        "dtype": DatasetTypes.NP_FLOAT(),
                    }
                ),
                EntrySchema(
                    {
                        "name": "time",
                        "shape": (0, 1),
                        "chunks": (1, 1),
                        "dtype": DatasetTypes.NP_STR(),
                    }
                ),
                EntrySchema(
                    {
                        "name": "tc_class",
                        "shape": (0, 1),
                        "chunks": (1, 1),
                        "dtype": DatasetTypes.NP_STR(),
                    }
                ),
                EntrySchema(
                    {
                        "name": "points",
                        "shape": (0, 1),
                        "chunks": (1, 1),
                        "dtype": DatasetTypes.NP_STR(),
                    }
                ),
            ]
        )

        # some input stacking params
        self.steps = 3
        self.hours = self.tracks_source.delta_hours
        self.useOnlyLabeled = True

    def _process_data_chunk(self, validTime):
        """
        The actual pre-processing routine
        Args:
            validTime:

        Returns:

        """

        logger.info("Processing time: %s", validTime)
        logger.debug("steps: %s", self.steps)
        finalTrain = None
        finalTest = None
        points = None
        tc_classes = None

        failed = False

        # goes backwards
        for i in range(self.steps - 1, -1, -1):
            logger.debug("processing step: %s", i)
            offset = (i) * self.hours
            timestamp = validTime - timedelta(hours=offset)

            # we process now timestamp
            needPoints = False
            if i == 0 and self.useOnlyLabeled:
                needPoints = True

            # this should be the routine that process a single field...
            train, test, _, points, tc_classes = self._processTime(
                timestamp, needPoints=needPoints
            )

            if train is None:
                failed = True
                break

            if needPoints and test is None:
                logger.warning(" no labels found ")
                failed = True
                break

            if finalTrain is None:
                finalTrain = np.zeros(
                    (1, train.shape[1], train.shape[2], self.steps), dtype=train.dtype
                )

            index = self.steps - i - 1
            finalTrain[0, :, :, index] = train[0, :, :, 0]

            if i == 0:
                finalTest = test
                logger.info(f"Added sample {timestamp} -> " f"cyclones {points}")

        # pack time
        time_np = time_to_numpy(validTime)

        if failed:

            logger.warning("no labels found")
            return {
                "train": None,
                "test": None,
                "time": time_np,
                "points": None,
                "tc_class": None,
            }
        else:

            logger.debug("returning data")

            # pack lat/lon
            points_np = numpy_to_npstring(points)

            # pack tc class
            tc_classes_np = numpy_to_npstring(tc_classes)

            logger.debug(f" POINTS {points}")
            logger.debug(f" CLASSES {tc_classes}")

            return {
                "train": finalTrain,
                "test": finalTest,
                "time": time_np,
                "points": points_np,
                "tc_class": tc_classes_np,
            }

    def tc_class_from_wind(self, wind):
        """
        Determing TC class
        Args:
            wind:

        Returns:

        """

        tc_class = 0

        if 34 < wind <= 64:
            tc_class = 1
        elif 64 < wind <= 82:
            tc_class = 2
        elif 82 < wind <= 96:
            tc_class = 3
        elif 96 < wind <= 113:
            tc_class = 4
        elif 113 < wind <= 137:
            tc_class = 5
        elif wind > 137:
            tc_class = 6

        return tc_class

    def getPointsAndClassForTime(self, data, timestep):
        """
        Get TC: lat, lon, class
        Args:
            data:
            timestep:

        Returns:

        """
        logger.debug("looking for: %s", timestep)
        dayLabels = data[(data.ISO_TIME == timestep)]
        logger.debug(dayLabels)

        points = []
        tc_classes = []

        for index, row in dayLabels.iterrows():
            logger.debug(row)
            lat = row["LAT"]
            lon = row["LON"]
            # roci = row['roci']
            # name = row['NAME']
            wind = row["wind"]

            points.append([lat, lon])
            tc_classes.append(self.tc_class_from_wind(wind))

        logger.info("found : %s points", len(points))

        return points, tc_classes

    def midpoint(self, x1, x2, y1, y2):
        # Input values as degrees

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

    def getPointsAndClassForIntermediateTime(self, data, validTime, timediff=360):
        """
        Get TC: lat, lon, class
        Args:
            data:
            validTime:
            timediff:

        Returns:

        """
        timestamp = validTime.strftime("%Y-%m-%d %H:%M:00")

        logger.info("looking for : %s", timestamp)
        dayLabels = data[(data.ISO_TIME == timestamp)]
        previousTime = validTime - timedelta(minutes=timediff)
        logger.info("previous time : %s", previousTime)

        previousstamp = previousTime.strftime("%Y-%m-%d %H:%M:00")
        previousLabels = data[(data.ISO_TIME == previousstamp)]

        points = []
        tc_classes = []

        for index, row in dayLabels.iterrows():

            sn = row["SID"]
            wind = row["wind"]
            prev = previousLabels.loc[previousLabels.SID == sn]

            if len(prev) > 0:
                prevrow = prev.iloc[0]

                # print (prevrow)
                curLat = row["LAT"]
                curLon = row["LON"]

                # print (row)
                prevLat = prevrow["LAT"]
                prevLon = prevrow["LON"]

                lat, lon = self.midpoint(prevLat, prevLon, curLat, curLon)

                points.append([lat, lon])
                tc_classes.append(self.tc_class_from_wind(wind))

        logger.debug("found : %s points", len(points))

        return points, tc_classes

    # this is the routine that processes a single timestep
    def _processTime(self, timestep, needPoints=False):

        logger.debug(
            "Processing timestep: %s %s needpoints: %s",
            timestep,
            timestep.hour,
            needPoints,
        )

        # get the appropriate time-points from the ibtracks file
        if timestep.hour % 6 == 3:
            logger.debug("Determining intermediate points")

            dt = timestep + timedelta(minutes=180)
            points, tc_classes = self.getPointsAndClassForIntermediateTime(
                self.ibtracks_df, dt, timediff=360
            )
        else:
            t = timestep.strftime("%Y-%m-%d %H:%M:00")
            points, tc_classes = self.getPointsAndClassForTime(self.ibtracks_df, t)

        logger.info(points)
        logger.debug(f"Processing cyclones \n{points}")

        if needPoints and points is None or len(points) == 0:
            logger.warning("need points and none were found")
            return None, None, timestep, None, None

        train, test = self._process_xarr_at_time(
            timestep, points, tc_classes, needPoints=needPoints
        )

        return train, test, timestep, points, tc_classes

    def _process_xarr_at_time(self, timestep, points, tc_classes, needPoints=False):
        """
        Interrogate the xarray and return the proper train and test numpy's
        """

        train = None
        test = None

        ts = timestep.strftime("%Y-%m-%dT%H:%M:%S")

        logger.debug(f"Time: {ts}")
        tidx = np.datetime64(ts, "ns")

        logger.debug(f"attempting to extract data for time {tidx}")

        try:
            data = getattr(self.source_data, self.var_name).loc[ts, :, :].values
        except KeyError:
            logger.debug(f"Time: {ts} NOT FOUND in Dataset!")

            return train, test

        lat_size = self.source_data.latitude.values.size
        lon_size = self.source_data.longitude.values.size
        lats = np.tile(self.source_data.latitude.values.reshape(-1, 1), (1, lon_size))

        lons = np.tile(self.source_data.longitude.values, (lat_size, 1))

        if data is not None:
            lons = lons - 180.0
            height, width = data.shape

            try:
                xypoints = pointsToDataPoints(
                    points, lats, lons, offsetX=True, width=width
                )

                # count = count + 1
                logger.debug("found %s points in region", len(xypoints))
                if not self.useOnlyLabeled or (
                    self.useOnlyLabeled and needPoints and len(xypoints) > 0
                ):
                    labels = []

                    # created labeled data same size as
                    # original for segmentation
                    # self.labelling_method ["square", "cone", "sine"]
                    createLabeledData(
                        data,
                        xypoints,
                        labels,
                        range_value=255,
                        center_mark=self.center_mark,
                        method=self.labelling_method,
                        tc_classes=tc_classes,
                    )

                    logger.debug(labels)

                    # recast label field according to method
                    if self.labelling_method == "square":
                        labels = np.array(labels, dtype=np.uint8)
                    else:
                        labels = np.array(labels, dtype=np.float)

                    # format array for appending others in file
                    tmpdata = np.array(data)
                    tmpdata = tmpdata[np.newaxis, :, :, np.newaxis]

                    tmptest = np.array(labels)
                    tmptest = tmptest[:, :, :, np.newaxis]

                    train = tmpdata
                    test = labels[:, :, :, np.newaxis]
                else:
                    tmpdata = np.array(data)

                    tmpdata = tmpdata[np.newaxis, :, :, np.newaxis]

                    train = tmpdata
                    test = None

            except Exception as e:

                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                logger.error(exc_type, fname, exc_tb.tb_lineno)
                logger.error(e)

            del data
            del tmpdata

        return train, test
