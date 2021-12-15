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
import os
import sys
from datetime import timedelta

import numpy as np

from makaniino.data_handling.datasets.base import DatasetTypes, EntrySchema, RecordSchema
from makaniino.data_preprocess.base import PreProcessorBase
from makaniino.label_data import createLabeledData, pointsToDataPoints
from makaniino.utils.generic_utils import numpy_to_npstring, time_to_numpy

logger = logging.getLogger(__name__)


class PreProcessorDefault(PreProcessorBase):
    """
    Default Pre-processor of the downloaded data
     - it write a dataset with:
       - input (stack of 3 single-var fields)
       - ground truth (1 field of labelled image)
       - time (string of date/time of the most recent sample)
       - makaniino points (string of makaniino coords)
    """

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        # some input stacking params
        self.steps = 3
        self.hours = self.tracks_source.delta_hours

        self.useOnlyLabeled = True

        field_size = self.source_data.get_field_size()

        # depth of input samples
        input_depth = len(self.var_name) * self.steps + int(self.include_latlon) * 2

        # size of input tensor
        input_size = [field_size[0], field_size[1], input_depth]

        # ground truth size
        ground_truth_size = [field_size[0], field_size[1], 1]

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
                        "name": "points",
                        "shape": (0, 1),
                        "chunks": (1, 1),
                        "dtype": DatasetTypes.NP_STR(),
                    }
                ),
            ]
        )

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

        failed = False

        # goes backwards
        for i in range(self.steps - 1, -1, -1):

            logger.debug("processing step: %s", i)

            offset = i * self.hours
            timestamp = validTime - timedelta(hours=offset)

            # we process now timestamp
            # we only require points for the last step
            needPoints = False
            if i == 0 and self.useOnlyLabeled:
                needPoints = True

            # this should be the routine that
            # process a single field...
            train, test, _, points = self._processTime(timestamp, needPoints=needPoints)

            if train is None:
                failed = True
                break

            if needPoints and test is None:
                logger.warning(" no labels found ")
                failed = True
                break

            if finalTrain is None:
                finalTrain = np.zeros(
                    (1, train.shape[1], train.shape[2], 0), dtype=train.dtype
                )

            # index = self.steps - i - 1
            # finalTrain[0, :, :, index] = train[0, :, :, 0]
            finalTrain = np.concatenate((finalTrain, train), axis=-1)

            if i == 0:
                finalTest = test
                logger.info(f"Added sample {timestamp} -> " f"cyclones {points}")

        # add lat/lon fields in the input stack, if required..
        if not failed and self.include_latlon:

            lats_ext = self.lats[np.newaxis, :, :, np.newaxis]
            lons_ext = self.lons[np.newaxis, :, :, np.newaxis]

            finalTrain = np.concatenate((finalTrain, lats_ext), axis=-1)
            finalTrain = np.concatenate((finalTrain, lons_ext), axis=-1)

        # prepare the time and point fields to store
        time_np = time_to_numpy(validTime)
        points_np = numpy_to_npstring(points)

        # return the data if not failed
        if failed:
            logger.warning("No labels found")
            return {"train": None, "test": None, "time": time_np, "points": points_np}
        else:
            logger.debug("Returning data..")
            return {
                "train": finalTrain,
                "test": finalTest,
                "time": time_np,
                "points": points_np,
            }

    # this is the routine that processes a single timestep
    def _processTime(self, timestep, needPoints=False):

        logger.debug(
            "Processing timestep: %s %s needpoints: %s",
            timestep,
            timestep.hour,
            needPoints,
        )

        # get the makaniino coordinates at time = timestep
        points = self.tracks_source.get_cyclone_coords_at_time(timestep)

        logger.info(f"Processing makaniino points: \n{points}")

        if needPoints and points is None or len(points) == 0:
            logger.warning("need points and none were found")
            return None, None, timestep, None

        train, test = self._process_xarr_at_time(
            timestep, points, needPoints=needPoints
        )

        return train, test, timestep, points

    def _process_xarr_at_time(self, timestep, points, needPoints=False):
        """
        Interrogate the xarray and return the
        proper train and test numpy's
        """

        train = None
        test = None

        train_all = None
        test_all = None

        ts = timestep.strftime("%Y-%m-%dT%H:%M:%S")

        logger.debug(f"Time: {ts}")
        tidx = np.datetime64(ts, "ns")

        logger.debug(f"attempting to extract data for time {tidx}")

        logger.info(f"ENTERING LOOP: self.var_names {self.var_name}")
        for var_name in self.var_name:

            logger.info(f"iteration: var_name: {var_name}")
            data = self._get_xr_values_at_time(var_name, ts)

            if data is None:
                return None, None
            else:

                height, width = data.shape

                # initialize the field-stack (here because we need to know data.shape)
                if train_all is None:
                    train_all = np.zeros((1, height, width, 0), dtype=data.dtype)
                    test_all = np.zeros((1, height, width, 0), dtype=data.dtype)

                try:
                    xypoints = pointsToDataPoints(
                        points, self.lats, self.lons, offsetX=True, width=width
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

                        # print(f"train.shape {train.shape}")
                        # print(f"test.shape {test.shape}")
                        # print(f"train_all.shape {train_all.shape}")
                        # print(f"test_all.shape {test_all.shape}")

                        train_all = np.concatenate((train_all, train), axis=-1)
                        test_all = np.concatenate((test_all, test), axis=-1)

                    else:
                        tmpdata = np.array(data)
                        tmpdata = tmpdata[np.newaxis, :, :, np.newaxis]

                        train = tmpdata
                        test = None

                        # stack only the train data..
                        train_all = np.concatenate((train_all, train), axis=-1)
                        test_all = None

                except Exception as e:

                    exc_type, exc_obj, exc_tb = sys.exc_info()
                    _fn = exc_tb.tb_frame.f_code.co_filename
                    fname = os.path.split(_fn)[1]
                    logger.error(exc_type, fname, exc_tb.tb_lineno)
                    logger.error(e)

                del data
                del tmpdata

        # print("tensor stack done!")
        return train_all, test
