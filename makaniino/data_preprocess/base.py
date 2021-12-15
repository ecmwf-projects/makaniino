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
import multiprocessing
import os
from datetime import timedelta

import dateutil.parser
import numpy as np
import pandas as pd

from makaniino.data_handling.datasets.zarr import CycloneDatasetZARR
from makaniino.tracks_sources.tracks_factory import tracks_factory

logger = logging.getLogger(__name__)


class PreProcessorBase:
    """
    Pre-process the source data
    """

    # dataset type (set to ZARR Dataset for now..)
    dataset_class = CycloneDatasetZARR

    def __init__(
        self,
        source_data,
        output_dir,
        include_latlon=False,
        tracks_source_tag="ibtracks",
        ibtracks_path="./../data/ibtracs.last3years.list.v04r00.csv",
        process_ibtracks=False,
        center_mark=50,
        labelling_method="square",
        n_procs=None,
        chunk_size=1,
        field_size=(361, 720),
    ):

        # args
        self.source_data = source_data
        self.out_dir = output_dir
        self.tracks_source_tag = tracks_source_tag
        self.tracks_path = ibtracks_path
        self.process_ibtracks = process_ibtracks
        self.center_mark = center_mark
        self.labelling_method = labelling_method
        self.n_procs = n_procs
        self.chunk_size = chunk_size
        self.field_size = field_size

        # include lat/lon if so required
        self.include_latlon = include_latlon

        # figure out variable names
        self.var_name = source_data.open_variables

        # Schema of the database created
        self.record_schema = None

        # cleaned version of the file
        if self.tracks_path.split("."):
            self.ib_fname_clean = (
                ".".join(self.tracks_path.split(".")[:-1]) + "_cleaned.csv"
            )
        else:
            self.ib_fname_clean = self.tracks_path + "_cleaned.csv"

        # load the data
        self.tracks_source = None
        self.load_tracks()

        # lat lon fields (calc once at the
        # beginning to be more efficient
        latitudes = self.source_data.get_latitudes()
        longitudes = self.source_data.get_longitudes()

        lat_size = latitudes.size
        lon_size = longitudes.size

        lats = np.tile(latitudes.reshape(-1, 1), (1, lon_size))
        lons = np.tile(longitudes, (lat_size, 1))
        lons = lons - 180.0

        self.lats = lats
        self.lons = lons

    def load_tracks(self):

        logger.debug("Reading track file %s", self.tracks_path)
        trk_cls = tracks_factory[self.tracks_source_tag]

        if self.process_ibtracks:
            self.tracks_source = trk_cls.from_file(self.tracks_path)
        else:
            self.tracks_source = trk_cls(pd.read_csv(self.tracks_path))

        # save a clean version of the tracks if it does not exist..
        if self.process_ibtracks and not os.path.exists(self.ib_fname_clean):

            logger.debug(
                f"Saving cleaned version " f"of tracks to {self.ib_fname_clean}"
            )

            self.tracks_source.to_csv(self.ib_fname_clean, index=None, header=True)

    def run(self, start_str, end_str):
        """
        The driving routine that distributes chunks of date/times
        to the processors in the pool for the pre-processing
        that each child class will then implement (in _process_data_chunk)..
        Args:
            start_str:
            end_str:

        Returns:

        """

        date_start = dateutil.parser.parse(start_str)
        date_end = dateutil.parser.parse(end_str)

        start = date_start.strftime("%Y-%m-%d")
        end = date_end.strftime("%Y-%m-%d")

        # Select N of processors as required
        if self.n_procs:
            procs = max(1, self.n_procs)
        else:
            procs = max(1, multiprocessing.cpu_count() - 1)

        logger.info(" Processors Available: %s", multiprocessing.cpu_count())
        logger.info(" Processors Requested: %s", procs)
        logger.info(" Map chunk size: %s", self.chunk_size)

        # create the pool of workers
        pool = multiprocessing.Pool(processes=procs)

        timestamp = start
        validTime = dateutil.parser.parse(timestamp)

        endstamp = end
        endTime = dateutil.parser.parse(endstamp)

        # list of times to distribute
        # to workers for processing
        args = []
        while validTime < endTime:
            args.append(validTime)
            validTime = validTime + timedelta(hours=3)

        logger.debug(" Run start date: %s", validTime)
        logger.debug(" Run valid times: %s", args)

        # NOTE substituted processPair function with _process_data_chunk
        n_samples = 0
        for result in pool.imap(self._process_data_chunk, args, self.chunk_size):

            # if tensors in the results are all valid:
            if all(tsr is not None for tsr in result.values()):

                filename = self.out_dir

                logger.debug("outfile: %s", filename)

                if self.dataset_class.exists_in_path(filename):

                    dataset = self.dataset_class.from_path(filename)
                    dataset.open()

                else:

                    dataset = self.dataset_class(filename, self.record_schema)

                # append the record..
                dataset.insert(
                    {
                        tsr_name["name"]: result[tsr_name["name"]]
                        for tsr_name in self.record_schema.to_list()
                    }
                )

                n_samples += 1

            else:
                logger.debug("No data")

        pool.close()
        pool.join()

        logger.info(f"Written {n_samples} samples")

    def _process_data_chunk(self, validTime):
        """
        Routine that actually process each chunk of data..
        Args:
            validTime:

        Returns:
            a dictionary {
              <tensor-name>: <tensor-value>
              ...
            }
            where the keys are the ones described in
            record_schema "names"

        """
        raise NotImplementedError

    def _get_xr_values_at_time(self, var_name, ts):
        """
        Extract values from the xarray
        """

        if var_name == "lat":
            return self.lats

        if var_name == "lon":
            return self.lons

        # check the name map
        logger.info(f"processing variable {var_name}")

        # and try to get the data required
        try:
            data = self.source_data.get_variable_at_time(var_name, ts)
        except KeyError:
            logger.warning(f"Variable {var_name} and Time: {ts} NOT FOUND in Dataset!")
            return None

        return data
