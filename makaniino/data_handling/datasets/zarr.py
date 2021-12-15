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

import os
from collections import OrderedDict

import numpy as np
import zarr

from makaniino.data_handling.datasets.base import CycloneDataset, DatasetTypes, logger


class CycloneDatasetZARR(CycloneDataset):
    """
    Dataset stored into a zarr archive
    """

    # zarr config
    compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=2)
    synchronizer = zarr.ProcessSynchronizer("example.sync")

    def __init__(self, path, record_schema=None):

        super().__init__(path, record_schema=record_schema)

        self._raw_data = {}

        # schema is set only at database creation,
        if record_schema:

            _store = zarr.DirectoryStore(self.ds_rawdata_path)
            _base = zarr.group(_store, overwrite=True, synchronizer=self.synchronizer)

            # Specific setup of the ZARR dataset
            for entry_schema in self.record_schema.to_list():

                self._raw_data[entry_schema.get("name")] = _base.create_dataset(
                    entry_schema.get("name"),
                    shape=entry_schema.get("shape"),
                    chunks=entry_schema.get("chunks"),
                    dtype=DatasetTypes.map[entry_schema.get("dtype")],
                    compressor=self.compressor,
                )

    def _open(self, *args, **kwargs):
        """
        Do all that is necessary to open the zarr archive
        and set the DS length
        """

        # data path
        if not os.path.exists(self.ds_path):
            logger.error(f"Path {self.ds_path} does not exists!")

        # open zarr archive
        self._raw_data = zarr.open(self.ds_rawdata_path, **kwargs)

        # read the database length (from the first key)
        first_key = next(iter(self._raw_data))
        self.length = len(self._raw_data[first_key])

    def __getitem__(self, idx):

        if not self.isopen:
            logger.debug("Dataset is not open! Doing it now..")
            self.open()

        # start/end indices
        cut_start = idx * self._batch_size
        cut_end = (idx + 1) * self._batch_size

        # slice
        indices = self.order[cut_start:cut_end]

        # pack the output variables to serve (with the right order)
        output_dict = OrderedDict(
            (var, np.array(self._raw_data[var].oindex[indices]))
            for var in self._served_variables
        )

        # apply all the necessary pre-process actions
        for action in self._online_processors:
            output_dict["train"], output_dict["test"] = action.apply(
                output_dict["train"], output_dict["test"]
            )

        # return a list
        output = output_dict.values()

        return output

    def _do_insert(self, record):
        """
        Append a dataset record
        Args:
            record:

        Returns:

        """

        # open database for appending
        storage = zarr.open(self.ds_rawdata_path)

        # append all the entries in the record
        for k, v in record.items():
            storage[k].append(v)

    def _close(self):
        """
        Nothing to close for ZARR ds
        """
        pass

    def __str__(self):
        """
        Brief description
        """

        return (
            f"Dataset serving {len(self)} "
            f"batches of {self._batch_size} "
            f"from path {self.ds_path}"
        )

    def write_at(self, record, idx, remove_first_axis=True):

        # open database for appending
        storage = zarr.open(self.ds_rawdata_path)

        # append all the entries in the record

        # remove the first axis..
        if remove_first_axis:
            for k, v in record.items():
                storage[k][idx] = v[0, :]
        else:
            for k, v in record.items():
                storage[k][idx] = v

    def _do_resize(self, new_length):

        # All the arrays of the dataset
        for entry_schema in self.record_schema.to_list():

            # current shape
            current_shape = self._raw_data[entry_schema.get("name")].shape

            print(f"{entry_schema.get('name')}: current_shape {current_shape}")

            self._raw_data[entry_schema.get("name")].resize(
                new_length, *current_shape[1:]
            )

            print(
                f"{entry_schema.get('name')}: new_shape {self._raw_data[entry_schema.get('name')].shape}"
            )
