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

import mpi4py.MPI as MPI
import numpy as np

from makaniino.data_handling.generators.base import DataGeneratorBase
from makaniino.data_handling.generators.data_cache import DataCache
from makaniino.utils.generic_utils import date_idx_in_zarr

logger = logging.getLogger(__name__)


class DataGeneratorMultiDS(DataGeneratorBase):
    """
    Serves and Handles data from multiple datasets
    """

    def __init__(
        self,
        data_path="",
        batch_size=1,
        max_n_batches=None,
        single_node_batch_read=False,
        cached_data=DataCache(),
        verbose=False,
    ):

        super().__init__(
            data_path=data_path,
            batch_size=batch_size,
            max_n_batches=max_n_batches,
            single_node_batch_read=single_node_batch_read,
            cached_data=cached_data,
            verbose=verbose,
        )

        # column-separated list of paths to all datasets
        # that the generator uses to serve data
        self.data_paths = self.extract_paths_and_dates(self.data_path)

        # these quantities are set at setup
        self.all_data = {}

        # maps a global index to a data path and local idx
        self.gidx_to_local_info = {}

        self.train_sample_inner_shape = None
        self.test_sample_inner_shape = None

    def setup(self):
        """
        Setup the data to be served
        """

        # data paths
        for path in self.data_paths.keys():
            if not os.path.exists(path):
                logger.error(f"Path {path} does not exists!")

        # prepare the internal data structures
        self.all_data = self.read_dataset(self.data_paths, self.served_variables)
        self.train_sample_inner_shape = self.all_data[list(self.all_data.keys())[0]][
            "train"
        ].shape[1:]
        self.test_sample_inner_shape = self.all_data[list(self.all_data.keys())[0]][
            "test"
        ].shape[1:]

        # the length is the total length of data from all the datasets
        gidx = 0  # global index

        for ds_path in self.all_data:

            start_date = self.data_paths[ds_path]["from"]
            end_date = self.data_paths[ds_path]["to"]

            valid_indexes = date_idx_in_zarr(
                self.all_data[ds_path]["time"], (start_date, end_date)
            )
            data_len = len(valid_indexes)

            # fill up the map between idx and data path
            idx_datapath_ = {
                ii + gidx: {"path": ds_path, "idx": idx}
                for ii, idx in enumerate(valid_indexes)
            }

            self.gidx_to_local_info.update(idx_datapath_)
            gidx += data_len

        # vector containing all the sample indexes
        self.indexes = list(self.gidx_to_local_info.keys())

        # By default no shuffling is involved
        self.set_order(self.indexes)

        # generator has now been setup
        self._is_gen_setup = True

    @staticmethod
    def extract_paths_and_dates(paths_string):

        # Each path is of type
        # path/to/dataset:<date_start>:<date_end>
        data_paths = {}
        for ii, path_info_string in enumerate(paths_string.split(";")):

            path_info = path_info_string.split(":")
            assert len(path_info) <= 3

            path = path_info[0]

            start_date = path_info[1] if len(path_info) > 1 else ""
            end_date = path_info[2] if len(path_info) > 2 else ""

            assert path not in list(data_paths.keys())

            data_paths.update({path: {"from": start_date, "to": end_date}})

        return data_paths

    @staticmethod
    def read_dataset(paths, served_variables):
        """
        Read arrays from zarr archive
        """

        raise NotImplementedError

    def __getitem__(self, idx):

        if not self._is_gen_setup:
            logger.debug("Generator has not been setup. Doing it now..")
            self.setup()

        if self._cached_data.has_idx(idx):
            train, test = self._cached_data.get_data(idx)
        else:

            cut_start = idx * self.batch_size
            cut_end = (idx + 1) * self.batch_size
            indices = self._order[cut_start:cut_end]

            if self.verbose:
                logger.info(
                    f"Generator on process {os.getpid()} is requested indices\n{indices}"
                )

            if self.single_node_batch_read:
                train, test = self.read_and_bcast_samples(indices)
            else:
                train, test = self.read_samples(indices)

            # apply all the necessary pre-process actions
            for action in self.online_processors:
                train, test = action.apply(train, test)

        return train, test

    def read_samples(self, indices):
        """
        Read the requested indexes from the Dataset
        """

        assert len(indices) > 0, f"indices: {indices}"

        # ////////////////////////// debug only.. ///////////////////////////////
        # train = np.random.rand(self.batch_size, *self.train_sample_inner_shape)
        # test = np.random.rand(self.batch_size, *self.test_sample_inner_shape)
        # return train, test
        # ///////////////////////////////////////////////////////////////////////

        idx0 = indices[0]

        ds_path = self.gidx_to_local_info[idx0]["path"]
        ds_local_idx = self.gidx_to_local_info[idx0]["idx"]
        data_ = self.all_data[ds_path]

        train = np.array(data_["train"].oindex[ds_local_idx])
        assert (
            train.shape == self.train_sample_inner_shape
        ), f"train.shape: {train.shape}"
        train = np.expand_dims(train, axis=0)

        test = np.array(data_["test"].oindex[ds_local_idx])
        assert test.shape == self.test_sample_inner_shape, f"train.shape: {train.shape}"
        test = np.expand_dims(test, axis=0)

        for isample in indices[1:]:

            # take ds_path and local idx of this sample
            try:
                ds_path = self.gidx_to_local_info[isample]["path"]
            except KeyError:
                raise KeyError(
                    f"KEY {isample} NOT found, available keys {self.gidx_to_local_info.keys()}"
                )

            ds_local_idx = self.gidx_to_local_info[isample]["idx"]

            # data relative to the sample
            data_ = self.all_data[ds_path]

            sample_x = data_["train"].oindex[ds_local_idx]
            assert (
                sample_x.shape == self.train_sample_inner_shape
            ), f"sample_x.shape: {sample_x.shape}"
            sample_x = np.expand_dims(sample_x, axis=0)

            sample_y = data_["test"].oindex[ds_local_idx]
            assert (
                sample_y.shape == self.test_sample_inner_shape
            ), f"sample_y.shape: {sample_y.shape}"
            sample_y = np.expand_dims(sample_y, axis=0)

            train = np.concatenate((train, sample_x), axis=0)
            test = np.concatenate((test, sample_y), axis=0)

        return train, test

    def read_and_bcast_samples(self, indices):

        comm = MPI.COMM_WORLD
        rank = comm.rank

        # send len of indices to rank-0
        if rank != 0:

            # train_tmp, test_tmp = self.read_samples(indices)
            # logger.info(f"RANK {rank} needs indexes:{indices}, expecting train like:\n{train_tmp[0,:3,:3,0]}")

            # send the requested indexes to rank 0
            comm.send(indices, dest=0, tag=1)

            # make space for samples
            train = np.empty((len(indices),) + self.train_sample_inner_shape)
            test = np.empty((len(indices),) + self.test_sample_inner_shape)

            # receive samples
            comm.Recv(train, source=0, tag=2)
            comm.Recv(test, source=0, tag=3)

        else:  # rank 0

            # read data as requested by the other ranks
            for irank in range(comm.size)[1:]:

                # requested sample indexes to read
                irank_indices = comm.recv(source=irank, tag=1)

                # read data
                irank_train, irank_test = self.read_samples(irank_indices)

                # logger.info(f"R0 instructed to get indexes:{irank_indices} for rank {irank}... "
                #             f"OK, sending back irank_train\n{irank_train[0,:3,:3,0]}")

                # send data back to requesting rank
                comm.Send(irank_train, dest=irank, tag=2)
                comm.Send(irank_test, dest=irank, tag=3)

            # process its own (rank 0) samples
            train, test = self.read_samples(indices)

        return train, test

    def reset_data_path(self, path):
        self.data_path = path
        self.data_paths = self.extract_paths_and_dates(self.data_path)
