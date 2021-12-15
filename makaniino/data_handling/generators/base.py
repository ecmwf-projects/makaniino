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

import numpy as np
from tensorflow.keras.utils import Sequence

from makaniino.data_handling.generators.data_cache import DataCache

logger = logging.getLogger(__name__)


class DataGeneratorBase(Sequence):
    """
    Serves and Handles data
    """

    served_variables = ["train", "test", "time"]

    def __init__(
        self,
        data_path="",
        batch_size=1,
        max_n_batches=None,
        single_node_batch_read=False,
        cached_data=DataCache(),
        verbose=False,
    ):

        # Comma-separated string of paths to datasets
        self.data_path = data_path

        assert batch_size > 0
        self.batch_size = batch_size

        # if only rank 0 reads and sends samples
        self.single_node_batch_read = single_node_batch_read

        # max N of batches to serve
        self.max_n_batches = max_n_batches

        # data cache
        self._cached_data = cached_data

        # length of all sample indexes
        self.length = None

        # all sample indexes
        self.indexes = None

        # order of sample indexes
        # (can be different from indexes if samples are shuffled)
        self._order = None

        # is generator setup
        self._is_gen_setup = False

        # series of pre-processing actions
        # to perform on each batch before serving
        self.online_processors = []

        # verbose flag
        self.verbose = verbose

    def setup(self):
        """
        Setup the data to be served
        """

        raise NotImplementedError

    def reset_data_path(self, path):
        self.data_path = path

    def __len__(self):
        """
        Returns max number of *batches* that can be served.
        It depends on:
          - self.length (the length of self.order - that determines the global order of samples considered)
          - self.max_n_batches (user choice of how many batches can be served at maximum)
        """

        if not self._is_gen_setup:
            logger.info("Generator has not been setup. Doing it now..")
            self.setup()

        # how many batches we serve
        if self.max_n_batches:
            return min(self.length // self.batch_size, self.max_n_batches)
        else:
            return self.length // self.batch_size

    def set_online_processors(self, actions):
        """
        Set a list of pre-process actions
        """
        self.online_processors = actions

    def add_process_action(self, action):
        """
        Add a pre-process action
        """
        self.online_processors.append(action)

    @staticmethod
    def read_dataset(paths, served_variables):
        """
        Read arrays from zarr archive
        """
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError

    def summary(self):
        """
        A summary of this generator
        """

        if not self._is_gen_setup:
            logger.info("Generator has not been setup. Doing it now..")
            self.setup()

        _str = "\n"
        _str += f"Data path: {self.data_path}\n"
        _str += f"N of samples: {self.length}\n"
        _str += f"Batch size: {self.batch_size}\n"
        _str += f"N of batches: {self.length // self.batch_size}\n"

        return _str

    @property
    def is_gen_setup(self):
        return self._is_gen_setup

    def set_order(self, order):
        """
        Set sample order (and update length)
        """
        self._order = np.asarray(order).astype(np.int64)
        self.length = len(self._order)

        try:
            assert self.length > 0
        except AssertionError:
            raise AssertionError(
                f"Generator seems to have negative length {self.length}"
            )

    def set_cached_data(self, cached_data):
        """
        Set cached data
        """
        self._cached_data = cached_data
