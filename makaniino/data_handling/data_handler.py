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

from makaniino.configurable import Configurable
from makaniino.data_handling.generators.data_cache import DataCache
from makaniino.data_handling.generators.shuffling import (
    DistributedShuffler,
    shuffling_factory,
)
from makaniino.data_handling.generators.zarr import DataGenerator_ZARR
from makaniino.data_handling.online_processing import (
    BatchShuffler,
    DataNormalizer,
    DataRecaster,
    EdgeTrimmer,
    NANSwapper,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class DataHandler(Configurable):
    """
    Configures a Generator for the HRES data
    """

    default_params = {
        "train_data_path": (
            "/var/tmp/cyclone_data_train/",
            "Path of the train zarr archive",
        ),
        "val_data_path": (
            "/var/tmp/cyclone_data_val/",
            "Path of the train zarr archive",
        ),
        "test_data_path": (
            "/var/tmp/cyclone_data_test/",
            "Path of the train zarr archive",
        ),
        "batch_size": (2, "Batch size"),
        "single_node_batch_read": (
            0,
            "Rank 0 reads each batch and sends it to the correct node",
        ),
        "x_cut_pxl": (40, "N of pixel to cutoff from the x-dim"),
        "y_cut_pxl": (16, "N of pixel to cutoff from the y-dim"),
        "norm_factor": ("55", "Normalization factor", str),
        "shift_factor": ("55", "Normalization-shift factor", str),
        "dataset_output_type": ("float", "force-cast output type"),
        "verbose_handler": (0, "Verbose Data handler"),
        "shuffle_train_data": ("sample", "Shuffle indexes of the training generator"),
        "shuffle_valid_data": ("sample", "Shuffle indexes of the validation generator"),
        "shuffle_test_data": ("sample", "Shuffle indexes of the testing generator"),
        "cache_data": (0, "Store data in memory"),
    }

    _out_type = {"float": np.float32, "int": np.int}

    generator_class = None

    def __init__(self, config):

        self.params = config

        super().__init__(self.params)

        self._setup()

        # data shape (not known until the generators are setup
        self.input_data_shape = None
        self.output_data_shape = None

    def _setup(self):
        """
        Setup the raw generators
        """

        # training generator
        train_gen = self.generator_class(
            data_path=self.params.train_data_path,
            batch_size=self.params.batch_size,
            single_node_batch_read=self.params.single_node_batch_read,
            verbose=self.params.verbose_handler,
        )
        train_gen.set_online_processors(
            [
                EdgeTrimmer(
                    x_cut_pxl=self.params.x_cut_pxl, y_cut_pxl=self.params.y_cut_pxl
                ),
                DataNormalizer(
                    norm_factor=self.params.norm_factor,
                    shift_factor=self.params.shift_factor,
                ),
                DataRecaster(y_type=self._out_type[self.params.dataset_output_type]),
                NANSwapper(),
                BatchShuffler(),
            ]
        )

        # validation generator
        val_gen = self.generator_class(
            data_path=self.params.val_data_path, batch_size=self.params.batch_size
        )
        val_gen.set_online_processors(
            [
                EdgeTrimmer(
                    x_cut_pxl=self.params.x_cut_pxl, y_cut_pxl=self.params.y_cut_pxl
                ),
                DataNormalizer(
                    norm_factor=self.params.norm_factor,
                    shift_factor=self.params.shift_factor,
                ),
                DataRecaster(y_type=self._out_type[self.params.dataset_output_type]),
                NANSwapper(),
                BatchShuffler(),
            ]
        )

        # validation generator
        test_gen = self.generator_class(
            data_path=self.params.test_data_path, batch_size=self.params.batch_size
        )
        test_gen.set_online_processors(
            [
                EdgeTrimmer(
                    x_cut_pxl=self.params.x_cut_pxl, y_cut_pxl=self.params.y_cut_pxl
                ),
                DataNormalizer(
                    norm_factor=self.params.norm_factor,
                    shift_factor=self.params.shift_factor,
                ),
                DataRecaster(y_type=self._out_type[self.params.dataset_output_type]),
                NANSwapper(),
                BatchShuffler(),
            ]
        )

        self.generator_train = train_gen
        self.generator_val = val_gen
        self.generator_test = test_gen

    def reshuffle_train_data(self, seed=0):
        self._shuffle_data(
            self.generator_train, self.params.shuffle_train_data, seed=seed
        )

    def reshuffle_validation_data(self, seed=0):
        self._shuffle_data(
            self.generator_val, self.params.shuffle_train_data, seed=seed
        )

    def reshuffle_test_data(self, seed=0):
        self._shuffle_data(
            self.generator_test, self.params.shuffle_train_data, seed=seed
        )

    @staticmethod
    def _shuffle_data(generator, mode="sample", seed=0):
        """
        Shuffle data according to a shuffling strategy
        """

        indexes = generator.indexes
        batch_size = generator.batch_size

        shuffler = shuffling_factory[mode](indexes, batch_size)
        shuffled_order = shuffler.apply(seed=seed)
        generator.set_order(shuffled_order)

    def distributed_reshuffle_train_data(self, seed=0, populate_cache=False):
        self._distributed_shuffle_data(
            self.generator_train,
            self.params.shuffle_train_data,
            seed=seed,
            populate_cache=populate_cache,
        )

    def distributed_reshuffle_validation_data(self, seed=0, populate_cache=False):
        self._distributed_shuffle_data(
            self.generator_val,
            self.params.shuffle_train_data,
            seed=seed,
            populate_cache=populate_cache,
        )

    def distributed_reshuffle_test_data(self, seed=0, populate_cache=False):
        self._distributed_shuffle_data(
            self.generator_test,
            self.params.shuffle_train_data,
            seed=seed,
            populate_cache=populate_cache,
        )

    @staticmethod
    def _distributed_shuffle_data(
        generator, shuffle_mode="sample", seed=0, populate_cache=False
    ):
        """
        Shuffle data according to a shuffling strategy
        """

        indexes = generator.indexes
        batch_size = generator.batch_size

        shuffler = shuffling_factory[shuffle_mode](indexes, batch_size)
        distributed_shuffler = DistributedShuffler(shuffler, seed=seed)

        if populate_cache:

            logger.info("Caching data..")

            # temporarily set order of the generator as sorted,
            # to (possibly) speed up populating cache
            generator.set_order(distributed_shuffler.local_idxs_sorted)
            cache_data = DataCache()
            for idx in range(len(generator)):
                cache_data.add_data(generator[idx])
            generator.set_cached_data(cache_data)

        # set up shuffled order of idxs in the generator
        generator.set_order(distributed_shuffler.local_idxs_shuffled)

    def reset_data_path(self, path, data_type):
        """
        Reset data path of the generators
        """

        _gen_map = {
            "training": self.generator_train,
            "validation": self.generator_val,
            "testing": self.generator_test,
        }

        assert isinstance(data_type, str)
        assert data_type in _gen_map

        _gen_map[data_type].reset_data_path(path)

    def summary(self):

        _str = "\n *** Data Handler ***\n"
        _str += f"Training generator: {self.generator_train.summary()}\n"
        _str += f"Validation generator: {self.generator_val.summary()}\n"
        _str += f"Testing generator: {self.generator_test.summary()}\n"
        return _str


class DataHandler_ZARR(DataHandler):
    """
    Uses a ZARR generator
    """

    generator_class = DataGenerator_ZARR
