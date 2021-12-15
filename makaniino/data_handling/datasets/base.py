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

import json
import logging
import os

import numpy as np
from tensorflow.keras.utils import Sequence

logger = logging.getLogger(__name__)


class DatasetTypes:
    @classmethod
    def NP_FLOAT(cls):
        return "NP_FLOAT"

    @classmethod
    def NP_INT(cls):
        return "NP_INT"

    @classmethod
    def NP_STR(cls):
        return "NP_STR"

    map = {"NP_FLOAT": np.float, "NP_INT": np.int, "NP_STR": np.str}


class EntrySchema:
    """
    Schema of a record entry (each record
    is made of several entries..)
    """

    required_fields = ["name", "shape", "dtype"]

    def __init__(self, schema):
        self._schema = schema
        self._check_schema()

    def _check_schema(self):

        # check that dict is not empty
        assert self._schema

        # check that it is a dict
        assert isinstance(self._schema, dict)

        for field in self.required_fields:

            try:
                assert field in self._schema
            except AssertionError:
                logger.error(
                    f"field {field} not in present in schema, "
                    f"but is required by {self.__class__}!"
                )
                raise AssertionError

    def get(self, key):
        """
        Get a property of the entry schema
        Args:
            key:

        Returns:

        """
        return self._schema[key]

    def to_dict(self):
        """
        Give the dictionary back
        Returns:
        """

        return self._schema


class RecordSchema:
    """
    Schema for a dataset record
    """

    def __init__(self, entries):

        # just a list of record entries
        self._entries = entries

        self._check_entries()

    def _check_entries(self):
        """
        minimal check on entries
        Returns:
        """

        for e in self._entries:

            try:
                assert isinstance(e, EntrySchema)
            except AssertionError:
                logger.error("found invalid entry in record schema!")
                raise TypeError

    @classmethod
    def from_json(cls, filename):
        """
        Read the record schema from a json file
        Args:
            filename:

        Returns:

        """

        logger.debug(f"reading JSON file {filename}")
        with open(filename, "r") as f:
            entry_schemas = json.load(f)

        entries = [EntrySchema(sch) for sch in entry_schemas]

        return cls(entries)

    def get(self, key):
        return self._entries[key]

    def to_list(self):
        """
        Record Schema as a dictionary
        Returns:
        """

        return [e.to_dict() for e in self._entries]

    def entry_names(self):
        """
        Record list of schema names
        Returns:
        """

        return [e.to_dict()["name"] for e in self._entries]


class CycloneDataset(Sequence):
    """
    A Cyclone Dataset offers an interface to store and
    serve data. Options include: batch size, shuffle, etc..
    DS metadata are stored in a JSON format containing:
      - DS schema
      - DS length
      - Additional info
    """

    schema_json_file = "schema.json"
    metadata_json_file = "metadata.json"

    def __init__(self, ds_path, record_schema=None):

        # Dataset path
        self.ds_path = ds_path
        self.ds_rawdata_path = os.path.join(self.ds_path, "_data")

        # record schema
        self.record_schema = record_schema

        # store the schema file in the dataset path
        # if the schema is passed on (at database creation)
        if record_schema:
            assert isinstance(record_schema, RecordSchema)
            self._store_schema()

        # database metadata
        self.metadata = None

        # raw data (each database knows what its data is..)
        self._raw_data = None

        # length of dataset
        self.length = 0

        # batch size to serve data
        self._batch_size = 1

        # max num of samples to be served
        self._max_n_samples = None

        # list of pre-process actions to
        # do before serving the data
        self._online_processors = []

        # variables to serve
        self._served_variables = self._get_entry_names()

        # is dataset open
        self.isopen = False

        # order of served samples
        self._order = None
        self._ordering_type = "sorted"  # ["sorted", "user", "shuffled"]

        # default shuffle seed
        self._shuffle_seed = 0

    @classmethod
    def from_path(cls, path):

        if not os.path.exists(path):
            raise FileNotFoundError(f"path {path} not found!")

        # instantiate a database from a path
        return cls(path)

    def open(self, *args, **kwargs):
        """
        Open the DS
        """

        # do the necessary to open the DS
        self._open(args, kwargs)

        # resurrect the schema
        schma_file = os.path.join(self.ds_path, self.schema_json_file)
        self.record_schema = RecordSchema.from_json(schma_file)

        # determine the sample ordering
        self._set_ordering()

        # set the DS as open
        self.isopen = True

    def close(self):
        """
        Close the DS
        """
        self._close()
        self.isopen = False

    @property
    def order(self):
        return self._order

    @order.setter
    def order(self, order_in):

        # use the order just assigned
        self._order = order_in

        # _ordering_type is set to "user"
        self._ordering_type = "user"

    def __getitem__(self, idx):
        """
        Get a sample from the Dataset
        Args:
            idx:

        Returns:

        """

        raise NotImplementedError

    def __len__(self):

        # how many batches do we have?
        return int(np.ceil(self.length / float(self._batch_size)))

    def batch(self, batch_size):
        """
        Set batch size
        Args:
            batch_size:

        Returns:

        """

        self._batch_size = batch_size

        return self

    def shuffle(self, seed=0):
        """
        Set shuffle flag
        Args:
            seed:

        Returns:

        """

        if self.isopen:
            raise ValueError("Database is now open! cannot reshuffle now..")

        self._ordering_type = "shuffle"
        self._shuffle_seed = seed

        return self

    def max_n_samples(self, n_samples):
        """
        Maximum number of samples to be served
        Args:
            n_samples:

        Returns:

        """

        assert n_samples > 0

        self._max_n_samples = n_samples

        return self

    def set_online_processors(self, _pre_proc_list):

        self._online_processors = _pre_proc_list

        return self

    def insert(self, record):
        """
        Append a record to the dataset
        according to the pre-defined
        record schema of the dataset
        Args:
            record:

        Returns:

        """

        # update the DS length
        self.length += 1

        self._do_insert(record)

    def resize(self, new_length):
        """
        Resize the dataset
        """

        assert new_length >= self.length

        self.length = new_length

        self._do_resize(new_length)

    def write_at(self, record, idx, remove_first_axis=True):

        raise NotImplementedError

    def serve(self, vars):

        self._served_variables = vars

        return self

    def _open(self, *args, **kwargs):
        raise NotImplementedError

    def _set_ordering(self):
        """
        After opening the DS a serving order is set.
        NB: The DS length must have already been set at this point
        it can be
          - shuffled
          - set by user
          - sorted (default)
        """

        if self.length is None:
            raise ValueError("Dataset length not yet set! cannot establish ordering..")

        self._order_sorted = np.arange(self.length)

        np.random.seed(self._shuffle_seed)
        self._order_shuffled = np.random.permutation(self.length)[0 : self.length]

        if self._ordering_type == "sorted":
            self.order = self._order_sorted

        elif self._ordering_type == "shuffle":
            self.order = self._order_shuffled
        else:
            # just check that ordering type has already been set by the "user"
            assert self._ordering_type == "user"
            assert self.order is not None

    def _close(self):
        raise NotImplementedError

    def _store_schema(self):
        """
        Tries to store the schema file in the dataset folder
        Returns:
        """

        schema_file = os.path.join(self.ds_path, self.schema_json_file)

        # if ds dir does not exist, create it..
        if not os.path.exists(self.ds_path):
            logger.info(f"DS dir {self.ds_path} does not exist, creating..")
            os.mkdir(self.ds_path)

        # if the schema file does not exist, write it..
        if not os.path.exists(schema_file):
            with open(schema_file, "w") as f:
                json.dump(self.record_schema.to_list(), f, indent=2, sort_keys=True)
        else:
            logger.warning(f"schema file {schema_file} already exists!")

    def _get_entry_names(self):
        """
        Get names of all the served entries
        """

        schema_file = os.path.join(self.ds_path, self.schema_json_file)
        with open(schema_file, "r") as f:
            schema_data = json.load(f)

        entry_names = [e["name"] for e in schema_data]

        return entry_names

    def __str__(self):
        """
        High-Level Description of the dataset
        Returns:
        """

        return "Base Dataset"

    def _check_path(self):

        # check path
        if not os.path.exists(self.ds_path):
            logger.info(f"Dataset path {self.ds_path} does not exist, creating..")
            os.mkdir(self.ds_path)

    @classmethod
    def exists_in_path(cls, filename):
        """
        Check that a dataset exists in path..
        (subclasses might add additional checks)
        Args:
            filename:

        Returns:

        """

        if (
            os.path.exists(os.path.join(filename, cls.schema_json_file))
            and cls._extra_existence_checks()
        ):
            return True
        else:
            return False

    @classmethod
    def _extra_existence_checks(cls):
        """
        Additional dataset existence checks
        Returns:

        """
        return True

    def _do_insert(self, record):
        """
        The actual append to be specialized
        Returns:

        """
        raise NotImplementedError

    def _do_resize(self, new_length):
        """
        Do the Resize of the dataset
        """
        raise NotImplementedError

    def sample_as_dict(self, sample):
        """
        Return a dict key:sample
        """

        return {k: ss for k, ss in zip(self.record_schema.entry_names(), list(sample))}
