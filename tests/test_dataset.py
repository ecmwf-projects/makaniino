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
import shutil
import unittest

from makaniino.data_handling.datasets.base import (
    CycloneDataset,
    DatasetTypes,
    EntrySchema,
    RecordSchema,
)


class TestDirectoryHandler:
    """
    manages creation deletion of test directories
    Returns:

    """

    existing_ds_path = "/var/tmp/dummy_ds"
    existing_ib_path = "/var/tmp/dummy_ib"
    non_existent_path = "non_existent_dir"

    def init_dirs(self):
        self.reset_dirs()

    def reset_dirs(self):

        # clean all folders if present
        self.clean_dirs()

        # create "existing" dirs
        if not os.path.exists(self.existing_ds_path):
            os.mkdir(self.existing_ds_path)

        if not os.path.exists(self.existing_ib_path):
            os.mkdir(self.existing_ib_path)

    def clean_dirs(self):

        if os.path.exists(self.existing_ds_path):
            shutil.rmtree(self.existing_ds_path)

        if os.path.exists(self.existing_ib_path):
            shutil.rmtree(self.existing_ib_path)

        if os.path.exists(self.non_existent_path):
            shutil.rmtree(self.non_existent_path)


class DatasetTests(unittest.TestCase):
    """
    Some tests of the workload splitting functionality
    """

    def test_record_entry_schema(self):

        # catch empty schema
        with self.assertRaises(AssertionError) as ar:  # noqa: F841
            e = EntrySchema({})  # noqa: F841

        # catch schema with invalid dict
        with self.assertRaises(AssertionError) as ar:  # noqa: F841
            e = EntrySchema(999)  # noqa: F841

        # catch schema with missing required fields
        with self.assertRaises(AssertionError) as ar:  # noqa: F841
            e = EntrySchema({"some_field": "blabla"})  # noqa: F841

        # catch schema with missing required fields
        with self.assertRaises(AssertionError) as ar:  # noqa: F841
            e = EntrySchema({1: 1})  # noqa: F841

    def test_record_schema(self):
        """init a dataset with an invalid schema"""

        dh = TestDirectoryHandler()
        dh.reset_dirs()

        # catch invalid record entries
        with self.assertRaises(TypeError) as ar:  # noqa: F841
            schema = RecordSchema(  # noqa: F841
                [{"invalid_field": 111}, {"invalid_field": 222}]
            )

    def test_dataset_instantiation(self):

        # utility variables
        dh = TestDirectoryHandler()
        dh.reset_dirs()

        # a dummy schema
        schema = RecordSchema(
            [
                EntrySchema(
                    {
                        "name": "var-1",
                        "shape": (11, 12, 1),
                        "dtype": DatasetTypes.NP_INT(),
                    }
                ),
                EntrySchema(
                    {
                        "name": "var-2",
                        "shape": (22, 23, 2),
                        "dtype": DatasetTypes.NP_FLOAT(),
                    }
                ),
            ]
        )

        # --------- test the ds
        ds = CycloneDataset(dh.existing_ds_path, schema)  # noqa: F841

        # init DS with invalid schema
        with self.assertRaises(AssertionError) as ar:  # noqa: F841
            CycloneDataset(dh.non_existent_path, 444)

        dh.clean_dirs()


if __name__ == "__main__":
    unittest.main()
