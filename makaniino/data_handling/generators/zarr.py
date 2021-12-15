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

import zarr

from makaniino.data_handling.generators.multi_ds import DataGeneratorMultiDS

logger = logging.getLogger(__name__)


class DataGenerator_ZARR(DataGeneratorMultiDS):
    """
    Serves data from (multiple) ZARR datasets
    """

    @staticmethod
    def read_dataset(paths, served_variables):
        """
        Read arrays from zarr archive
        """

        # Data from multiple datasets
        all_data = {}
        for k, v in paths.items():

            logger.info(f"Reading data from path {k} - from {v['from']} to {v['to']}")

            # open zarr archive
            loaded_data = zarr.open(k, mode="r")
            logger.info(f"Dataset contains fields: {[k for k in loaded_data]}")

            all_data.update(
                {k: {var_name: loaded_data[var_name] for var_name in served_variables}}
            )

        return all_data
