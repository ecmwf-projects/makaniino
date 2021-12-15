#!/bin/bash
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

# clean config files
rm -f ./download_config.json
rm -f ./preprocess_config.json

# clean output directories
rm -rf ./example.sync

#rm -rf ./makaniino_download
rm -rf ./makaniino_dataset
rm -rf ./makaniino_dataset_augmented/

rm -rf ./makaniino_output
rm -rf ./makaniino_output_tools
