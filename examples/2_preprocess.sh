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

source env.sh

# configure the pre-process config file
sed 's|default_output_dir|'"${makaniino_dataset_dir}"'|g' preprocess_config.template > preprocess_config.json
sed -i 's|default_config_path|'"${makaniino_working_dir}"'/download_config.json|g' preprocess_config.json
sed -i 's|default_ibtracks_path|'"${makaniino_ibtracks_path}"'|g' preprocess_config.json

# execute pre-process
mk-preprocess preprocess_config.json