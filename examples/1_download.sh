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

# configure the download config file
sed 's|default_cache_path|'"${makaniino_cache_path}"'|g' \
  download_config.template > download_config.json

# execute the download command
if [ ! -e ${makaniino_cache_path} ]; then
  mkdir -p ${makaniino_cache_path}
fi

mk-download download_config.json
