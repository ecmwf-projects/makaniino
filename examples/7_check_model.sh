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

if [ ! -e ${makaniino_model_check_dir} ]; then
  mkdir -p ${makaniino_model_check_dir}
fi

# Check a model
time mk-diagnostics \
  --name=${makaniino_model_name_full} \
  --config_path=${makaniino_training_out_dir} \
  --weights=${weights_save_dir}/dummy_example.weights.h5 \
  --out_path=${makaniino_model_check_dir} \
  --layer="conv2d_1" \
#  --info \
