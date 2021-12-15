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

mk-keras-model \
  ${makaniino_training_out_dir}/${makaniino_model_name}.config \
  ${weights_save_dir}/dummy_example.weights.h5 \
  --out_path=${makaniino_keras_dir}

