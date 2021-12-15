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

mk-evaluate-model \
  --name=${makaniino_model_name_full} \
  --config_path=${makaniino_training_out_dir} \
  --weights=${weights_save_dir}/dummy_example.weights.h5 \
  --data_path=${makaniino_dataset_dir} \
  --score_type="precision,recall,f1_score" \
  --n_evals=3 \
  --shuffle

