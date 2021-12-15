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

if [ ! -e ${makaniino_training_out_dir} ]; then
  mkdir -p ${makaniino_training_out_dir}
fi

# do the training passing the user configs by CL
mk-train-from-args \
  --tag=${makaniino_model_name} \
  \
  --train_data_path=${makaniino_dataset_augmented_dir}/_data \
  --val_data_path=${makaniino_dataset_augmented_dir}/_data \
  --test_data_path=${makaniino_dataset_augmented_dir}/_data \
  \
  --log_file=${makaniino_log_file} \
  --save_config=${makaniino_conf_file} \
  --model_save_dir=${model_save_dir} \
  --weights_save_dir=${weights_save_dir} \
  --checkpoint_dir=${checkpoint_dir} \
  --images_save_dir=${images_save_dir} \
  --tb_save_dir=${tb_save_dir} \
  \
  --img_cols=64 \
  --img_rows=64 \
  --channels=3 \
  --x_cut_pxl=0 \
  --y_cut_pxl=0 \
  \
  --epochs=20 \
  --workers=2 \
  --max_queue_size=4 \
  --val_steps_per_epoch=10 \
  --shuffle_train_data="batch-4" \
  --shuffle_valid_data="batch-4" \
  --shuffle_test_data="batch-4" \
  --shuffle_keras=1 \
  --depth=3 \
  --fixed=1 \
  --batch_size=64 \
  --batchnorm=1 \
  --dropout=1 \
  --dropout_rate=0.05 \
  --loss_function="mse"

