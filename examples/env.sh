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

makaniino_working_dir=$(pwd)

# model name
makaniino_model_name=dummy_example

# model name (for full-field configuration)
makaniino_model_name_full=dummy_example_full

# download dirs
makaniino_download_dir=${makaniino_working_dir}/makaniino_download
makaniino_cache_path=${makaniino_download_dir}/cache

# pre-process dirs
makaniino_dataset_dir=${makaniino_working_dir}/makaniino_dataset
makaniino_ibtracks_path=${makaniino_working_dir}/ibtracs.since2018.csv
makaniino_dataset_augmented_dir=${makaniino_working_dir}/makaniino_dataset_augmented

# training dirs
makaniino_training_out_dir=${makaniino_working_dir}/makaniino_output
model_save_dir=${makaniino_training_out_dir}/models
weights_save_dir=${makaniino_training_out_dir}/weights
checkpoint_dir=${makaniino_training_out_dir}/checkpoints
images_save_dir=${makaniino_training_out_dir}/images
tb_save_dir=${makaniino_training_out_dir}/tensorboard

makaniino_log_file=${makaniino_training_out_dir}/${makaniino_model_name}.log
makaniino_conf_file=${makaniino_training_out_dir}/${makaniino_model_name}.config

makaniino_conf_file_full=${makaniino_training_out_dir}/${makaniino_model_name_full}.config

# other tools (data augmentation, etc..)
makaniino_analysis_output=${makaniino_working_dir}/makaniino_output_tools
makaniino_plot_dir=${makaniino_analysis_output}/plots
makaniino_model_check_dir=${makaniino_analysis_output}/model_check
makaniino_keras_dir=${makaniino_analysis_output}/keras_model
makaniino_prediction_dir=${makaniino_analysis_output}/predictions
