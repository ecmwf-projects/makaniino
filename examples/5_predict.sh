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

if [ ! -e ${makaniino_prediction_dir} ]; then
  mkdir -p ${makaniino_prediction_dir}
fi

# write a configuration for a model for "full-size" fields
# that can reuses the weights of the trained model
sed    's|"img_cols": 64|"img_cols": 704|g' ${makaniino_conf_file} > ${makaniino_conf_file_full}
sed -i 's|"img_rows": 64|"img_rows": 320|g' ${makaniino_conf_file_full}
sed -i 's|"x_cut_pxl": 0|"x_cut_pxl": 20|g' ${makaniino_conf_file_full}
sed -i 's|"y_cut_pxl": 0|"y_cut_pxl": 8|g'  ${makaniino_conf_file_full}
sed -i 's|makaniino_dataset_augmented|makaniino_dataset|g' ${makaniino_conf_file_full}

mk-predict \
  --name=${makaniino_model_name_full} \
  --config_path=${makaniino_training_out_dir} \
  --weights=${weights_save_dir}/${makaniino_model_name}.weights.h5 \
  --data_path=${makaniino_dataset_dir} \
  --out_path=${makaniino_prediction_dir} \
  --number_predicitons 5 \
  --twod_only \
