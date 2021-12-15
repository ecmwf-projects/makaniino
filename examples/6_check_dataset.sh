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

if [ ! -e ${makaniino_plot_dir} ]; then
  mkdir -p ${makaniino_plot_dir}
fi

# Check a dataset
time mk-check-dataset \
  --dataset_path ${makaniino_dataset_augmented_dir} \
  --n_plots=10 \
  --shuffle \
  --serve "train,test,time" \
  --out_path=${makaniino_plot_dir}
  #--dataset_path ${makaniino_dataset_dir} \
