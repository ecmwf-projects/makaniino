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

if [ ! -e ${makaniino_dataset_augmented_dir} ]; then
  mkdir -p ${makaniino_dataset_augmented_dir}
fi

# Example of data augmentation (methodology: "window_neg_grid64")
time mk-augment-data \
  ${makaniino_dataset_dir} \
  ${makaniino_dataset_augmented_dir} \
  -t "window_neg_grid64" \
  --workers 1 \
  --chunk_size_ds 20 \
  --lat_cut_pxl 40 \
  --lon_cut_pxl 8 \
  --tracks_source_type "ibtracks_all" \
  --tracks_source_path ${makaniino_ibtracks_path} \
  --flush_every 10
