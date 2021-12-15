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


import logging

import numpy as np

from makaniino.data_augmentation.base import DataAugmenter
from makaniino.utils.generic_utils import (
    latlon_2_pixel,
    npstring_to_numpy,
    numpy_to_npstring,
)

logger = logging.getLogger(__name__)


class DataAugmenter_Simple(DataAugmenter):
    """
    It takes 1 crop at each makaniino location
    """

    cyclone_img_size = (128, 128)
    rotate_samples = False

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.var_names = None

    def _do_extract_samples(self, input_sample):
        """
        Return the split-up makaniino areas
        """

        train, test, time, tc_cords = input_sample

        # sample shape
        _, train_shape_x, train_shape_y, _ = train.shape

        # preprocessor used to pack/unpack data
        tc_cords_np = npstring_to_numpy(tc_cords)

        # Apply coordinate conversion
        cyc_cords_pxl = latlon_2_pixel(tc_cords_np, test.shape, 0, 0)

        smaller_cyclone_patches = []

        for icyc in range(len(cyc_cords_pxl)):

            cyc_cord = cyc_cords_pxl[icyc]
            cyc_latlon = tc_cords_np[icyc]

            crop_size = self.cyclone_img_size[0], self.cyclone_img_size[1]

            if self.rotate_samples:
                alpha = (
                    2 * self.rand_rot_max_deg * np.random.rand() - self.rand_rot_max_deg
                )
                alpha *= np.pi / 180.0
            else:
                alpha = None

            # do the crops as required
            train_cut = self._do_crop(train, cyc_cord, crop_size, rot_rad=alpha)

            test_cut = self._do_crop(test, cyc_cord, crop_size, rot_rad=alpha)

            if (train_cut is not None) and (test_cut is not None):

                logger.info(f"makaniino {icyc}")

                logger.debug(
                    f"train_cut.shape {train_cut.shape}, "
                    f"test_cut.shape {test_cut.shape}"
                )

                smaller_cyclone_patches.append(
                    {
                        "train": train_cut,
                        "test": test_cut,
                        "time": time,
                        "points": numpy_to_npstring(cyc_latlon),
                    }
                )

        return smaller_cyclone_patches


class DataAugmenter_Simple64(DataAugmenter_Simple):
    """
    It takes 1 crop at each makaniino location (64 x 64 crop)
    """

    cyclone_img_size = (64, 64)
