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

# import matplotlib
# matplotlib.use("TkAgg")
# import matplotlib.pyplot as plt


logger = logging.getLogger(__name__)


class DataAugmenter_Window(DataAugmenter):
    """
    It takes several crops in a window
    around each makaniino
    """

    # N of cutouts to extract along each dimension
    cutout_nx = 3
    cutout_ny = 3

    def __init__(self, *args, **kwargs):

        super(DataAugmenter_Window, self).__init__(*args, **kwargs)

        # total area dims from which cutouts are extracted..
        self.cutout_total_dx = int(self.cyclone_img_size[0] * 1.3)
        self.cutout_total_dy = int(self.cyclone_img_size[1] * 1.3)

        # make sure that the total area is covered by the windowing procedure
        assert self.cyclone_img_size[0] * self.cutout_nx > self.cutout_total_dx
        assert self.cyclone_img_size[1] * self.cutout_ny > self.cutout_total_dy

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

        # cutout strides
        cutout_stride_x = int(
            (self.cutout_total_dx - self.cyclone_img_size[0]) / (self.cutout_nx - 1)
        )
        cutout_stride_y = int(
            (self.cutout_total_dy - self.cyclone_img_size[1]) / (self.cutout_ny - 1)
        )

        cyclone_crops_pos = []  # positive crops (around cyclones)
        cyclone_crops_neg = []  # negative crops (far from cyclones)
        for icyc in range(len(cyc_cords_pxl)):

            cyc_cord = cyc_cords_pxl[icyc]
            cyc_latlon = tc_cords_np[icyc]

            xmins = [
                (cyc_cord[0] - self.cutout_total_dx / 2) + i * cutout_stride_x
                for i in range(self.cutout_nx)
            ]

            xmaxs = [
                (cyc_cord[0] - self.cutout_total_dx / 2)
                + i * cutout_stride_x
                + self.cyclone_img_size[0]
                for i in range(self.cutout_nx)
            ]

            ymins = [
                (cyc_cord[1] - self.cutout_total_dy / 2) + j * cutout_stride_y
                for j in range(self.cutout_ny)
            ]

            ymaxs = [
                (cyc_cord[1] - self.cutout_total_dy / 2)
                + j * cutout_stride_y
                + self.cyclone_img_size[1]
                for j in range(self.cutout_ny)
            ]

            # cut the area of the makaniino
            # and also 8 cutouts all around
            icount = 0
            for xmin, xmax in zip(xmins, xmaxs):

                jcount = 0
                for ymin, ymax in zip(ymins, ymaxs):

                    patch_ctr = (int((xmin + xmax) / 2), int((ymin + ymax) / 2))

                    if self.rotate_samples:
                        alpha = (
                            2 * self.rand_rot_max_deg * np.random.rand()
                            - self.rand_rot_max_deg
                        )
                        alpha *= np.pi / 180.0
                    else:
                        alpha = None

                    # do the crops as required
                    train_cut = self._do_crop(
                        train,
                        patch_ctr,
                        (int(xmax - xmin), int(ymax - ymin)),
                        rot_rad=alpha,
                    )

                    test_cut = self._do_crop(
                        test,
                        patch_ctr,
                        (int(xmax - xmin), int(ymax - ymin)),
                        rot_rad=alpha,
                    )

                    # take only cutouts fully contained in the input image
                    if train_cut is not None and test_cut is not None:

                        # show image
                        # print(f"taken sample at {(xmin, xmax, ymin, ymax)}, time {time}")
                        # plt.subplots(2, 1)
                        # plt.subplot(2, 1, 1)
                        # plt.imshow(train_cut[0, :, :, 0], cmap="gray")
                        # plt.subplot(2, 1, 2)
                        # plt.imshow(test_cut[0, :, :, 0])
                        # plt.show()
                        # plt.close()

                        cyclone_crops_pos.append(
                            {
                                "train": train_cut,
                                "test": test_cut,
                                "time": time,
                                "points": numpy_to_npstring(cyc_latlon),
                            }
                        )
                    jcount += 1
                icount += 1

            # for each makaniino one could extract also some negatives..
            cyclone_crops_neg += self._extract_negatives(input_sample)

        return cyclone_crops_pos, cyclone_crops_neg

    def _extract_negatives(self, input_sample):
        """
        Place holder for extracting negative samples
        nothing to do here..
        """

        return []


class DataAugmenter_Window_Rot(DataAugmenter_Window):
    """
    It takes several crops in a window
    around each makaniino (64 x 64 crop)
    """

    rotate_samples = True
    rand_rot_max_deg = 20


class DataAugmenter_Window64(DataAugmenter_Window):
    """
    It takes several crops in a window
    around each makaniino (64 x 64 crop)
    """

    cyclone_img_size = (64, 64)


class DataAugmenter_Window64_Rot(DataAugmenter_Window64):
    """
    It takes several crops in a window
    around each makaniino (64 x 64 crop)
    """

    rotate_samples = True
    rand_rot_max_deg = 20
