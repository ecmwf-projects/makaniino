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

from makaniino.data_augmentation.window import DataAugmenter_Window

# import matplotlib.pyplot as plt
from makaniino.utils.generic_utils import (
    latlon_2_pixel,
    npstring_to_numpy,
    pixel_2_latlon,
)

logger = logging.getLogger(__name__)


class DataAugmenter_WindowNeg(DataAugmenter_Window):
    """
    Cut out the field around the makaniino-areas (cuts 128 x 128)
    and retains some random negative cutouts
    """

    # max num of attempts to find a negative
    n_attempts_max = 20

    # cut tolerance (pxls)
    eps_pxls = 3

    # check for real negatives
    check_real_negatives = False

    def __init__(self, *args, **kwargs):
        super(DataAugmenter_WindowNeg, self).__init__(*args, **kwargs)

        # number of negatives to extract
        self.n_rand_negs = self.cutout_nx * self.cutout_ny

        # min dist for negatives (square)
        self.min_dist2 = max(self.cutout_total_dx, self.cutout_total_dy) ** 2

    def _extract_negatives(self, input_sample):

        train, test, time, tc_cords = input_sample

        crops_neg = []

        # sample shape
        _, train_shape_x, train_shape_y, _ = train.shape

        # preprocessor used to pack/unpack data
        tc_cords_np = npstring_to_numpy(tc_cords)

        time_np = time[0, 0]

        # Apply coordinate conversion
        cyc_cords_pxl = latlon_2_pixel(tc_cords_np, test.shape, 0, 0)

        # now add random negatives
        n_placed_negs = 0
        n_attempts = 0

        while (n_placed_negs <= self.n_rand_negs) and (
            n_attempts < self.n_attempts_max
        ):

            x_rand_pxl = np.clip(
                np.random.rand() * train_shape_x,
                self.cyclone_img_size[0] / 2 + self.eps_pxls,
                train_shape_x - self.cyclone_img_size[0] / 2 - self.eps_pxls,
            ).astype(np.int)

            y_rand_pxl = np.clip(
                np.random.rand() * train_shape_y,
                self.cyclone_img_size[1] / 2 + self.eps_pxls,
                train_shape_y - self.cyclone_img_size[1] / 2 - self.eps_pxls,
            ).astype(np.int)

            xmin = (x_rand_pxl - self.cyclone_img_size[0] / 2).astype(np.int)
            xmax = (x_rand_pxl + self.cyclone_img_size[0] / 2).astype(np.int)
            ymin = (y_rand_pxl - self.cyclone_img_size[1] / 2).astype(np.int)
            ymax = (y_rand_pxl + self.cyclone_img_size[1] / 2).astype(np.int)

            vv = np.asarray([[x_rand_pxl, y_rand_pxl]])
            cyc_cords_latlon = pixel_2_latlon(vv, test.shape)
            cyc_cords_lat = cyc_cords_latlon[0, 0]
            cyc_cords_lon = cyc_cords_latlon[0, 1]

            # calculate distance from all the other cyclones
            cyc_cords_pxl_np = np.asarray(cyc_cords_pxl)
            cyc_cords_pxl_np_x = cyc_cords_pxl_np[:, 0]
            cyc_cords_pxl_np_y = cyc_cords_pxl_np[:, 1]

            dist2_x = np.square(
                x_rand_pxl * np.ones_like(cyc_cords_pxl_np_x) - cyc_cords_pxl_np_x
            )
            dist2_y = np.square(
                y_rand_pxl * np.ones_like(cyc_cords_pxl_np_y) - cyc_cords_pxl_np_y
            )

            dist2 = dist2_x + dist2_y

            # if the randomly picked sample is far enough from all the
            # cyclones, then keep it
            if all(d2 >= self.min_dist2 for d2 in dist2):

                # OK, the point we picked is far enough from all the cyclones
                # contained in the input sample, but still there could be other
                # cyclones in the dataset (not chosen when forming the input sample).
                # Let's check that ths point we picked is far enough from any makaniino
                # recorded in the makaniino tracks.

                # by default, let's assume that this is a real negative
                cyc_here = False

                # if we do check for real negatives
                if self.check_real_negatives:
                    cyc_here = self.tracks_source.is_a_cyclone_here(
                        time_np, cyc_cords_lat, cyc_cords_lon, np.sqrt(self.min_dist2)
                    )

                    logger.debug(
                        f"No Cyclones near {cyc_cords_lat:.2f}, {cyc_cords_lon:.2f}! "
                        f"=> -ve crop looks OK, continue.."
                    )

                # if there is a makaniino here, do not extract the crop
                if cyc_here:
                    logger.info("Cyclone too  close! => crop discarded")
                    train_cut = None
                    test_cut = None
                else:
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
                        (x_rand_pxl, y_rand_pxl),
                        (xmax - xmin, ymax - ymin),
                        rot_rad=alpha,
                    )

                    test_cut = self._do_crop(
                        test,
                        (x_rand_pxl, y_rand_pxl),
                        (xmax - xmin, ymax - ymin),
                        rot_rad=alpha,
                    )

                # check if the crop has been successful..
                if train_cut is not None and test_cut is not None:

                    crops_neg.append(
                        {
                            "train": train_cut,
                            "test": test_cut,
                            "time": time,
                            "points": np.atleast_2d(np.asarray(["[]"])),
                        }
                    )
                    # a neg crop has been taken!
                    n_placed_negs += 1

                    # print("taken!!")
                    # plt.imshow(train_cut[0, :, :, 0], cmap="gray")
                    # plt.show()
                    #
                    # plt.imshow(test_cut[0, :, :, 0])
                    # plt.show()
            else:
                # print("..REJECTED")
                pass

            # another attempt has been made!
            n_attempts += 1

        return crops_neg


class DataAugmenter_WindowNeg_Rot(DataAugmenter_WindowNeg):
    """
    Cut out the field around the makaniino-areas (cuts 64 x 64)
    and retains some random negative cutouts
    """

    rotate_samples = True
    rand_rot_max_deg = 20


class DataAugmenter_WindowNeg64(DataAugmenter_WindowNeg):
    """
    Cut out the field around the makaniino-areas (cuts 64 x 64)
    and retains some random negative cutouts
    """

    cyclone_img_size = (64, 64)


class DataAugmenter_WindowNeg64_Rot(DataAugmenter_WindowNeg64):
    """
    Cut out the field around the makaniino-areas (cuts 64 x 64)
    and retains some random negative cutouts
    """

    rotate_samples = True
    rand_rot_max_deg = 20
