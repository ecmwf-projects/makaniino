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

import numpy as np

from makaniino.data_augmentation.window_neg import DataAugmenter_WindowNeg, logger
from makaniino.utils.generic_utils import (
    latlon_2_pixel,
    npstring_to_numpy,
    pixel_2_latlon,
)

# import matplotlib
# matplotlib.use("TkAgg")
# import matplotlib.pyplot as plt


class DataAugmenter_WindowNeg_GRID(DataAugmenter_WindowNeg):
    """
    Cut out the field around the makaniino-areas (cuts 128 x 128)
    and retains some random negative cutouts spread in a GRID.
    This in an attempt to retain more widely spread negative examples
    that could be more representative of the global field (including
    extreme low/high latitudes, etc..)
    """

    # max num of attempts to find a negative
    # within a certain cell of the grid
    n_attempts_max_per_cell = 2

    # grid size
    grid_size_x = 3
    grid_size_y = 3

    def _extract_negatives(self, input_sample):

        train, test, time, tc_cords = input_sample

        crops_neg = []

        # input image shape
        _, train_shape_x, train_shape_y, _ = train.shape

        # cell dimension (in pxls)
        self.grid_cell_dim_pxl_x = train_shape_x // self.grid_size_x
        self.grid_cell_dim_pxl_y = train_shape_y // self.grid_size_y

        logger.debug(f"self.grid_cell_dim_pxl_x {self.grid_cell_dim_pxl_x}")
        logger.debug(f"self.grid_cell_dim_pxl_y {self.grid_cell_dim_pxl_y}")

        # preprocessor used to pack/unpack data
        tc_cords_np = npstring_to_numpy(tc_cords)

        time_np = time[0, 0]

        # Apply coordinate conversion
        cyc_cords_pxl = latlon_2_pixel(tc_cords_np, test.shape, 0, 0)

        # Extraction from the grid cells
        n_placed_negs_grid = 0
        for igrid in range(self.grid_size_x):
            for jgrid in range(self.grid_size_y):

                logger.debug(f"extracting from cell {igrid},{jgrid}..")

                cell_x_min = self.grid_cell_dim_pxl_x * igrid
                cell_x_max = self.grid_cell_dim_pxl_x * (igrid + 1)
                cell_y_min = self.grid_cell_dim_pxl_y * jgrid
                cell_y_max = self.grid_cell_dim_pxl_y * (jgrid + 1)

                logger.debug(f"cell_x: min max [{cell_x_min},{cell_x_max}]")
                logger.debug(f"cell_y: min max [{cell_y_min},{cell_y_max}]")

                n_placed_negs_cell = 0
                n_attempts_in_cell = 0
                while (n_placed_negs_grid <= self.n_rand_negs) and (
                    n_attempts_in_cell < self.n_attempts_max_per_cell
                ):

                    # print(f"n_placed_negs_cell {n_placed_negs_cell}, "
                    #       f"self.n_rand_negs {self.n_rand_negs}, "
                    #       f"n_attempts_in_cell {n_attempts_in_cell}, "
                    #       f"self.n_attempts_max_per_cell {self.n_attempts_max_per_cell}")

                    x_rand_pxl = np.clip(
                        cell_x_min + np.random.rand() * self.grid_cell_dim_pxl_x,
                        self.cyclone_img_size[0] / 2 + self.eps_pxls,
                        train_shape_x - self.cyclone_img_size[0] / 2 - self.eps_pxls,
                    ).astype(np.int)

                    y_rand_pxl = np.clip(
                        cell_y_min + np.random.rand() * self.grid_cell_dim_pxl_y,
                        self.cyclone_img_size[1] / 2 + self.eps_pxls,
                        train_shape_y - self.cyclone_img_size[1] / 2 - self.eps_pxls,
                    ).astype(np.int)

                    xmin = (x_rand_pxl - self.cyclone_img_size[0] / 2).astype(np.int)
                    xmax = (x_rand_pxl + self.cyclone_img_size[0] / 2).astype(np.int)
                    ymin = (y_rand_pxl - self.cyclone_img_size[1] / 2).astype(np.int)
                    ymax = (y_rand_pxl + self.cyclone_img_size[1] / 2).astype(np.int)

                    vv = np.asarray([[x_rand_pxl, y_rand_pxl]])
                    cyc_cords_latlon = pixel_2_latlon(vv, test.shape)

                    logger.debug(
                        f"selected negtv in LAT/LON (PXLS) {(x_rand_pxl, y_rand_pxl)} "
                        f"=>  REAL {cyc_cords_latlon}"
                    )

                    cyc_cords_lat = cyc_cords_latlon[0, 0]
                    cyc_cords_lon = cyc_cords_latlon[0, 1]

                    # calculate distance from all the other cyclones
                    cyc_cords_pxl_np = np.asarray(cyc_cords_pxl)
                    cyc_cords_pxl_np_x = cyc_cords_pxl_np[:, 0]
                    cyc_cords_pxl_np_y = cyc_cords_pxl_np[:, 1]

                    dist2_x = np.square(
                        x_rand_pxl * np.ones_like(cyc_cords_pxl_np_x)
                        - cyc_cords_pxl_np_x
                    )
                    dist2_y = np.square(
                        y_rand_pxl * np.ones_like(cyc_cords_pxl_np_y)
                        - cyc_cords_pxl_np_y
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
                                time_np,
                                cyc_cords_lat,
                                cyc_cords_lon,
                                np.sqrt(self.min_dist2),
                            )

                        # if there is a makaniino here, do not extract the crop
                        if cyc_here:
                            logger.info("Cyclone too  close! => crop discarded")
                            train_cut = None
                            test_cut = None
                        else:

                            logger.debug(
                                f"No Cyclones near {cyc_cords_lat:.2f}, {cyc_cords_lon:.2f}! "
                                f"=> -ve crop looks OK, continue.."
                            )

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
                            n_placed_negs_cell += 1

                            # print("taken!!")
                            # plt.subplots(2, 1)
                            # plt.subplot(2, 1, 1)
                            # plt.imshow(train_cut[0, :, :, 0], cmap="gray")
                            # plt.subplot(2, 1, 2)
                            # plt.imshow(test_cut[0, :, :, 0])
                            # plt.show()
                            # plt.close()
                    else:
                        logger.debug("..REJECTED")
                        pass

                    # another attempt has been made!
                    n_attempts_in_cell += 1
                    logger.debug(f"n_placed_negs {n_placed_negs_cell}")

                n_placed_negs_grid += n_placed_negs_cell

        return crops_neg


class DataAugmenter_WindowNeg_GRID64(DataAugmenter_WindowNeg_GRID):
    """
    Cut out the field around the makaniino-areas (cuts 64 x 64)
    and retains some random negative cutouts
    """

    cyclone_img_size = (64, 64)
