#!/usr/bin/env python
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

# -*- coding: utf-8 -*-

"""
   labelData.py

   covert lat/lon points to grids points and method to generate binary mask for labels

   author Jebb Q Stewart (jebb.q.stewart@noaa.gov)
   Copyright 2019. Colorado State University. All rights reserved.

"""

import logging

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Setup logging
from makaniino.utils.plot_utils import plot_prediction

logging_format = "%(asctime)s - %(name)s - %(message)s"
logging.root.setLevel(logging.INFO)
logging.basicConfig(
    level=logging.INFO, format=logging_format, datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("LabelData")

# Set to true to show image of data and labels,
# useful for debugging, pain for processing
show_image = False
gray = plt.cm.gray


def pointsToDataPoints(points, lats, lons, offsetX=False, width=0):
    """given points with lat lons, create x,y mapping to points in grid"""
    points = np.array(points)
    logger.debug("Processing % cyclones", points.shape[0])
    xypoints = []

    for p in points:

        # find center point x, y values for grib file
        # 1) Compute the absolute difference between the grid
        #    lat/lon and the point
        if offsetX:
            p[1] += 180
            if p[1] > 180:
                p[1] -= 360

        abslat = np.abs(lats - p[0])
        abslon = np.abs(lons - p[1])

        # 2) Element-wise maxima. (Plot this with pcolormesh
        #    to see what I've done.)
        c = np.maximum(abslon, abslat)

        # if min is greater than 1, it means the point
        # is off the edge of the data
        if np.min(c) < 1:
            # print (np.min(c))

            # 3) The index of the minimum maxima
            #    (which is the nearest lat/lon)
            x, y = np.where(c == np.min(c))
            x = x[0]
            y = y[0]

            # store center points for later
            xypoints.append([x, y])
        else:
            logger.warn("point %s not in bounds of data", p)

    xypoints = np.array(xypoints)

    return xypoints


def createLabeledData(
    data,
    xypoints,
    test_data,
    center_mark=10,
    range_value=0,
    method="square",
    tc_classes=None,
):
    """create mask for underlying data based on points

    creates a binary mask same size of data using xy points as the center

    Arguments:
        data {2d array} -- only need for shape of mask or test images
        xypoints {list(int, int)} -- list of grid x,y points for labels
        test_data {list} -- array to append list too

    Keyword Arguments:
        center_mark {number} -- number of points off center_mark to create label (default: {10})
        range_value {number} -- if you are testing, specify range_value to help color code image (default: {0})
    """
    logger.debug(f"Creating labeled data with method {method}")
    # print(f"xypoints \n{xypoints}")
    # print(f"tc_classes \n{tc_classes}")

    # if tc classes are provided, order tc according to
    # their intensity (so that the most intense will be
    # labelled last => its label will dominate)
    if tc_classes:
        tc_classes = np.asarray(tc_classes)
        tc_class_sort_idx = tc_classes.argsort()
        xypoints = xypoints[tc_class_sort_idx, :]
        tc_classes = tc_classes[tc_class_sort_idx]
    else:
        tc_classes = np.ones((xypoints.shape[0], 1))

    # print(f"ORDERED xypoints \n{xypoints}")
    # print(f"ORDERED tc_classes \n{tc_classes}")

    if method == "cone":

        test = np.zeros(data.shape, dtype=np.float)

        ylen, xlen = data.shape

        xidx = np.arange(xlen)
        yidx = np.arange(ylen)

        mesh_grid = np.meshgrid(xidx, yidx)
        mesh_zipped = np.stack((mesh_grid[0], mesh_grid[1]), axis=2)

        for p, cl in zip(xypoints, tc_classes):
            p = np.flip(p).reshape((1, 1, 2))  # pt coords
            diff = mesh_zipped - np.tile(p, (*data.shape, 1))  # coord diff
            d2 = np.sum(diff ** 2, axis=2)  # square distance
            mask = np.array(d2 < center_mark ** 2)  # within-circle mask

            # TODO: still need to find a good way of using the tc class
            # test[mask] = np.maximum(test[mask], 1 - d2[mask]**0.5/center_mark)  # flag the points within as a cone
            test[mask] = 1 - d2[mask] ** 0.5 / center_mark
            test[mask] *= cl

        if np.isnan(data).any():
            mask = np.where(np.isnan(data))
            test[mask[0], mask[1]] = 0

        test_data.append(test)

    if method == "sine":

        test = np.zeros(data.shape, dtype=np.float)

        ylen, xlen = data.shape

        xidx = np.arange(xlen)
        yidx = np.arange(ylen)

        mesh_grid = np.meshgrid(xidx, yidx)
        mesh_zipped = np.stack((mesh_grid[0], mesh_grid[1]), axis=2)

        # for p in xypoints:
        for p, cl in zip(xypoints, tc_classes):
            p = np.flip(p).reshape((1, 1, 2))  # pt coords
            diff = mesh_zipped - np.tile(p, (*data.shape, 1))  # coord diff
            d2 = np.sum(diff ** 2, axis=2)  # square distance
            mask = np.array(d2 < center_mark ** 2)  # within-circle mask

            # TODO: still need to find a good way of using the tc class
            test[mask] = np.maximum(
                test[mask], (1 + np.cos(d2[mask] ** 0.5 / center_mark * np.pi)) / 2
            )
            # test[mask] *= cl

        if np.isnan(data).any():
            mask = np.where(np.isnan(data))
            test[mask[0], mask[1]] = 0

        test_data.append(test)

    if method == "circle":

        test = np.zeros(data.shape, dtype=np.float)

        ylen, xlen = data.shape

        xidx = np.arange(xlen)
        yidx = np.arange(ylen)

        mesh_grid = np.meshgrid(xidx, yidx)
        mesh_zipped = np.stack((mesh_grid[0], mesh_grid[1]), axis=2)

        # for p in xypoints:
        for p, cl in zip(xypoints, tc_classes):
            p = np.flip(p).reshape((1, 1, 2))  # pt coords
            diff = mesh_zipped - np.tile(p, (*data.shape, 1))  # coord diff
            d2 = np.sum(diff ** 2, axis=2)  # square distance
            mask = np.array(d2 < center_mark ** 2)  # within-circle mask
            test[mask] = 1
            test[mask] *= cl

        if np.isnan(data).any():
            mask = np.where(np.isnan(data))
            test[mask[0], mask[1]] = 0

        test_data.append(test)

    elif method == "square":

        img = None
        if show_image:
            if range_value == 0:
                max_value = np.max(data)
                min_value = np.min(data)
                range_value = max_value - min_value
                logger.debug("range: %s", range_value)

            img = Image.fromarray(
                gray((data.astype(np.float32) / range_value), bytes=True)
            )
            img.show()

        # Mark Points on original data
        if show_image:
            logger.debug(xypoints)
            for p in xypoints:
                for i in range(-center_mark, center_mark):
                    for j in range(-center_mark, center_mark):
                        if (
                            p[1] + j < img.width
                            and p[1] + j >= 0
                            and p[0] + j < img.height
                            and p[0] + j >= 0
                        ):
                            #   r, g, b, a = img.getpixel((p[1]+j, p[0]+i))
                            img.putpixel((p[1] + j, p[0] + i), (255, 0, 0))

        test = np.zeros(data.shape, dtype=np.uint8)
        # for p in xypoints:
        for p, cl in zip(xypoints, tc_classes):
            ymin = max(0, p[0] - center_mark)
            xmin = max(0, p[1] - center_mark)
            ymax = min(data.shape[0] - 1, p[0] + center_mark)
            xmax = min(data.shape[1] - 1, p[1] + center_mark)
            # test[ymin:ymax, xmin:xmax] = 1
            test[ymin:ymax, xmin:xmax] = cl

        if np.isnan(data).any():
            mask = np.where(np.isnan(data))
            test[mask[0], mask[1]] = 0

        if show_image:
            img2 = Image.fromarray(gray(test / 1.0, bytes=True))
            img2.show()
            img.show()

        test_data.append(test)


# ********* testing only *********
if __name__ == "__main__":

    # data = np.zeros((360, 720))
    data = np.zeros((640, 1408))

    xypoints = np.array([[140, 587], [145, 220], [145, 650]])

    # since the 2nd makaniino is stringer,
    # we expect it to be labelled last
    tc_classes = [1, 4, 2]

    test_data = []

    # flagging_method = "cone"
    # flagging_method = "square-flat"
    flagging_method = "circle"

    createLabeledData(
        data,
        xypoints,
        test_data,
        center_mark=100,
        range_value=0,
        method=flagging_method,
        tc_classes=tc_classes,
    )

    ground_truth = test_data[0]

    # print(f"ground_truth.shape {ground_truth.shape}")
    # print(f"MAX TEST {np.max(ground_truth)}")
    # print(f"MIN TEST {np.min(ground_truth)}")
    # print(f"AVG TEST {np.mean(ground_truth)}")
    #
    # for i in range(ground_truth.shape[0]):
    #     for j in range(ground_truth.shape[1]):
    #         if 0 < ground_truth[i, j] < 1:
    #             print(f"ground_truth[{i}, {j}]: {ground_truth[i, j]}")

    plot_prediction(
        ground_truth,
        ground_truth,
        ground_truth,
        output_dir="legacy_cyclone/",
        title="prediction",
        twod_only=True,
    )
