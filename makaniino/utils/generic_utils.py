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

# import bisect
import datetime
import json
import logging
import math
import os

import numpy as np

logger = logging.getLogger(__name__)


def read_json(json_path):
    """
    Simply read a json file
    """

    if not os.path.exists(json_path):
        raise OSError(f"Path {json_path} does not exist!")

    with open(json_path, "r") as f:
        json_data = json.load(f)

    return json_data


def pixel_2_latlon(pixel_indexes, sample_shape, x_cut_pxl=0, y_cut_pxl=0):
    """
    Function that maps pixel indexes with
    lat long coordinates (according to size and x,y cut areas
    """

    # array of pixel indexes
    pixel_indexes = np.asarray(pixel_indexes)

    if not pixel_indexes.size:
        return []

    # data shapes
    _, shape_x, shape_y, _ = sample_shape

    x_plus_xcut = pixel_indexes[:, 0] + x_cut_pxl
    y_plus_ycut = pixel_indexes[:, 1] + y_cut_pxl

    x_rel_glob = -x_plus_xcut.reshape(-1, 1) / (shape_x + 2 * x_cut_pxl) * 180 + 90
    y_rel_glob = y_plus_ycut.reshape(-1, 1) / (shape_y + 2 * y_cut_pxl) * 360

    # longitude goes from -180 to 180
    gt180_idx = np.where(y_rel_glob > 180)
    y_rel_glob[gt180_idx] = y_rel_glob[gt180_idx] - 360.0

    coords = np.hstack((x_rel_glob, y_rel_glob))

    return coords


def latlon_2_pixel(lat_lon, sample_shape, x_cut_pxl=0, y_cut_pxl=0):
    """
    Function that maps lat/lon to pixel indexes
    (according to size and x,y cut areas)
    """

    lat_lon = np.array(lat_lon, copy=True)
    lats = lat_lon[:, 0]
    lons = lat_lon[:, 1]

    lons_lt_0 = np.where(lons < 0)
    lons[lons_lt_0] = 360 + lons[lons_lt_0]

    # data shapes
    _, shape_x, shape_y, _ = sample_shape

    y_pxl = (-lats + 90) / 180.0 * (shape_x + 2 * x_cut_pxl) - x_cut_pxl
    y_pxl = y_pxl.astype(np.int)
    y_pxl = y_pxl.reshape((-1, 1))

    x_pxl = lons / 360.0 * (shape_y + 2 * y_cut_pxl) - y_cut_pxl
    x_pxl = x_pxl.astype(np.int)
    x_pxl = x_pxl.reshape((-1, 1))

    return np.hstack((y_pxl, x_pxl))


def haversine_dist(lat1, lon1, lat2, lon2):
    """
    Haversine distance
    """
    p = math.pi / 180

    a = (
        0.5
        - math.cos((lat2 - lat1) * p) / 2
        + math.cos(lat1 * p)
        * math.cos(lat2 * p)
        * (1 - math.cos((lon2 - lon1) * p))
        / 2
    )

    return 12742 * math.asin(math.sqrt(a))  # 2*R*asin...


def time_to_numpy(time):

    # pack the time numpy to be stored
    time_np = np.array(time.strftime("%Y-%m-%d %H:%M:00"))
    time_np.resize((1, 1))

    return time_np


def numpy_to_npstring(points, str_buffr_len=2048, str_buffr_char="b"):
    """
    Dummy function to transform [1-2]d numpys
    to string numpy (with max len = STR_BUFFR_LEN)
    """

    points = np.atleast_2d(points)

    # makaniino coordinates string
    try:
        coord_string = ";".join(",".join(str(p) for p in pp) for pp in points)
    except TypeError:
        print(f"=> points \n {points}")
        raise TypeError

    # pad the makaniino coordinate string as necessary
    assert len(coord_string) < str_buffr_len
    coord_string += str_buffr_char * (str_buffr_len - len(coord_string))

    points_np = np.array(coord_string)
    points_np.resize((1, 1))

    return points_np


def npstring_to_numpy(points_from_zarr, str_buffr_char="b"):

    pc = str_buffr_char
    data_str = points_from_zarr[0, 0].replace(pc, "")
    cyc_coords_latlon_real = np.asarray(
        [
            [float(c) for c in coords.split(",")]
            for coords in data_str.split(";")
            if coords
        ]
    )

    return cyc_coords_latlon_real


def date_idx_in_zarr(date_array, date_range, start_from=0):
    """
    Find the index corresponding to a specific date in the zarr array
    """

    # work out the start date
    if date_range[0] != "":
        start_date = datetime.datetime.strptime(date_range[0], "%Y-%m-%d")
    else:

        first_date_in_array = str(date_array[0][0])
        start_date = datetime.datetime.strptime(
            first_date_in_array, "%Y-%m-%d %H:%M:%S"
        )

    # work out the end date
    if date_range[1] != "":
        end_date = datetime.datetime.strptime(date_range[1], "%Y-%m-%d")
    else:

        last_date_in_array = str(date_array[-1][0])
        end_date = datetime.datetime.strptime(last_date_in_array, "%Y-%m-%d %H:%M:%S")

    logger.info(
        f" Searching for timestamps in range [{start_date}, {end_date}] - "
        f"(total DS length {date_array.shape[0]})"
    )

    valid_indexes = []
    for idx, v in enumerate(date_array[start_from:]):

        t_idx = idx + start_from
        if v:
            t_ = datetime.datetime.strptime(str(v[0]), "%Y-%m-%d %H:%M:%S")
            if start_date <= t_ <= end_date:
                valid_indexes.append(t_idx)

    min_idx = min(valid_indexes) if valid_indexes else None
    max_idx = max(valid_indexes) if valid_indexes else None

    logger.info(f"Min idx found {min_idx}")
    logger.info(f"Max idx found {max_idx}")

    return valid_indexes
