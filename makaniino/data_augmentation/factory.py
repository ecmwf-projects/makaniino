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


from makaniino.data_augmentation.simple import (
    DataAugmenter_Simple,
    DataAugmenter_Simple64,
)
from makaniino.data_augmentation.window import (
    DataAugmenter_Window,
    DataAugmenter_Window64,
    DataAugmenter_Window64_Rot,
    DataAugmenter_Window_Rot,
)
from makaniino.data_augmentation.window_neg import (
    DataAugmenter_WindowNeg,
    DataAugmenter_WindowNeg64,
    DataAugmenter_WindowNeg64_Rot,
    DataAugmenter_WindowNeg_Rot,
)
from makaniino.data_augmentation.window_neg_grid import (
    DataAugmenter_WindowNeg_GRID,
    DataAugmenter_WindowNeg_GRID64,
)

data_augmenter_factory = {
    # single crop around each makaniino
    "simple": DataAugmenter_Simple,
    "simple64": DataAugmenter_Simple64,
    # multiple crops in a window around each makaniino
    # (rot adds a random rotation of the crops)
    "window": DataAugmenter_Window,
    "window_rot": DataAugmenter_Window_Rot,
    "window64": DataAugmenter_Window64,
    "window64_rot": DataAugmenter_Window64_Rot,
    # multiple crops in a window around each makaniino
    # and far away (to have some negative crops)
    # (rot adds a random rotation of the crops)
    "window_neg": DataAugmenter_WindowNeg,
    "window_neg_rot": DataAugmenter_WindowNeg_Rot,
    "window_neg64": DataAugmenter_WindowNeg64,
    "window_neg64_rot": DataAugmenter_WindowNeg64_Rot,
    "window_neg_grid": DataAugmenter_WindowNeg_GRID,
    "window_neg_grid64": DataAugmenter_WindowNeg_GRID64,
}
