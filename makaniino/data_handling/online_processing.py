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

logger = logging.getLogger(__file__)


class OnLineProcessor:
    """
    Class that processes data batches ready to be served
    by the data generator
    """

    def apply(self, train, test):
        raise NotImplementedError(
            f"apply not implemented in class {self.__class__.__name__}!"
        )


class EdgeTrimmer(OnLineProcessor):
    """
    Trim the edges of a batch of fields
    """

    def __init__(self, x_cut_pxl=20, y_cut_pxl=8):
        self.x_cut_pxl = x_cut_pxl
        self.y_cut_pxl = y_cut_pxl

    def apply(self, train, test):

        if self.x_cut_pxl == 0 and self.y_cut_pxl == 0:
            return train, test

        # resize and normalize data
        # NB -1 is added only to reduce from 361 to 360
        train = train[
            :,
            self.x_cut_pxl : train.shape[1] - self.x_cut_pxl - 1,
            self.y_cut_pxl : train.shape[2] - self.y_cut_pxl,
            :,
        ]

        test = test[
            :,
            self.x_cut_pxl : test.shape[1] - self.x_cut_pxl - 1,
            self.y_cut_pxl : test.shape[2] - self.y_cut_pxl,
            :,
        ]

        return train, test


class DataNormalizer(OnLineProcessor):
    """
    Normalize data upon a norm factor
    """

    def __init__(self, norm_factor="55.0", shift_factor=None):
        """
        Configured with a CSV string of norm factors
        """
        self.norm_factor = [float(fact) for fact in norm_factor.split(",") if fact]

        # if a shift factor is not passed, it is taken as equal to the norm factor
        if shift_factor is None:
            self.shift_factor = self.norm_factor
        else:
            self.shift_factor = [
                float(fact) for fact in shift_factor.split(",") if fact
            ]

        assert len(self.norm_factor) == len(self.shift_factor)

    def apply(self, train, test):

        # make sure that we have passed all the norm factors we need
        assert len(self.norm_factor) == train.shape[-1] or len(self.norm_factor) == 1

        # print(f"self.norm_factor {self.norm_factor}")
        # print(f"self.shift_factor {self.shift_factor}")

        # divide each channel for each norm factor
        train = (train - self.shift_factor) / self.norm_factor
        return train, test


class DataRecaster(OnLineProcessor):
    """
    Recast data
    """

    def __init__(self, x_type=None, y_type=np.float32):
        self.x_type = x_type
        self.y_type = y_type

    def apply(self, train, test):

        # normalize train
        train = train if not self.x_type else train.astype(self.x_type)
        test = test if not self.y_type else test.astype(self.y_type)

        return train, test


class NANSwapper(OnLineProcessor):
    """
    Substitute Nan data with a prescribed value
    """

    def __init__(self, nan_new_value=0):
        self.nan_new_value = nan_new_value

    def apply(self, train, test):

        # normalize train
        train = np.nan_to_num(train, nan=self.nan_new_value)

        return train, test


class BatchShuffler(OnLineProcessor):
    """
    Substitute Nan data with a prescribed value
    """

    def __init__(self):
        pass

    def apply(self, train, test):

        # shuffle train
        shuffled_idxs = np.random.permutation(train.shape[0])
        train = train[shuffled_idxs, :]
        test = test[shuffled_idxs, :]

        return train, test
