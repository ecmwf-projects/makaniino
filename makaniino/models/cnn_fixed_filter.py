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

import copy
import logging

import keras
import numpy as np
import tensorflow as tf

from makaniino.ml_model import MLModel

logger = logging.getLogger("trans_learn_model")


class CycloneFilter(tf.keras.initializers.Initializer):
    def __init__(self):
        pass

    def __call__(self, shape, dtype=None):

        with tf.init_scope():
            x_filter_1 = tf.ones(shape=(shape[0] // 2, shape[1]), dtype=dtype)
            x_filter_0 = tf.zeros(shape=(shape[0] // 2, shape[1]), dtype=dtype)
            x_filter = tf.concat((x_filter_1, x_filter_0), axis=0)

            y_filter_1 = tf.ones(shape=(shape[0], shape[1] // 2), dtype=dtype)
            y_filter_0 = tf.zeros(shape=(shape[0], shape[1] // 2), dtype=dtype)
            y_filter = tf.concat((y_filter_0, y_filter_1), axis=1)

            return_filter = tf.stack((x_filter, y_filter), axis=-1)
            return_filter = tf.expand_dims(return_filter, axis=-1)

        return return_filter


class CurlFilter(tf.keras.initializers.Initializer):
    def __init__(self):
        pass

    def __call__(self, shape, dtype=None):

        print(f"shape {shape}")
        assert shape[:2] == (3, 3)

        with tf.init_scope():

            # vort = dv/dx - du/dy
            x_filter = tf.Variable([[0, -1, 0], [0, 0, 0], [0, +1, 0]], dtype=dtype)

            y_filter = tf.Variable([[0, 0, 0], [-1, 0, 1], [0, 0, 0]], dtype=dtype)

            return_filter = tf.stack((x_filter, y_filter), axis=-1)
            return_filter = tf.expand_dims(return_filter, axis=-1)

        return return_filter


class CurlLikeFilter(tf.keras.initializers.Initializer):
    def __init__(self):
        pass

    def __call__(self, shape, dtype=None):
        print(f"shape {shape}")
        assert shape[:2] == (3, 3)

        with tf.init_scope():

            x_filter = tf.Variable(
                [[-0.5, -1, -0.5], [0, 0, 0], [0.5, +1, 0.5]], dtype=dtype
            )

            y_filter = tf.Variable(
                [[-0.5, 0, 0.5], [-1, 0, 1], [-0.5, 0, 0.5]], dtype=dtype
            )

            return_filter = tf.stack((x_filter, y_filter), axis=-1)
            return_filter = tf.expand_dims(return_filter, axis=-1)

        return return_filter


class IrrotationalFilter(tf.keras.initializers.Initializer):
    def __init__(self, rot_mult_factor):

        self.rot_mult_factor = rot_mult_factor

    def __call__(self, shape, dtype=None):
        print(f"shape {shape}")
        np.set_printoptions(precision=2, threshold=np.inf)

        ctr = np.zeros(2)
        xfilter_np = np.zeros(shape=(shape[0], shape[1]))
        yfilter_np = np.zeros(shape=(shape[0], shape[1]))

        # let's normalize the kernel area from -1 to 1
        lx = 2
        ly = 2
        dx = lx / shape[1]
        dy = ly / shape[0]

        for i in range(shape[1]):
            for j in range(shape[0]):

                x = (-lx / 2 + (i * dx + dx / 2)) / lx
                y = (ly / 2 - (j * dy + dy / 2)) / ly

                coord = np.asarray((x, y))
                rad = coord - ctr

                rad_norm = np.linalg.norm(rad)
                rad_1 = rad / rad_norm

                # vel perpendicular to radius
                vel = np.asarray((rad_1[1], -rad_1[0])) * 1 / rad_norm

                # assign values to filter
                xfilter_np[j, i] = self.rot_mult_factor * vel[0]
                yfilter_np[j, i] = self.rot_mult_factor * vel[1]

        xfilter_np = (xfilter_np - np.min(xfilter_np)) / (
            np.max(xfilter_np) - np.min(xfilter_np)
        ) * 2 - 1
        yfilter_np = (yfilter_np - np.min(yfilter_np)) / (
            np.max(yfilter_np) - np.min(yfilter_np)
        ) * 2 - 1

        print(f"xfilter_np.shape \n{xfilter_np}")
        print(f"yfilter_np.shape \n{yfilter_np}")

        with tf.init_scope():

            x_filter = tf.Variable(xfilter_np, dtype=dtype)
            y_filter = tf.Variable(yfilter_np, dtype=dtype)

            return_filter = tf.stack((x_filter, y_filter), axis=-1)
            return_filter = tf.expand_dims(return_filter, axis=-1)

        return return_filter


class CNN_Fixed_Filter_Model(MLModel):
    """
    A simple model to experiment with filters
    """

    default_params = copy.deepcopy(MLModel.default_params)
    default_params.update(
        {
            "img_rows": (320, "Input img rows"),
            "img_cols": (704, "Input img cols"),
            "channels": (2, "Input img channels"),
            "output_channels": (1, "Output channels"),
        }
    )

    def __init__(self, config=None):
        super(CNN_Fixed_Filter_Model, self).__init__(config)

    def build(self):

        inputs = keras.layers.Input(
            (self.params.img_rows, self.params.img_cols, self.params.channels)
        )

        # ==== curl filter
        # filter_size = 3
        # ker_init_pos = ker_init_neg = CurlFilter()

        # ======== curl-like filter
        # filter_size = 3
        # ker_init_pos = ker_init_neg = CurlLikeFilter()

        # ======== makaniino filter
        # filter_size = 3
        # ker_init_pos = ker_init_neg = CycloneFilter()

        # ======== irrotational filter
        filter_size = 8
        ker_init_pos = IrrotationalFilter(1)
        ker_init_neg = IrrotationalFilter(-1)

        # layer to filter positive rotation
        lay_pos_rot = keras.layers.Conv2D(
            filters=1,
            kernel_size=(filter_size, filter_size),
            strides=(1, 1),
            use_bias=False,
            kernel_initializer=ker_init_pos,
            padding="same",
        )(inputs)

        lay_pos_rot.trainable = False

        lay_pos_rot = keras.layers.ReLU()(lay_pos_rot)

        # layer to filter negative rotation
        lay_neg_rot = keras.layers.Conv2D(
            filters=1,
            kernel_size=(filter_size, filter_size),
            strides=(1, 1),
            use_bias=False,
            kernel_initializer=ker_init_neg,
            padding="same",
        )(inputs)
        lay_neg_rot.trainable = False

        lay_neg_rot = keras.layers.ReLU()(lay_neg_rot)

        # sum of both layers
        lay = keras.layers.add([lay_pos_rot, lay_neg_rot])

        self._model = keras.models.Model(inputs=inputs, outputs=lay)
