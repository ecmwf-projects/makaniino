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
   losses.py

   collection of various losses for keras

   author Jebb Q Stewart (jebb.q.stewart@noaa.gov)
   Copyright 2019. Colorado State University. All rights reserved.

"""

import keras.backend as K
import tensorflow as tf
from keras.losses import binary_crossentropy


def dice_coeff(y_true, y_pred, smooth=1.0e-6):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    score = (2.0 * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return score


def dice_loss(y_true, y_pred, smooth=1.0e-6):
    loss = 1 - dice_coeff(y_true, y_pred)
    return loss


# =======================================================
def dice_loss_modified(y_true, y_pred, smooth=1.0e-6):

    _threshold = K.ones_like(y_true) * 0.1

    y_true_int = K.cast(K.greater(y_true, _threshold), "float32")
    y_pred_int = K.cast(K.greater(y_pred, _threshold), "float32")

    _dice_loss = 1.0 - dice_coeff(y_true_int, y_pred_int)
    _mse_loss = K.mean(K.square(y_true - y_pred))

    return (_dice_loss + _mse_loss) / 2


# =======================================================


def focal_loss(gamma=2.0, alpha=0.25):
    """
    Binary form of focal loss.
      FL(p_t) = -alpha * (1 - p_t)**gamma * log(p_t)
      where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.
    References:
        https://arxiv.org/pdf/1708.02002.pdf
    Usage:
     model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """

    def binary_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred:  A tensor resulting from a sigmoid
        :return: Output tensor.
        """
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

        epsilon = K.epsilon()
        # clip to prevent NaN's and Inf's
        pt_1 = K.clip(pt_1, epsilon, 1.0 - epsilon)
        pt_0 = K.clip(pt_0, epsilon, 1.0 - epsilon)

        return -K.sum(alpha * K.pow(1.0 - pt_1, gamma) * K.log(pt_1)) - K.sum(
            (1 - alpha) * K.pow(pt_0, gamma) * K.log(1.0 - pt_0)
        )

    return binary_focal_loss_fixed


def tversky_coeff(alpha=0.3, beta=0.7, smooth=1e-10):
    """Tversky coef function.

    Parameters
    ----------
    y_true : keras tensor
        tensor containing target mask.
    y_pred : keras tensor
        tensor containing predicted mask.
    alpha : float
        real value, weight of '0' class.
    beta : float
        real value, weight of '1' class.
    smooth : float
        small real value used for avoiding division by zero error.

    Returns
    -------
    keras tensor
        tensor containing tversky loss.
    """

    def tversky(y_true, y_pred):
        y_true = K.flatten(y_true)
        y_pred = K.flatten(y_pred)
        truepos = K.sum(y_true * y_pred)
        fp_and_fn = alpha * K.sum(y_pred * (1 - y_true)) + beta * K.sum(
            (1 - y_pred) * y_true
        )
        answer = (truepos + smooth) / ((truepos + smooth) + fp_and_fn)
        return answer

    return tversky


def tversky_loss(alpha=0.3, beta=0.7, smooth=1e-10):
    coeff = tversky_coeff(alpha=alpha, beta=beta)

    def tversky(y_true, y_pred):
        loss = 1 - coeff(y_true, y_pred)
        return loss

    return tversky


def bce_dice_loss(y_true, y_pred):
    loss = binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    return loss


def weighted_dice_coeff(y_true, y_pred, weight):
    smooth = 1.0
    w, m1, m2 = weight * weight, y_true, y_pred
    intersection = m1 * m2
    score = (2.0 * K.sum(w * intersection) + smooth) / (
        K.sum(w * m1) + K.sum(w * m2) + smooth
    )
    return score


def weighted_dice_loss(y_true, y_pred):
    y_true = K.cast(y_true, "float32")
    y_pred = K.cast(y_pred, "float32")
    # if we want to get same size of output, kernel size must be odd number
    if K.int_shape(y_pred)[1] == 128:
        kernel_size = 11
    elif K.int_shape(y_pred)[1] == 256:
        kernel_size = 21
    elif K.int_shape(y_pred)[1] == 512:
        kernel_size = 21
    elif K.int_shape(y_pred)[1] == 1024:
        kernel_size = 41
    else:
        raise ValueError("Unexpected image size")
    averaged_mask = K.pool2d(
        y_true,
        pool_size=(kernel_size, kernel_size),
        strides=(1, 1),
        padding="same",
        pool_mode="avg",
    )
    border = K.cast(K.greater(averaged_mask, 0.005), "float32") * K.cast(
        K.less(averaged_mask, 0.995), "float32"
    )
    weight = K.ones_like(averaged_mask)
    w0 = K.sum(weight)
    weight += border * 2
    w1 = K.sum(weight)
    weight *= w0 / w1
    loss = 1 - weighted_dice_coeff(y_true, y_pred, weight)
    return loss


def weighted_bce_loss(y_true, y_pred, weight):
    # avoiding overflow
    epsilon = 1e-7
    y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
    logit_y_pred = K.log(y_pred / (1.0 - y_pred))

    # https://www.tensorflow.org/api_docs/python/tf/nn/weighted_cross_entropy_with_logits
    loss = (1.0 - y_true) * logit_y_pred + (1.0 + (weight - 1.0) * y_true) * (
        K.log(1.0 + K.exp(-K.abs(logit_y_pred))) + K.maximum(-logit_y_pred, 0.0)
    )
    return K.sum(loss) / K.sum(weight)


def weighted_bce_dice_loss(y_true, y_pred):
    y_true = K.cast(y_true, "float32")
    y_pred = K.cast(y_pred, "float32")
    # if we want to get same size of output, kernel size must be odd number
    if K.int_shape(y_pred)[1] == 128:
        kernel_size = 11
    elif K.int_shape(y_pred)[1] == 256:
        kernel_size = 21
    elif K.int_shape(y_pred)[1] == 512:
        kernel_size = 21
    elif K.int_shape(y_pred)[1] == 1024:
        kernel_size = 41
    else:
        raise ValueError("Unexpected image size")
    averaged_mask = K.pool2d(
        y_true,
        pool_size=(kernel_size, kernel_size),
        strides=(1, 1),
        padding="same",
        pool_mode="avg",
    )
    border = K.cast(K.greater(averaged_mask, 0.005), "float32") * K.cast(
        K.less(averaged_mask, 0.995), "float32"
    )
    weight = K.ones_like(averaged_mask)
    w0 = K.sum(weight)
    weight += border * 2
    w1 = K.sum(weight)
    weight *= w0 / w1
    loss = weighted_bce_loss(y_true, y_pred, weight) + (
        1 - weighted_dice_coeff(y_true, y_pred, weight)
    )
    return loss


def jaccard_coef(y_true, y_pred, smooth=100):
    # __author__ = Vladimir Iglovikov
    intersection = K.sum(y_true * y_pred, axis=[0, -1, -2])
    sum_ = K.sum(y_true + y_pred, axis=[0, -1, -2])

    jac = (intersection + smooth) / (sum_ - intersection + smooth)

    return K.mean(jac)


def jaccard_loss(y_true, y_pred, smooth=100):
    loss = 1 - jaccard_coef(y_true, y_pred, smooth)
    return loss


def jaccard_coef_int(y_true, y_pred, smooth=100):
    # __author__ = Vladimir Iglovikov
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))

    intersection = K.sum(y_true * y_pred_pos, axis=[0, -1, -2])
    sum_ = K.sum(y_true + y_pred_pos, axis=[0, -1, -2])
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return K.mean(jac)
