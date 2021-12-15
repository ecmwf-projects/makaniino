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

from keras import regularizers
from keras.layers import (  # noqa: F401
    Activation,
    Add,
    BatchNormalization,
    Conv2D,
    Conv2DTranspose,
    Dense,
    Dropout,
    GaussianNoise,
    Input,
    LeakyReLU,
    MaxPooling2D,
    UpSampling2D,
    concatenate,
)
from keras.models import Model

L2_WEIGHT_DECAY = 1e-4
BATCH_NORM_DECAY = 0.9
BATCH_NORM_EPSILON = 1e-5

logging_format = "%(asctime)s - %(name)s - %(message)s"
logging.root.setLevel(logging.INFO)
logging.basicConfig(
    level=logging.INFO, format=logging_format, datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("UNet")


def conv2d_block(
    input_tensor,
    n_filters,
    kernel_size=3,
    activation="relu",
    kernel_init="he_normal",
    depth=2,
    batchnorm=True,
    resnet=False,
):
    """Function to add 2 convolutional layers with the parameters passed to it"""
    x = input_tensor
    for d in range(0, depth):
        # layer
        x = Conv2D(
            filters=n_filters,
            kernel_size=(kernel_size, kernel_size),
            kernel_initializer=kernel_init,
            padding="same",
        )(x)
        if batchnorm:
            x = BatchNormalization()(x)
        if d is not depth - 1:
            if activation == "leaky":
                x = LeakyReLU(alpha=0.2)(x)
            else:
                x = Activation(activation)(x)

    if resnet:
        res = Conv2D(
            filters=n_filters,
            kernel_size=(1, 1),
            strides=(1, 1),
            use_bias=False,
            kernel_initializer="he_normal",
            kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
        )(input_tensor)

        # x = concatenate([shortcut, x], axis=3)
        x = Add()([x, res])

    if activation == "leaky":
        x = LeakyReLU(alpha=0.2)(x)
    else:
        x = Activation(activation)(x)

    return x


def unet(
    img_rows=64,
    img_cols=64,
    channels=1,
    output_channels=1,
    n_filters=32,
    activation="relu",
    final_activation="sigmoid",
    init="he_normal",
    depth=6,
    dropout=False,
    dropout_rate=0.1,
    up_block=3,
    noise=False,
    noise_rate=0.1,
    batchnorm=True,
    fixed=False,
    resnet=False,
    verbose=0,
    conv2d=False,
):
    """
    unet -- create unet model
    img_rows = height of input
    img_cols = width of input
    channels = depth of input
    output_channels = depth of output (labels)
    n_filters = number of filters for each conv layer
    activation = activation of each leayer
    final_activation = final activation
    init = kernel_initializer
    up_block = carvana unet uses 3 conv layers on expansion path,  others use 2
    depth = depth of unet
    batchnorm = use batch normalization
    dropout, dropout_rate = use dropout after conv with dropout_rate
    noise, noise_rate = use gaussian noise after conv with noise rate
    fixed = use fixed filters for each conv or not
    verbose = if greater than 0 print output


    """
    # n_fitlers
    # carvana
    # unet_256 = 32, depth = 6, up_block=3, batchnorm=True, noise=False, dropout=False
    # unet_512 = 16, depth = 7, up_block=3, batchnorm=True, noise=False, dropout=False
    # unet_1024 = 8, depth = 8, up_block=3, batchnorm=True, noise=False, dropout=False
    # orig unet
    # unet_256 = 32, depth = 5, up_block=2, batchnorm=False, noise=False, dropout=False
    # unet_512 = 16, depth = 6, up_block=2, batchnorm=False, noise=False, dropout=False
    # unet_1024 = 8, depth = 7, up_block=2, batchnorm=False, noise=False, dropout=False

    if verbose > 0:
        logger.info(
            "Creating Unet network with rows: %s cols: %s channels: %s output: %s",
            img_rows,
            img_cols,
            channels,
            output_channels,
        )
        logger.info(
            "   -- n_filters: %s  activation: %s  final_activation: %s",
            n_filters,
            activation,
            final_activation,
        )
        logger.info(
            "   -- kernel init: %s  batchnorm: %s  fixed: %s  depth: %s",
            init,
            batchnorm,
            fixed,
            depth,
        )
        logger.info(
            "   -- dropout: %s  dropout_rate: %s  noise: %s  noise_rate: %s",
            dropout,
            dropout_rate,
            noise,
            noise_rate,
        )
        logger.info("   -- resnet: %s", resnet)

    inputs = Input((img_rows, img_cols, channels))
    multiplier = 1

    last_layer = inputs

    layers = []

    if verbose:
        logger.info("creating contracting path")
    for d in range(0, depth):
        if verbose:
            if d < (depth - 1):
                logger.info(
                    "creating constricting layer %s with filters %s",
                    (d + 1),
                    (n_filters * multiplier),
                )
            if d == (depth - 1):
                logger.info(
                    "creating center %s with filters: %s",
                    (d + 1),
                    (n_filters * multiplier),
                )

        x = conv2d_block(
            last_layer,
            n_filters=n_filters * multiplier,
            activation=activation,
            kernel_init=init,
            depth=2,
            batchnorm=batchnorm,
            resnet=resnet,
        )
        # at center don't multiply, pool, dropout, or add noise
        if d < depth - 1:
            layers.append(x)
            x = MaxPooling2D((2, 2), strides=(2, 2))(x)
            if dropout:
                x = Dropout(dropout_rate)(x)
            if noise:
                x = GaussianNoise(noise_rate)(x)

            if not fixed:
                multiplier = multiplier * 2

        last_layer = x

    if verbose:
        logger.info("creating expansive path")

    for d in range(0, depth - 1):
        if not fixed:
            multiplier = int(multiplier / 2)

        if verbose:
            logger.info(
                "creating expanding layer %s with filters: %s",
                (depth - 1 - d),
                (n_filters * multiplier),
            )

        # --------------- testing -------------
        if conv2d:
            x = Conv2DTranspose(1, (2, 2), strides=(2, 2), padding="same")(last_layer)
        else:
            x = UpSampling2D((2, 2))(last_layer)
        # -------------------------------------

        x = concatenate([layers.pop(), x], axis=3)
        if dropout:
            x = Dropout(dropout_rate)(x)
        if noise:
            x = GaussianNoise(noise_rate)(x)

        x = conv2d_block(
            x,
            n_filters=n_filters * multiplier,
            activation=activation,
            kernel_init=init,
            depth=up_block,
            batchnorm=batchnorm,
            resnet=resnet,
        )

        last_layer = x

    if final_activation == "leaky":
        x = Conv2D(output_channels, (1, 1))(last_layer)
        classify = LeakyReLU(alpha=0.2)(x)
    else:
        classify = Conv2D(output_channels, (1, 1), activation=final_activation)(
            last_layer
        )

    model = Model(inputs=inputs, outputs=classify)

    return model


def half_unet(
    img_rows=64,
    img_cols=64,
    channels=1,
    output_channels=1,
    n_filters=32,
    activation="relu",
    final_activation="sigmoid",
    init="he_normal",
    depth=6,
    dropout=False,
    dropout_rate=0.1,
    up_block=3,
    noise=False,
    noise_rate=0.1,
    batchnorm=True,
    fixed=False,
    resnet=False,
    verbose=0,
):
    # n_fitlers
    # carvana
    # unet_256 = 32, depth = 6, up_block=3, batchnorm=True, noise=False, dropout=False
    # unet_512 = 16, depth = 7, up_block=3, batchnorm=True, noise=False, dropout=False
    # unet_1024 = 8, depth = 8, up_block=3, batchnorm=True, noise=False, dropout=False
    # orig unet
    # unet_256 = 32, depth = 5, up_block=2, batchnorm=False, noise=False, dropout=False
    # unet_512 = 16, depth = 6, up_block=2, batchnorm=False, noise=False, dropout=False
    # unet_1024 = 8, depth = 7, up_block=2, batchnorm=False, noise=False, dropout=False

    if verbose > 0:
        logger.info(
            "Creating Half Unet network with rows: %s cols: %s channels: %s output: %s",
            img_rows,
            img_cols,
            channels,
            output_channels,
        )
        logger.info(
            "   -- n_filters: %s  activation: %s  final_activation: %s",
            n_filters,
            activation,
            final_activation,
        )
        logger.info(
            "   -- kernel init: %s  batchnorm: %s  fixed: %s  depth: %s",
            init,
            batchnorm,
            fixed,
            depth,
        )
        logger.info(
            "   -- dropout: %s  dropout_rate: %s  noise: %s  noise_rate: %s",
            dropout,
            dropout_rate,
            noise,
            noise_rate,
        )
        logger.info("   -- resnet: %s", resnet)

    inputs = Input((img_rows, img_cols, channels))
    multiplier = 1

    last_layer = inputs

    layers = []

    if verbose:
        logger.info("creating contracting path")
    for d in range(0, depth):
        if verbose:
            if d < (depth - 1):
                logger.info(
                    "creating constricting layer %s with filters %s",
                    (d + 1),
                    (n_filters * multiplier),
                )
            if d == (depth - 1):
                logger.info(
                    "creating center %s with filters: %s",
                    (d + 1),
                    (n_filters * multiplier),
                )

        x = conv2d_block(
            last_layer,
            n_filters=n_filters * multiplier,
            activation=activation,
            kernel_init=init,
            depth=2,
            batchnorm=batchnorm,
            resnet=resnet,
        )
        # at center don't multiply, pool, dropout, or add noise
        if d < depth - 1:
            layers.append(x)
            x = MaxPooling2D((2, 2), strides=(2, 2))(x)
            if dropout:
                x = Dropout(dropout_rate)(x)
            if noise:
                x = GaussianNoise(noise_rate)(x)

            if not fixed:
                multiplier = multiplier * 2

        last_layer = x

    if verbose:
        logger.info("creating expansive path")

    classify = Conv2DTranspose(
        output_channels, kernel_size=(16, 16), strides=(16, 16), use_bias=False
    )(last_layer)

    model = Model(inputs=inputs, outputs=classify)

    return model
