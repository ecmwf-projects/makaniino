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

from makaniino.learning.models.unet import conv2d_block
from makaniino.ml_model import MLModel

logger = logging.getLogger("trans_learn_model")


class UNET_1HOT(MLModel):
    """
    Class that loads a model trained with a specific grid resolution
    and adds additional layers for transfer learning
    """

    default_params = copy.deepcopy(MLModel.default_params)
    default_params.update(
        {
            "img_rows": (320, "Input img rows"),
            "img_cols": (704, "Input img cols"),
            "channels": (3, "Input img channels"),
            "output_channels": (7, "Output channels"),
            "n_filters": (32, "Number of filters"),
            "activation": ("relu", "Activation function"),
            "final_activation": ("relu", "Final activation function"),
            "init": ("he_normal", "Init algorithm"),
            "depth": (3, "NN depth"),
            "dropout": (0, "Dropout"),
            "dropout_rate": (0.1, "Dropout rate"),
            "up_block": (3, "Up-sampling block"),
            "noise": (0, "Add noise"),
            "noise_rate": (0.1, "Noise rate"),
            "batchnorm": (1, "Batch normalization"),
            "fixed": (0, "Fixed"),
            "resnet": (0, "Use resnet"),
            "verbose": (0, "Verbose"),
        }
    )

    def __init__(self, config=None):

        super(UNET_1HOT, self).__init__(config)

    def build(self):
        """
        Build the transfer-learning model
        """

        self._model = self._unet_mc(
            img_rows=self.params.img_rows,
            img_cols=self.params.img_cols,
            channels=self.params.channels,
            activation=self.params.activation,
            final_activation=self.params.final_activation,
            fixed=self.params.fixed,
            batchnorm=self.params.batchnorm,
            output_channels=self.params.output_channels,
            resnet=self.params.resnet,
            n_filters=self.params.n_filters,
            depth=self.params.depth,
            dropout=self.params.dropout,
            dropout_rate=self.params.dropout_rate,
            noise=self.params.noise,
            noise_rate=self.params.noise_rate,
            verbose=self.params.verbose,
        )

    def _unet_mc(
        self,
        img_rows=64,
        img_cols=64,
        channels=1,
        output_channels=6,  # N of TC classes to classify
        n_filters=32,
        activation="relu",
        final_activation="softmax",  # softmax (=> multi-class prob)
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

        if verbose > 0:
            logger.info(
                "Creating Unet network with rows: %s "
                "cols: %s "
                "channels: %s "
                "output: %s",
                img_rows,
                img_cols,
                channels,
                output_channels,
            )
            logger.info(
                "   -- n_filters: %s  " "activation: %s " " final_activation: %s",
                n_filters,
                activation,
                final_activation,
            )
            logger.info(
                "   -- kernel init: %s  " "batchnorm: %s  " "fixed: %s  " "depth: %s",
                init,
                batchnorm,
                fixed,
                depth,
            )

            logger.info(
                "   -- dropout: %s "
                " dropout_rate: %s"
                " noise: %s  "
                "noise_rate: %s",
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
                        "creating constricting layer %s" " with filters %s",
                        (d + 1),
                        (n_filters * multiplier),
                    )
                if d == (depth - 1):
                    logger.info(
                        "creating center %s " "with filters: %s",
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
                    "creating expanding layer %s " "with filters: %s",
                    (depth - 1 - d),
                    (n_filters * multiplier),
                )

            x = UpSampling2D((2, 2))(last_layer)
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
