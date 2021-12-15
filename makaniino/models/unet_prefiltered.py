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

from makaniino.learning.models.unet import unet
from makaniino.ml_model import MLModel
from makaniino.models.cnn_fixed_filter import CNN_Fixed_Filter_Model

logger = logging.getLogger("trans_learn_model")


class UNET_Prefiltered(MLModel):
    """
    UNET model with a filter layer in front
    """

    default_params = copy.deepcopy(MLModel.default_params)
    default_params.update(
        {
            "img_rows": (320, "Input img rows"),
            "img_cols": (704, "Input img cols"),
            "output_channels": (1, "Output channels"),
            "n_filters": (32, "Number of filters"),
            "activation": ("relu", "Activation function"),
            "final_activation": ("sigmoid", "Final activation function"),
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
        super(UNET_Prefiltered, self).__init__(config)

    def build(self):
        """
        Build the pre-filtered unet model
        """

        # build the filter first
        filter_model = CNN_Fixed_Filter_Model.from_params_dict(
            {
                "img_rows": self.params.img_rows,
                "img_cols": self.params.img_cols,
                "channels": 2,
                "output_channels": 1,
                "name": "pre-filter",
            }
        )
        filter_model.build()

        # U and V input fields
        inputs = keras.layers.Input((self.params.img_rows, self.params.img_cols, 2))
        filtered_layer = filter_model.model(inputs)

        # concatenate the inputs with the filtered layer
        inputs_and_filtered = keras.layers.concatenate(
            [inputs, filtered_layer], axis=-1
        )

        # build the unet model
        unet_model = unet(
            img_rows=self.params.img_rows,
            img_cols=self.params.img_cols,
            channels=3,
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

        # feed the unet with U,V and the filtered layer
        unet_output = unet_model(inputs_and_filtered)

        # the final model
        self._model = keras.models.Model(inputs=inputs, outputs=unet_output)
