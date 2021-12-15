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

from makaniino.learning.models.unet import unet
from makaniino.ml_model import MLModel

logger = logging.getLogger("trans_learn_model")


class UNET_Model(MLModel):
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

        super(UNET_Model, self).__init__(config)

    def build(self):
        """
        Build the transfer-learning model
        """

        self._model = unet(
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
