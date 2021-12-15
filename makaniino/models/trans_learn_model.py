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
import json
import logging
import os

import keras

from makaniino.editable_model import EditableModel
from makaniino.learning.models.unet import conv2d_block
from makaniino.ml_model import MLModel

logger = logging.getLogger("trans_learn_model")


class TRANS_LEARN_Model(MLModel):
    """
    Class that loads a model trained with a specific grid resolution
    and adds additional layers for transfer learning
    """

    default_params = copy.deepcopy(MLModel.default_params)
    default_params.update(
        {
            "img_rows": (640, "Input img rows"),
            "img_cols": (1408, "Input img cols"),
            "channels": (1, "Input img channels"),
            "output_channels": (1, "Output channels"),
            "n_filters": (32, "Number of filters"),
            "activation": ("relu", "Activation function"),
            "final_activation": ("sigmoid", "Final activation function"),
            "init": ("he_normal", "Init algorithm"),
            "depth": (1, "NN depth"),
            "dropout": (0, "Dropout"),
            "dropout_rate": (0.1, "Dropout rate"),
            "up_block": (3, "Up-sampling block"),
            "noise": (0, "Add noise"),
            "noise_rate": (0.1, "Noise rate"),
            "batchnorm": (1, "Batch normalization"),
            "fixed": (0, "Fixed"),
            "resnet": (0, "Use resnet"),
            "verbose": (0, "Verbose"),
            "pre_trained_name": ("test0", "Root name of the pre-trained model"),
            "pretrain_model_dir": ("/var/tmp/cyclone_input", "Input directory"),
        }
    )

    def __init__(self, config=None):

        super(TRANS_LEARN_Model, self).__init__(config)

        self.pretrained_core_model = None

    def build(self):
        """
        Build the transfer-learning model
        """

        model_path = os.path.join(self.params.pretrain_model_dir, "models")
        model_file = os.path.join(model_path, self.params.pre_trained_name + ".json")

        weights_path = os.path.join(self.params.pretrain_model_dir, "weights")
        weights_file = os.path.join(weights_path, self.params.pre_trained_name + ".h5")

        # load pretrained model JSON
        pretrained_model_json = self._load_pretrained_json(model_file)

        # edit the pre-trained model as needed
        self.pretrained_core_model = self._edit_pretrained_model(
            pretrained_model_json, weights_file
        )

        logger.info("\n\n**** PRE-TRAINED EDITED MODEL ****\n\n")
        self.pretrained_core_model.summary()

        inputs = keras.layers.Input(
            (self.params.img_rows, self.params.img_cols, self.params.channels)
        )

        # initialization
        multiplier = 1
        last_layer = inputs
        layers = []

        # contraction part
        for d in range(0, self.params.depth):
            if d < (self.params.depth - 1):
                logger.info(
                    "creating constricting layer %s with filters %s",
                    (d + 1),
                    (self.params.n_filters * multiplier),
                )
            if d == (self.params.depth - 1):
                logger.info(
                    "creating center %s with filters: %s",
                    (d + 1),
                    (self.params.n_filters * multiplier),
                )

            x = conv2d_block(
                last_layer,
                n_filters=self.params.n_filters * multiplier,
                activation=self.params.activation,
                kernel_init=self.params.init,
                depth=2,
                batchnorm=self.params.batchnorm,
                resnet=self.params.resnet,
            )

            layers.append(x)
            x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(x)
            if self.params.dropout:
                x = keras.layers.Dropout(self.params.dropout_rate)(x)
            if self.params.noise:
                x = keras.layers.GaussianNoise(self.params.noise_rate)(x)

            if not self.params.fixed:
                multiplier = multiplier * 2

            last_layer = x

        for lay in layers:
            print(lay._name)
            if lay._name:
                lay._name = lay._name + "_contraction"

        # here we insert the pre-trained core model
        last_layer = self.pretrained_core_model(last_layer)

        # expansion part
        for d in range(0, self.params.depth):
            if not self.params.fixed:
                multiplier = int(multiplier / 2)

            logger.info(
                "creating expanding layer %s with filters: %s",
                (self.params.depth - 1 - d),
                (self.params.n_filters * multiplier),
            )

            x = keras.layers.UpSampling2D((2, 2))(last_layer)
            x = keras.layers.concatenate([layers.pop(), x], axis=3)
            if self.params.dropout:
                x = keras.layers.Dropout(self.params.dropout_rate)(x)
            if self.params.noise:
                x = keras.layers.GaussianNoise(self.params.noise_rate)(x)

            x = conv2d_block(
                x,
                n_filters=self.params.n_filters * multiplier,
                activation=self.params.activation,
                kernel_init=self.params.init,
                depth=self.params.up_block,
                batchnorm=self.params.batchnorm,
                resnet=self.params.resnet,
            )

            last_layer = x

        # close the expansion branch
        if self.params.final_activation == "leaky":
            x = keras.layers.Conv2D(self.params.output_channels, (1, 1))(last_layer)
            classify = keras.layers.LeakyReLU(alpha=0.2)(x)
        else:
            classify = keras.layers.Conv2D(
                self.params.output_channels,
                (1, 1),
                activation=self.params.final_activation,
            )(last_layer)

        self._model = keras.models.Model(
            inputs=inputs, outputs=classify, name="HRES_model"
        )

    @staticmethod
    def _load_pretrained_json(model_file):
        """
        Load and the pre-trained model json
        """

        # read pre-trained model
        logger.info(f"Loading model from file: {model_file}")
        with open(model_file, "r") as json_file:
            loaded_model_json = json.load(json_file)

        return loaded_model_json

    @staticmethod
    def _edit_pretrained_model(pretrained_model_json, weights_file):
        """
        Edits the pretrained model
        """

        # init an editable model - ready to be edited..
        pretrained_core_model = EditableModel(pretrained_model_json)

        # reshape input
        pretrained_core_model.layer("input_1").set_config(
            "batch_input_shape", (None, 320, 704, 32)
        )

        # remove the next 2 layers
        pretrained_core_model.remove_layers(["conv2d_1", "activation_1"])

        # set input as conv2d source
        pretrained_core_model.layer("conv2d_2").set_source_layers(["input_1"])

        # remove the last 2 layers before output
        pretrained_core_model.remove_layers(["conv2d_27", "activation_27", "conv2d_28"])

        # re-set the output layer
        pretrained_core_model.set_output_layer("conv2d_26")

        # print( json.dumps(json.loads(transfer_model.to_string()), indent=4) )
        pretrained_core_model_keras = keras.models.model_from_json(
            pretrained_core_model.to_string()
        )
        pretrained_core_model_keras.load_weights(
            weights_file, by_name=True, skip_mismatch=True
        )

        # make core-model weights not trainable
        for layer in pretrained_core_model_keras.layers:
            layer.trainable = False

        return pretrained_core_model_keras
