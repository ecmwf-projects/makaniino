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

import tensorflow as tf

from makaniino.configurable import Configurable

logger = logging.getLogger(__name__)


class MLModel(Configurable):
    """
    Minimal model interface
    """

    default_params = {
        "name": ("N/A", "Model name"),
    }

    def __init__(self, config=None):

        # the internal keras model
        self._model = None

        super().__init__(config)

    def build(self):
        """
        Defines the architecture
        """
        raise NotImplementedError

    def compile(self):
        """
        Defines model params like loss, optimizer, etc..
        """
        raise NotImplementedError

    def train(self, trainer, data_gen):
        """
        Train the model according to a trainer
        """

        trainer.train_model(self, data_gen)

    def summary(self):
        """
        Print model summary
        """

        if not self._model:
            logger.error("Build the model first!")
            return None

        self._model.summary(print_fn=logger.info)

    def load_weights_from_file(self, f_name):
        """
        Load weights from file
        """
        self._model.load_weights(f_name)

    def predict(self, input):
        """
        Make a prediction
        """
        return self._model.predict(input)

    @property
    def model(self):

        if not self._model:
            logger.error("internal Keras model not defined!")
            raise RuntimeError

        return self._model

    @staticmethod
    def _set_gpu_memory_growth(flag):
        gpus = tf.config.experimental.list_physical_devices("GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, flag)
