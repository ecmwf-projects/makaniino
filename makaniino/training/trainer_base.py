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
import os

import keras
import matplotlib  # noqa: E402

matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import tensorflow as tf  # noqa: E402

from makaniino.configurable import Configurable  # noqa: E402
from makaniino.learning.losses.losses import (  # noqa: E402
    bce_dice_loss,
    dice_coeff,
    dice_loss,
    dice_loss_modified,
    focal_loss,
    tversky_coeff,
    tversky_loss,
)

logger = logging.getLogger(__name__)


class ConvergenceLogger(keras.callbacks.Callback):
    """
    A basic logger callback that logs
    convergence metrics
    """

    def on_epoch_end(self, epoch, logs=None):

        if isinstance(logs, dict):
            basic_metrics = (
                f"|========> EPOCH: {epoch}, "
                f"loss: {logs['loss']:.4f}, "
                f"val_loss: {logs['val_loss']:.4f}, "
                f"acc: {logs['accuracy']:.4f}, "
            )

            user_metrics = ", ".join(f"{k}: {v:.6f}" for k, v in logs.items())
        else:
            basic_metrics = ""
            user_metrics = ""

        # do the logging
        logger.info(basic_metrics + user_metrics)

    def on_batch_end(self, batch, logs=None):

        assert isinstance(logs, dict), "tensorflow logs not a dictionary!"

        logger.info(
            f"|---> batch: {batch}, "
            f"loss: {logs['loss']:.4f}, "
            f"acc: {logs['accuracy']:.4f}"
        )


class Trainer(Configurable):
    """
    Trains a ML model
    """

    losses_avail = {
        "tversky": tversky_loss(alpha=0.3, beta=0.7),
        "dice": dice_loss,
        "bce_dice": bce_dice_loss,
        "focal": focal_loss(gamma=2, alpha=0.6),
        "bce": "binary_crossentropy",
        "mse": "mse",
        "scc": keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        "dice_loss_modified": dice_loss_modified,
    }

    metrics_avail = {
        "tversky_coeff": tversky_coeff(alpha=0.3, beta=0.7),
        "dice_coeff": dice_coeff,
        "accuracy": "accuracy",
    }

    optimizers_avail = {
        "adam": tf.keras.optimizers.Adam,
        "rmsprop": tf.keras.optimizers.RMSprop,
    }

    default_params = {
        "tag": (None, "TAG required for this training session"),
        "verbose_trainer": (0, "Verbose"),
        "epochs": (5, "Number of epochs"),
        "steps_per_epoch": (
            -1,
            "Number of steps per epoch [-1 means all steps available]",
        ),
        "val_steps_per_epoch": (
            -1,
            "Number of validation steps per epoch [-1 means all steps available]",
        ),
        "workers": (2, " Training workers"),
        "max_queue_size": (8, "Max queue size of served data"),
        "shuffle_keras": (1, "Shuffle data within Keras 'fit' function"),
        "model_save_dir": (
            os.path.join(os.getcwd(), "models"),
            "Path to saved model JSON",
        ),
        "weights_save_dir": (
            os.path.join(os.getcwd(), "weights"),
            "Path to saved model weights",
        ),
        "checkpoint_dir": (
            os.path.join(os.getcwd(), "checkpoints"),
            "Path to saved model checkpoints",
        ),
        "images_save_dir": (
            os.path.join(os.getcwd(), "images"),
            "Path to saved model images",
        ),
        "tb_save_dir": (
            os.path.join(os.getcwd(), "tensorboard"),
            "Path to saved tensorboard logs",
        ),
        "metrics": (
            "tversky_coeff,dice_coeff,accuracy",
            f"metrics to monitor during training, "
            f"choices {','.join(str(k) for k in metrics_avail.keys())}",
        ),
        "loss_function": (
            "dice",
            f"Loss function that drives training, "
            f"choices {','.join(str(k) for k in losses_avail.keys())}",
        ),
        "learning_rate": (0.00001, "learning rate"),
        "optimizer": ("adam", f"Optimizer, choices [{optimizers_avail}]"),
    }

    def __init__(self, params):

        self.model = None

        super().__init__(params)

        self.history = None
        self.training_steps = None
        self.validation_steps = None

    def _check_user_config(self, configs):
        """
        It runs an additional check on the TAG which is
        required to run the trainer
        """

        super()._check_user_config(configs)

        # check that a TAG has been specified
        if not self.params.tag:
            logger.error("A training TAG must be specified!")
            raise RuntimeError

    @staticmethod
    def _max_steps(gen_max_steps, requested_steps):
        """
        Training/validation steps according to user request
        """

        if requested_steps != -1:
            assert 0 <= requested_steps <= gen_max_steps
            return requested_steps
        else:
            return gen_max_steps

    def _create_save_dirs(self):
        """
        Create save dirs..
        """

        # make sure the checkpoint dir exists
        if not os.path.isdir(self.params.checkpoint_dir):
            os.mkdir(self.params.checkpoint_dir)

        # make sure the model save dir exists
        if not os.path.isdir(self.params.model_save_dir):
            os.mkdir(self.params.model_save_dir)

        # make sure the weights save dir exists
        if not os.path.isdir(self.params.weights_save_dir):
            os.mkdir(self.params.weights_save_dir)

        # make sure the images save dir exists
        if not os.path.isdir(self.params.images_save_dir):
            os.mkdir(self.params.images_save_dir)

    def train_model(self, model, data_handler):
        """
        Do the training
        """
        self.model = model

        # create save dirs
        self._create_save_dirs()

        # summary of the data_handler
        logger.info(data_handler.summary())

        # do the training
        self._do_train(model, data_handler)

    def _do_train(self, model, data_gen):
        """
        Do the actual training
        """
        raise NotImplementedError

    def save_model(self):
        """
        Save model json
        """

        # save model and weights
        f_name = os.path.join(
            self.params.model_save_dir, f"{self.params.tag}.model.json"
        )
        with open(f_name, "w") as json_file:
            json_file.write(self.model.model.to_json())

        logger.info(f"Saved model to {f_name}")

    def save_weights(self):
        """
        Save model weights
        """

        f_name = os.path.join(
            self.params.weights_save_dir, f"{self.params.tag}.weights.h5"
        )
        self.model.model.save_weights(f_name)

        logger.info(f"Saved weights to {f_name}")

    def save_model_and_weights(self):
        """
        Save model weights
        """

        f_name = os.path.join(
            self.params.model_save_dir, f"{self.params.tag}.keras_model"
        )
        self.model.model.save(f_name)

        logger.info(f"Saved model+weights to {f_name}")

    def save_plots(self):
        """
        Save images of convergence
        """

        plt.figure(111)
        plt.plot(self.history.history["accuracy"])
        plt.plot(self.history.history["val_accuracy"])
        plt.title("model accuracy")
        plt.ylabel("accuracy")
        plt.xlabel("epoch")
        plt.legend(["train", "test"], loc="upper left")

        f_name = os.path.join(self.params.images_save_dir, self.params.tag + "_acc.png")
        plt.savefig(f_name)
        logger.info(f"Saved plot to {f_name}")
        plt.close()

        plt.figure(222)
        plt.plot(self.history.history["loss"])
        plt.plot(self.history.history["val_loss"])
        plt.title("model loss")
        plt.ylabel("loss")
        plt.xlabel("epoch")
        plt.legend(["train", "test"], loc="upper left")

        f_name = os.path.join(
            self.params.images_save_dir, self.params.tag + "_loss.png"
        )
        plt.savefig(f_name)
        logger.info(f"Saved plot to {f_name}")
        plt.close()

        # # model graph
        # f_name = os.path.join(self.params.images_save_dir, self.params.tag + "_graph.png")
        # keras.utils.plot_model(
        #     self.model.model,
        #     to_file=f_name,
        #     show_shapes=False,
        #     show_layer_names=True,
        #     rankdir='TB',
        #     expand_nested=False,
        #     dpi=96
        # )

    def get_monitored_metrics(self):
        """
        Get the list of selected monitored metrics
        """

        # selected metrics
        monitored_metrics = ["accuracy"]
        if self.params.metrics:
            monitored_metrics += [
                self.metrics_avail[m] for m in self.params.metrics.split(",")
            ]
        return list(set(monitored_metrics))
