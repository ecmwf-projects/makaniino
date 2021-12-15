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
import os

import keras

from makaniino.training.trainer_base import ConvergenceLogger, Trainer

logger = logging.getLogger(__name__)


class TrainerSerial(Trainer):
    """
    A serial trainer (mainly to be use for small tests..)
    """

    default_params = copy.deepcopy(Trainer.default_params)
    default_params.update(
        {
            "early_stop_patience": (
                10,
                "# epochs to keep running with convergence not going down",
            ),
            "early_stop_restore_weights": (
                1,
                "restore weights at lowest convergence step",
            ),
        }
    )

    def __init__(self, config):

        super(TrainerSerial, self).__init__(config)

    def _do_train(self, model, data_handler):

        # keras model to train
        keras_model = model.model

        # selected loss
        loss = self.losses_avail[self.params.loss_function]

        # optimizer
        if self.params.optimizer == "adam":
            opt = self.optimizers_avail[self.params.optimizer](
                lr=self.params.learning_rate
            )
        else:
            opt = self.params.optimizer()

        # compile the model
        keras_model.compile(
            optimizer=opt, loss=loss, metrics=self.get_monitored_metrics()
        )

        # for ll, lay in enumerate(keras_model.layers):
        #     for ww, w in enumerate(lay.get_weights()):
        #         print(f" lay {ll} weights {ww} shape: {w}")

        # callbacks
        callbacks = [
            ConvergenceLogger(),
            keras.callbacks.EarlyStopping(
                patience=self.params.early_stop_patience,
                restore_best_weights=self.params.early_stop_restore_weights,
            ),
            keras.callbacks.ModelCheckpoint(
                os.path.join(
                    self.params.checkpoint_dir,
                    self.params.tag + "_ckpt-{epoch:02d}-{val_loss:.3f}.h5",
                ),
                monitor="val_loss",
                save_best_only=True,
                save_weights_only=False,
            ),
            keras.callbacks.TensorBoard(log_dir=self.params.tb_save_dir),
        ]

        # run the training loop
        self.history = keras_model.fit(
            data_handler.generator_train,
            epochs=self.params.epochs,
            callbacks=callbacks,
            steps_per_epoch=self.training_steps,
            validation_data=data_handler.generator_val,
            validation_steps=self.validation_steps,
            workers=self.params.workers,
            max_queue_size=self.params.max_queue_size,
            shuffle=self.params.shuffle_keras,
        )

        # save model/weights/images
        self.save_model()
        self.save_weights()
        self.save_model_and_weights()

        # plot history
        if self.history.history:
            self.save_plots()
