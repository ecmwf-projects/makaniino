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

import horovod.keras as hvd
import keras
import tensorflow as tf
from keras.callbacks import EarlyStopping

from makaniino.training.trainer_base import ConvergenceLogger, Trainer

logger = logging.getLogger(__name__)


class ShuffleTrainingData:
    """
    Base class to command a reshuffling
    during training
    """

    def __init__(self, data_handler):
        self.data_handler = data_handler
        super().__init__()

    def _reshuffle(self, epoch):

        logger.info("Reshuffling data..")
        epoch_as_seed = epoch
        self.data_handler.distributed_reshuffle_train_data(seed=epoch_as_seed)
        self.data_handler.distributed_reshuffle_validation_data(seed=epoch_as_seed)
        self.data_handler.distributed_reshuffle_test_data(seed=epoch_as_seed)


class InitShufflingCallback(keras.callbacks.Callback, ShuffleTrainingData):
    """
    Re-shuffle the data at the beginning of training
    """

    def __init__(self, data_handler):
        self.data_handler = data_handler
        keras.callbacks.Callback.__init__(self)
        ShuffleTrainingData.__init__(self, data_handler)

    def on_train_begin(self, logs=None):
        self._reshuffle(0)


class EpochShufflingCallback(keras.callbacks.Callback, ShuffleTrainingData):
    """
    Re-shuffle the data at the beginning of each epoch
    """

    def __init__(self, data_handler):
        self.data_handler = data_handler
        keras.callbacks.Callback.__init__(self)
        ShuffleTrainingData.__init__(self, data_handler)

    def on_epoch_begin(self, epoch, logs=None):
        self._reshuffle(epoch)


class TrainerParallel(Trainer):
    """
    A serial trainer
    """

    default_params = copy.deepcopy(Trainer.default_params)
    default_params.update(
        {
            "cpu_only": (0, "CPU only computation"),
            "early_stop": (1, "Early stop flag"),
            "learning_rate_schedule": (
                "15,40,1.0;40,70,0.1;70,100,0.01;100,,0.001",
                "Learning rate schedule, format: '<epoch_start>,<epoch_end>,<multiplier>;...' ",
            ),
            "learning_rate_warmup": (
                1,
                "Warm-up initial time to find best learning rate",
            ),
            "shuffle_every_epoch": (0, "Reshuffle at the end of each epoch"),
        }
    )

    def __init__(self, config):

        super(TrainerParallel, self).__init__(config)

        hvd.init()
        print(
            f"horovod size {hvd.size()}, rank {hvd.rank()}, dev rank {hvd.local_rank()}"
        )

        if self.params.cpu_only:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
            cpus = tf.config.list_physical_devices("CPU")
            tf.config.experimental.set_visible_devices(cpus[0], "CPU")
        else:
            gpus = tf.config.experimental.list_physical_devices("GPU")
            print(f"GPUS available {gpus}")
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            if gpus:
                print("GPUS in HVD node {hvd.local_rank()} => {gpus}")
                tf.config.experimental.set_visible_devices(
                    gpus[hvd.local_rank()], "GPU"
                )

    def _do_train(self, model, data_handler):

        # define level of verbosity
        if hvd.rank() == 0:
            verbose = 2
        else:
            verbose = self.params.verbose

        # keras model to train
        keras_model = model.model

        # selected loss
        loss = self.losses_avail[self.params.loss_function]

        # optimizer
        opt = hvd.DistributedOptimizer(
            self.optimizers_avail[self.params.optimizer](lr=self.params.learning_rate * hvd.size())
        )

        # compile the model
        keras_model.compile(
            optimizer=opt, loss=loss, metrics=self.get_monitored_metrics()
        )

        # define callbacks
        # noinspection PyListCreation
        callbacks = []

        callbacks.append(hvd.callbacks.BroadcastGlobalVariablesCallback(0))
        callbacks.append(hvd.callbacks.MetricAverageCallback())

        if self.params.learning_rate_warmup:
            callbacks.append(
                hvd.callbacks.LearningRateWarmupCallback(
                    self.params.learning_rate * hvd.size(), warmup_epochs=5, verbose=verbose
                )
            )

        if self.params.early_stop:
            callbacks.append(
                EarlyStopping(
                    monitor="val_loss",
                    patience=30,
                    verbose=verbose,
                    min_delta=1e-4,
                    restore_best_weights=True,
                )
            )

        # shuffle every epoch
        if self.params.shuffle_every_epoch:
            callbacks.append(EpochShufflingCallback(data_handler))

        # Save checkpoints only on the first worker to prevent other workers from corrupting them.
        if hvd.rank() == 0:

            ckpt_path = os.path.join(
                self.params.checkpoint_dir,
                self.params.tag + "_checkpoint-{epoch:02d}-{val_loss:.6f}.hdf5",
            )
            callbacks.append(
                keras.callbacks.ModelCheckpoint(
                    ckpt_path,
                    monitor="val_loss",
                    save_best_only=True,
                    save_weights_only=False,
                )
            )

            # callbacks.append(keras.callbacks.TensorBoard(log_dir=self.params.tb_save_dir))

            # log to file
            callbacks.append(ConvergenceLogger())

        # learning rate schedule callbacks
        lr_callbacks = self._learning_schedule_callbacks(
            self.params.learning_rate * hvd.size(), self.params.learning_rate_schedule
        )
        callbacks.extend(lr_callbacks)

        # Shuffle and distribute batches to each node
        data_handler.distributed_reshuffle_train_data(
            seed=0, populate_cache=data_handler.params.cache_data
        )
        data_handler.distributed_reshuffle_validation_data(
            seed=0, populate_cache=data_handler.params.cache_data
        )

        # determine the correct number of simulation steps
        self.training_steps = self._max_steps(
            len(data_handler.generator_train), self.params.steps_per_epoch
        )
        self.validation_steps = self._max_steps(
            len(data_handler.generator_val), self.params.val_steps_per_epoch
        )

        # run the training loop
        self.history = keras_model.fit(
            data_handler.generator_train,
            epochs=self.params.epochs,
            steps_per_epoch=self.training_steps,
            validation_data=data_handler.generator_val,
            validation_steps=self.validation_steps,
            callbacks=callbacks,
            workers=self.params.workers,
            max_queue_size=self.params.max_queue_size,
            shuffle=self.params.shuffle_keras,
            verbose=verbose,
        )

        if hvd.rank() == 0:
            print("Training finished!")

        # save accuracy/loss convergence plots
        if hvd.rank() == 0:
            self.save_plots()

        #hvd.allreduce([0], name="Barrier")

    @staticmethod
    def _learning_schedule_callbacks(lr_init, user_string):
        """
        Decode user learning schedule string
        and prepare the callbacks
        """

        lr_segments = [
            [float(s) if s else -1 for s in seg.split(",")]
            for seg in user_string.split(";")
        ]
        logger.info(f"Learning rate schedule {lr_segments}")

        lr_callbacks = []
        for segment in lr_segments:

            start_epoch = int(segment[0])
            end_epoch = int(segment[1])
            lr_value = float(segment[2])

            assert start_epoch != -1 or end_epoch != -1, (
                "Learning rate schedule: "
                "either start or end epoch must be specified!"
            )

            if start_epoch == -1:
                _callback = hvd.callbacks.LearningRateScheduleCallback(
                    lr_init, end_epoch=end_epoch, multiplier=lr_value
                )
            elif end_epoch == -1:
                _callback = hvd.callbacks.LearningRateScheduleCallback(
                    lr_init, start_epoch=start_epoch, multiplier=lr_value
                )
            else:
                _callback = hvd.callbacks.LearningRateScheduleCallback(
                    lr_init, start_epoch=start_epoch, end_epoch=end_epoch, multiplier=lr_value
                )

            lr_callbacks.append(_callback)

        return lr_callbacks
