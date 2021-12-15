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


def get_model_memory_usage(batch_size, model):
    """

    based on batch size, find estimated size of model in GPU memory

    Arguments:
        batch_size {int} -- size of batches used for training
        model {keras.Model} -- Model of NN from Keras

    Returns:
        int -- estimated size in Gigabytes
    """
    import numpy as np
    import tensorflow.keras.backend as K

    # gather the number of parameters
    trainable_count = np.sum([K.count_params(w) for w in model.trainable_weights])
    non_trainable_count = np.sum(
        [K.count_params(w) for w in model.non_trainable_weights]
    )

    # gather shapes of inputs
    shapes_mem_count = 0
    for lay in model.layers:
        single_layer_mem = 1
        for s in lay.output_shape:
            if s is None or isinstance(s, tuple):
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    # based on 4 byte for int32, calculate size
    total_memory = (
        4.0 * batch_size * (shapes_mem_count + trainable_count + non_trainable_count)
    )
    gbytes = np.round(total_memory / (1024.0 ** 3), 3)
    return gbytes
