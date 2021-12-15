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

import collections
import json
import logging
import os

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger("debug_utils")


def find_layer_in_model(model, layer_name):
    """
    Find a layer in a model by name
    """

    layer = None
    for lay in model.model.layers:
        if lay.name == layer_name:
            layer = lay
            break
    if not layer:
        raise ValueError(f"layer {layer_name} not found!")

    return layer


def sample_summary(gen, save_path, nsamples):
    """
    Takes a few samples from the input generator
    and reports some generic statistics/plots of the samples
    """
    gen_iter = iter(gen)

    for isample in range(nsamples):

        logger.info(f"Processing sample: {isample}")

        # imgs is a tuple of x_train, y_train batches..
        (img_input_batch, img_label_batch) = next(gen_iter)

        logger.info(f"img_input_batch.shape {img_input_batch.shape}")
        logger.info(f"img_label_batch.shape {img_label_batch.shape}")

        for itime in range(img_input_batch.shape[-1]):

            logger.info(f"Processing time-frame: {itime}")

            # take first one in the batch, and then each
            # sample in the "time-stack"
            img_input = img_input_batch[0, :, :, itime]

            # take always the first single-channel sample in the batch
            img_label = img_label_batch[0, :, :, 0]

            logger.info(f"img_input.shape {img_input.shape}")
            logger.info(f"img_label.shape {img_label.shape}")

            # print summary stats
            summary_dict = {
                "sample id": isample,
                "input shape": img_input.shape,
                "input avg(img_in)": np.mean(img_input),
                "input std(img_in)": np.std(img_input),
                "input max(img_in)": np.max(img_input),
                "input min(img_in)": np.min(img_input),
                "label shape": img_label.shape,
                "label avg(img_in)": np.mean(img_label),
                "label std(img_in)": np.std(img_label),
                "label max(img_in)": np.max(img_label),
                "label min(img_in)": np.min(img_label),
            }

            # save log to file
            json_dict = collections.OrderedDict(summary_dict)

            with open(
                os.path.join(save_path, f"sample_{isample}_time{itime}_log.json"), "w"
            ) as f:
                jstr = json.dumps({k: str(v) for k, v in json_dict.items()}, indent=4)
                f.write(jstr)

            # normalized arrays
            normalized_img_input = np.squeeze(
                img_input - np.min(img_input) / (np.max(img_input) - np.min(img_input))
            )
            normalized_img_label = np.squeeze(
                img_label - np.min(img_label) / (np.max(img_label) - np.min(img_label))
            )

            # save sample picture
            plt.figure(isample)
            plt.subplot(2, 1, 1)
            plt.imshow(normalized_img_input, cmap="gray")
            plt.subplot(2, 1, 2)
            plt.imshow(normalized_img_label)
            plt.savefig(os.path.join(save_path, f"sample_{isample}_time{itime}.png"))
            plt.close()
