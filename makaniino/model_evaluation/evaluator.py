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

import numpy as np

from makaniino.cyclone_localizers.factory import cyclone_localizer_factory
from makaniino.data_handling.datasets.zarr import CycloneDatasetZARR
from makaniino.data_handling.online_processing import (
    DataNormalizer,
    DataRecaster,
    EdgeTrimmer,
    NANSwapper,
)
from makaniino.utils.generic_utils import (
    haversine_dist,
    latlon_2_pixel,
    npstring_to_numpy,
    pixel_2_latlon,
)
from makaniino.utils.plot_utils import plot_prediction

logger = logging.getLogger(__name__)


class Evaluator:
    """
    Evaluate a model on a dataset
    according to a defined metric function
    """

    # list of fields required to be
    # taken from the dataset
    required_fields = ["train", "test", "time", "points"]

    def __init__(self, model_config, model_weights_path, debug=None):

        # self.model = model
        self.model_config = model_config

        # model weights
        self.model_weights_path = model_weights_path

        # debug flag
        self.debug_flag = debug

        # list of scores of the model on the dataset
        self.tp_fp_fn = []

        # get the model from configs
        self.model = self._resurrect_model()

    def evaluate(
        self, scorers, ds_path, n_evals=None, shuffle=False, cyclone_localizer="dbscan"
    ):
        """
        Evaluate the model over the dataset
        Returns:
        """

        data_handler = self.model_config.get_configurable("data_handler")

        # set test data path
        data_handler.set_param("test_data_path", ds_path)

        # this is a quick and dirty workaround but there should be
        # a more thorough thinking about lazy building of configurables
        data_handler.reset_data_path(os.path.join(ds_path, "_data"), "testing")

        # open and configure the dataset
        ds = CycloneDatasetZARR(ds_path)
        ds = ds.batch(1).serve(self.required_fields)
        ds = ds.shuffle(shuffle)

        x_cut_pxl = data_handler.get_param("x_cut_pxl")
        y_cut_pxl = data_handler.get_param("y_cut_pxl")
        norm_factor = data_handler.get_param("norm_factor")
        shift_factor = data_handler.get_param("shift_factor")
        dataset_output_type = data_handler.get_param("dataset_output_type")

        ds = ds.set_online_processors(
            [
                EdgeTrimmer(x_cut_pxl=x_cut_pxl, y_cut_pxl=y_cut_pxl),
                DataNormalizer(norm_factor=norm_factor, shift_factor=shift_factor),
                DataRecaster(y_type=dataset_output_type),
                NANSwapper(),
            ]
        )

        ds.open()

        # build makaniino localizer
        localizer = cyclone_localizer_factory[cyclone_localizer]

        # empty the evaluation scores
        hit_miss_dict = {
            "true_positives": [],
            "false_positives": [],
            "false_negatives": [],
        }

        # evaluate the model over the ds
        # on a defined number of samples
        if n_evals is not None:

            # make sure that we requested
            # n_eval < ds.length
            assert n_evals <= ds.length

            # loop for n_evals times
            ds_iter = iter(ds)
            for sidx in range(n_evals):

                sample = next(ds_iter)
                sample_dict = ds.sample_as_dict(sample)

                tp, fp, fn = self.process_sample(
                    sidx, sample_dict, x_cut_pxl, y_cut_pxl, localizer
                )

                hit_miss_dict["true_positives"].extend(tp)
                hit_miss_dict["false_positives"].extend(fp)
                hit_miss_dict["false_negatives"].extend(fn)

                logger.info(
                    f"Sample {sidx:6} of {n_evals} => "
                    f"TP: {len(tp)}, FP: {len(fp)}, FN: {len(fn)}"
                )

        else:

            # loop over the whole dataset
            sidx = 0
            for sample in ds:

                sample_dict = ds.sample_as_dict(sample)

                tp, fp, fn = self.process_sample(
                    sidx, sample_dict, x_cut_pxl, y_cut_pxl
                )

                hit_miss_dict["true_positives"].extend(tp)
                hit_miss_dict["false_positives"].extend(fp)
                hit_miss_dict["false_negatives"].extend(fn)

                logger.info(
                    f"Sample {sidx:6} of {n_evals} => "
                    f"TP: {len(tp)}, FP: {len(fp)}, FN: {len(fn)}"
                )

                sidx += 1

        # do the evaluation
        global_scores = [scorer(hit_miss_dict) for scorer in scorers]

        # close the dataset
        ds.close()

        # return the score over the dataset
        return global_scores

    def process_sample(self, sidx, sample_dict, x_cut_pxl, y_cut_pxl, localizer):

        prediction = self.model.predict(sample_dict["train"])
        cyc_true = npstring_to_numpy(sample_dict["points"])
        date_time = sample_dict["time"][0, 0].replace(" ", "_").replace(":", "_")

        # Predict makaniino centers
        cyc_pred = self.predict_cyclone_centers(
            prediction, x_cut_pxl, y_cut_pxl, localizer
        )

        # plot the sample if debugging
        if self.debug_flag:
            self._plot_sample(
                sample_dict,
                x_cut_pxl,
                y_cut_pxl,
                cyc_true,
                date_time,
                prediction,
                cyc_pred,
                sidx,
            )

        # tp, fp, fn
        tp, fp, fn = self._calc_tp_fp_fn(cyc_true, cyc_pred)

        return tp, fp, fn

    def _resurrect_model(self):

        # Resurrect the configuration from config JSON file
        print(self.model_config.to_string())

        # Resurrect model and data-handler
        model = self.model_config.get_configurable("model")

        # re-build the model
        model.build()

        # load weights from file
        model.load_weights_from_file(self.model_weights_path)
        model.summary()

        return model

    def predict_cyclone_centers(self, prediction, x_cut_pxl, y_cut_pxl, localizer):
        """
        Seek makaniino centers through KMeans
        Args:
            prediction:
            x_cut_pxl:
            y_cut_pxl:
            localizer:

        Returns:

        """

        # find makaniino centers
        finder = localizer(prediction[0, :, :, 0])
        cyc_coords_pxl_pred = finder.find()

        if not cyc_coords_pxl_pred:
            return []

        # if predicted makaniino centers are found, use them
        cyc_coords_latlon_pred = pixel_2_latlon(
            cyc_coords_pxl_pred, prediction.shape, x_cut_pxl, y_cut_pxl
        )

        return cyc_coords_latlon_pred

    @staticmethod
    def _calc_tp_fp_fn(cyc_true, cyc_pred, min_dist_km=300):
        """
        Calculate true-positives, false-positives, false-negatives
        in a single sample
        """

        cyc_true = np.asarray(cyc_true)
        cyc_pred = np.asarray(cyc_pred)

        # init tp, fn, fp
        tp = []
        fp = []
        # tp_idxs = []

        # find matrix of distances pred true
        len_pred = cyc_pred.shape[0]
        len_true = cyc_true.shape[0]
        d_mat = np.zeros((len_pred, len_true))
        for ii, ip in enumerate(cyc_pred):
            for jj, it in enumerate(cyc_true):
                d_mat[ii, jj] = haversine_dist(*ip, *it)

        # assign TP and FP
        matched_true_idxs = []
        for pred_idx, pred in enumerate(cyc_pred):

            closest_true_idx = np.argmin(d_mat[pred_idx, :])
            min_dist = d_mat[pred_idx, closest_true_idx]

            # it's a true positive
            if min_dist < min_dist_km and closest_true_idx not in matched_true_idxs:

                tp.append(pred)
                matched_true_idxs.append(closest_true_idx)

            else:  # it's a false positive
                fp.append(pred)

        # populate false negatives
        fn = [
            cyc_true[it] for it in range(len(cyc_true)) if it not in matched_true_idxs
        ]

        logger.debug(f"TP {tp}, FP {fp}, FN {fn}")

        return tp, fp, fn

    def _check_sample(self, sample):
        """
        Check that the sample contains
        all the required fields
        Args:
            sample:
        Returns:
        """

        for k in self.required_fields:
            try:
                assert k in sample
            except AssertionError:
                raise AssertionError(f"tensor {k} required but not provided in sample")

    @staticmethod
    def _plot_sample(
        sample_dict,
        x_cut_pxl,
        y_cut_pxl,
        cyc_true,
        date_time,
        prediction,
        cyc_pred,
        eval_idx,
    ):
        """
        Plot the prediction for debugging
        """

        cyc_coords_pxl_real = latlon_2_pixel(
            cyc_true, prediction.shape, x_cut_pxl=x_cut_pxl, y_cut_pxl=y_cut_pxl
        )

        if len(cyc_pred):
            cyc_coords_pxl_pred = latlon_2_pixel(
                cyc_pred, prediction.shape, x_cut_pxl=x_cut_pxl, y_cut_pxl=y_cut_pxl
            )
        else:
            cyc_coords_pxl_pred = None

        ich = 0
        x_data = sample_dict["train"]
        y_data = sample_dict["test"]
        plot_prediction(
            x_data[0, :, :, ich],
            prediction[0, :, :, 0],
            y_data[0, :, :, 0],
            output_dir=os.getcwd(),
            title=f"eval_{eval_idx}_{date_time}",
            cyc_coords_pxl_real=cyc_coords_pxl_real,
            cyc_coords_latlon_real=cyc_true,
            cyc_coords_pxl_pred=cyc_coords_pxl_pred,
            cyc_coords_latlon_pred=cyc_pred,
            twod_only=True,
        )
