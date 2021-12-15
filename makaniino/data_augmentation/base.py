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
import multiprocessing

import numpy as np

from makaniino.data_handling.datasets.base import DatasetTypes, EntrySchema, RecordSchema
from makaniino.data_handling.datasets.zarr import CycloneDatasetZARR
from makaniino.utils.generic_utils import npstring_to_numpy

logger = logging.getLogger(__name__)

# multiprocessing.util.log_to_stderr(10)


class DataAugmenter:
    """
    Handles a data augmentation to a
    specific dataset
    """

    out_ds_schema = None
    cyclone_img_size = (128, 128)

    rotate_samples = False
    rand_rot_max_deg = 20

    chunk_size_ds_default = 1

    def __init__(self, ds, out_path, tracks_source=None, seed=1, chunk_size_ds=None):

        # input ds
        self.ds = ds

        # output ds setup
        self.out_path = out_path
        self.ds_out = None

        # dataset internal chunk size
        self.chunk_size_ds = (
            chunk_size_ds if chunk_size_ds else self.chunk_size_ds_default
        )

        # makaniino tracks (optional),
        # some classes might need
        # makaniino tracks information..
        self.tracks_source = tracks_source

        # random numbers seed
        self.seed = np.random.seed(seed)

    def _setup_schema(self, length, nch_input, nch_groundtruth):
        """
        Setup the schema of the output dataset
        """

        # size of input tensor
        input_size = [self.cyclone_img_size[0], self.cyclone_img_size[1], nch_input]

        # ground truth size
        ground_truth_size = [
            self.cyclone_img_size[0],
            self.cyclone_img_size[1],
            nch_groundtruth,
        ]

        # Schema of the database to be created
        # b this pre-processor
        record_schema = RecordSchema(
            [
                EntrySchema(
                    {
                        "name": "train",
                        "shape": (length, input_size[0], input_size[1], input_size[2]),
                        "chunks": (
                            self.chunk_size_ds,
                            input_size[0],
                            input_size[1],
                            input_size[2],
                        ),
                        "dtype": DatasetTypes.NP_FLOAT(),
                    }
                ),
                EntrySchema(
                    {
                        "name": "test",
                        "shape": (
                            length,
                            ground_truth_size[0],
                            ground_truth_size[1],
                            ground_truth_size[2],
                        ),
                        "chunks": (
                            self.chunk_size_ds,
                            ground_truth_size[0],
                            ground_truth_size[1],
                            ground_truth_size[2],
                        ),
                        "dtype": DatasetTypes.NP_FLOAT(),
                    }
                ),
                EntrySchema(
                    {
                        "name": "time",
                        "shape": (length, 1),
                        "chunks": (self.chunk_size_ds, 1),
                        "dtype": DatasetTypes.NP_STR(),
                    }
                ),
                EntrySchema(
                    {
                        "name": "points",
                        "shape": (length, 1),
                        "chunks": (self.chunk_size_ds, 1),
                        "dtype": DatasetTypes.NP_STR(),
                    }
                ),
            ]
        )

        self.var_names = [v["name"] for v in record_schema.to_list()]

        return record_schema

    def run(self, workers=1, flush_every=None):
        """
        Run the data augmentation process
        """

        # Calc all the new samples and then write to
        # the database in one go (legacy code)
        if flush_every is None:
            self._run_oneshot(workers=workers)

        # Cals and flush at pre-set interval
        else:
            self._run_flush_at_interval(workers=workers, flush_every=flush_every)

    def _run_flush_at_interval(self, workers=1, flush_every=1):

        assert workers >= 1

        logger.info(f"Processing started with {workers} workers")

        # open the reading ds
        self.ds.open()

        input_ds_length = self.ds.length

        # total # of new samples calculated
        n_new_samples = 0

        # main loop over all the ds indexes
        read_idx_start = 0

        workers_pool = multiprocessing.Pool(processes=workers)

        while read_idx_start < input_ds_length:

            batch_samples = []

            read_idx_end = min(read_idx_start + flush_every, input_ds_length)
            sample_indexes = range(read_idx_start, read_idx_end)

            logger.info(
                f"====> Reading input DS samples from {read_idx_start}, to {read_idx_end}"
            )

            # do the calculations for all the samples in range(idx_start, idx_end)
            for new_samples in workers_pool.map(self.extract_samples, sample_indexes):
                batch_samples.extend(new_samples)
                n_new_samples += len(new_samples)

            logger.debug(f"Generated/extracted {len(batch_samples)} new samples!")

            # setup the out_ds schema
            if read_idx_start == 0:
                logger.info(f"Init output DS with size {len(batch_samples)}")
                nch_input = batch_samples[0]["train"].shape[-1]
                nch_groundtruth = batch_samples[0]["test"].shape[-1]
                out_schema = self._setup_schema(
                    len(batch_samples), nch_input, nch_groundtruth
                )

                # prepare the output dataset
                self.ds_out = CycloneDatasetZARR(self.out_path, out_schema)

                # now write all the samples in the output dataset
                self.ds_out.open()

                # indexes where to write the newly computed samples
                batch_samples_out_idxs = range(len(batch_samples))

            else:

                logger.info(f"Resizing output DS by {len(batch_samples)}")

                # indexes where to write the newly computed samples
                batch_samples_out_idxs = [
                    _idx + self.ds_out.length for _idx in range(len(batch_samples))
                ]

                # resize the output ds to fit the new samples
                self.ds_out.resize(self.ds_out.length + len(batch_samples))

            # parallel write of the new samples
            logger.info(f"Writing {len(batch_samples)} samples..")
            workers_pool.map(
                self.write_sample, zip(batch_samples_out_idxs, batch_samples)
            )
            # for s_ in zip(batch_samples_out_idxs, batch_samples):
            #     self.write_sample(s_)

            read_idx_start = read_idx_end

        logger.info(
            f"The output dataset in {self.ds_out.ds_path} has been written."
            f" It contains {self.ds_out.length} samples."
        )

        self.ds_out.close()

        workers_pool.close()
        workers_pool.join()

    def _run_oneshot(self, workers=1):

        assert workers >= 1

        if workers > 1:

            logger.info(f"Processing started with {workers} workers")

            pool = multiprocessing.Pool(processes=workers)

            self.ds.open()

            # prepares all the cuts
            isample = 0
            all_samples = []
            for new_samples in pool.imap(
                self.extract_samples, range(self.ds.length), 1
            ):
                all_samples.extend(new_samples)
                isample += len(new_samples)
            pool.close()
            pool.join()

            # setup the out_ds schema (for all the samples)
            nch_input = all_samples[0]["train"].shape[-1]
            nch_groundtruth = all_samples[0]["test"].shape[-1]
            out_schema = self._setup_schema(
                len(all_samples), nch_input, nch_groundtruth
            )

            # prepare the output dataset
            self.ds_out = CycloneDatasetZARR(self.out_path, out_schema)

            # now write all the samples in the output dataset
            self.ds_out.open()
            pool = multiprocessing.Pool(processes=workers)
            pool.imap(self.write_sample, enumerate(all_samples), 1)
            pool.close()
            pool.join()

        else:

            logger.info("Processing started in SERIAL mode")

            # start the reading/writing
            self.ds.open()

            # loop over the ds and cut out only the
            # regions around the cyclones
            isample = 0
            all_samples = []
            for idx in range(self.ds.length):
                new_samples = self.extract_samples(idx)
                all_samples.extend(new_samples)
                isample += len(new_samples)

            # setup the out_ds schema (for all the samples)
            nch_input = all_samples[0]["train"].shape[-1]
            nch_groundtruth = all_samples[0]["test"].shape[-1]
            out_schema = self._setup_schema(
                len(all_samples), nch_input, nch_groundtruth
            )

            # prepare the output dataset
            self.ds_out = CycloneDatasetZARR(self.out_path, out_schema)
            self.ds_out.open()

            # add the new samples in the dataset one by one
            for idx, sample in enumerate(all_samples):
                self.write_sample((idx, sample))

        logger.info(
            f"The output dataset in {self.ds_out.ds_path} has been written."
            f" It contains {self.ds_out.length} samples."
        )

    def extract_samples(self, sample_idx):
        """self.extract_samples
        Extract single or multiple samples from each
        input sample
        """
        input_sample = self.ds[sample_idx]
        tr, test, time, pts = input_sample
        new_samples_pos, new_samples_neg = self._do_extract_samples(input_sample)

        logger.info(
            f"Worker {multiprocessing.current_process().name} read sample {sample_idx}, "
            f"containing {npstring_to_numpy(pts).shape[0]} cyclones "
            f"=> +ve crops: {len(new_samples_pos)}, -ve crops {len(new_samples_neg)}"
        )

        return new_samples_pos + new_samples_neg

    def write_sample(self, idx_and_sample):
        """
        Write the extracted samples into
        the output dataset
        """
        idx, sample = idx_and_sample
        self.ds_out.write_at(sample, idx)
        logger.debug(
            f"Worker {multiprocessing.current_process()} has written sample {idx}!"
        )

    def _do_extract_samples(self, input_sample):
        """
        The routine that extract one or more samples from
        each input sample, by applying a data extraction or
        augmentation strategy
        """
        raise NotImplementedError

    def _do_crop(self, data, cyc_cord, crop_size, rot_rad=None):
        """
        Utility function to apply a random rotation to a
        patch of the input sample
        """

        # grid
        win_sx, win_sy = crop_size
        win_ctr = np.asarray(cyc_cord)

        # find the cut area around the single makaniino
        x_min = cyc_cord[0] - int(win_sx / 2)
        x_max = cyc_cord[0] + int(win_sx / 2)
        y_min = cyc_cord[1] - int(win_sy / 2)
        y_max = cyc_cord[1] + int(win_sy / 2)

        # not really needed for now..
        assert (
            x_max - x_min == win_sx
        ), f"x_max - x_min {x_max - x_min} but expected {win_sx}"
        assert (
            y_max - y_min == win_sy
        ), f"y_max - y_min {y_max - y_min} but expected {win_sy}"

        # check consistency of x_min, x_max, y_min, y_max
        if x_min < 0 or x_max >= data.shape[1] or y_min < 0 or y_max >= data.shape[2]:

            logger.info("invalid cut!")
            return None

        # no rotation applied, just crop and return
        if rot_rad is None:
            return data[:, x_min:x_max, y_min:y_max, :]

        # idx grid
        grid_x, grid_y = np.meshgrid(range(win_sx), range(win_sy), indexing="ij")

        grid_x = grid_x.astype(np.float64)
        grid_y = grid_y.astype(np.float64)

        # re-center the grid about 0
        grid_x -= win_sx / 2
        grid_y -= win_sy / 2

        grid_unit = np.stack((grid_x, grid_y), axis=-1)
        # grid_unit_ctr = grid_unit + win_ctr

        # random rotation
        Mrot = [[np.cos(rot_rad), np.sin(rot_rad)], [-np.sin(rot_rad), np.cos(rot_rad)]]

        grid_unit_rot = np.matmul(grid_unit, Mrot)
        # grid_unit_rot = grid_unit

        grid_unit_rot_ctr = grid_unit_rot + win_ctr

        nch = data.shape[-1]
        x_interps = grid_unit_rot_ctr[:, :, 0].reshape(-1, 1).astype(np.int)
        y_interps = grid_unit_rot_ctr[:, :, 1].reshape(-1, 1).astype(np.int)

        # check that the rotation did not end up outside the image
        if (
            np.min(x_interps) < 0
            or np.max(x_interps) >= data.shape[1]
            or np.min(y_interps) < 0
            or np.max(y_interps) >= data.shape[2]
        ):
            return None

        grid_unit_rot_ctr_vals = data[0, x_interps, y_interps, :]
        grid_unit_rot_ctr_vals = grid_unit_rot_ctr_vals.reshape(
            (1, win_sx, win_sy, nch)
        )

        # # ************** PLOTS **************
        # fig_sc = 4
        # figsize = (int(win_sy / win_sx * fig_sc), fig_sc)
        # plt.figure(22, figsize=figsize)
        # plt.plot(grid_unit_ctr[:, :, 1], grid_unit_ctr[:, :, 0], ".r")
        # plt.plot(grid_unit_rot_ctr[:, :, 1], grid_unit_rot_ctr[:, :, 0], ".b")
        #
        # figsize = (int(win_sy / win_sx * fig_sc), fig_sc)
        # plt.figure(23, figsize=figsize)
        #
        # print(f"grid_unit_ctr.shape {grid_unit_ctr.shape}")
        # plt.pcolor(data[0, x_min:x_max, y_min:y_max, 0])
        #
        # figsize = (int(win_sy / win_sx * fig_sc), fig_sc)
        # plt.figure(24, figsize=figsize)
        # plt.pcolor(grid_unit_rot_ctr_vals[0, :, :, 0])
        #
        # plt.show()
        # # ***********************************

        return grid_unit_rot_ctr_vals
