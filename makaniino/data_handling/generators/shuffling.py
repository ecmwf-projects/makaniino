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

import numpy as np


class Shuffling:
    """
    Shuffle indexes according to different strategies
    """

    def __init__(self, idxs, batch_size):
        self.idxs = idxs
        self.batch_size = batch_size

    def apply(self, seed=0):
        """
        Shuffle
        """
        np.random.seed(seed)

        return self._do_apply()

    def _do_apply(self):
        """
        Do the shuffling
        """
        raise NotImplementedError


class DoNotShuffle(Shuffling):
    """
    Do not shuffle
    """

    def _do_apply(self):
        return self.idxs


class SampleShuffle(Shuffling):
    """
    Per-sample shuffling
    """

    def _do_apply(self):
        sample_permutation = np.random.permutation(self.idxs)
        return sample_permutation


class BatchShuffle(Shuffling):
    """
    Per-batch shuffling
    """

    def _do_apply(self):

        n_idx_rows = len(self.idxs) // self.batch_size
        n_idx_cols = self.batch_size
        full_batch_len = n_idx_rows * n_idx_cols

        _idxs = np.asarray(self.idxs[:full_batch_len]).reshape((n_idx_rows, n_idx_cols))
        batch_permutation = _idxs[np.random.permutation(range(n_idx_rows)), :].flatten()
        return batch_permutation


class N2BatchShuffle(Shuffling):
    """
    2-batch shuffling
    """

    n_combining_rows = 2

    def _do_apply(self):

        n_idx_rows = len(self.idxs) // self.batch_size
        n_idx_cols = self.batch_size
        full_batch_len = n_idx_rows * n_idx_cols

        _idxs = np.asarray(self.idxs[:full_batch_len]).reshape((n_idx_rows, n_idx_cols))

        # each row will be made up of elements from N separate rows
        shuffled_idxs = np.zeros_like(_idxs)
        for rr, row in enumerate(_idxs):
            combining_rows = np.random.choice(
                n_idx_rows, size=self.n_combining_rows, replace=False
            )
            elems = _idxs[combining_rows, :]
            sampled_elems = np.random.choice(
                elems.flatten(), size=self.batch_size, replace=False
            )
            shuffled_idxs[rr] = sampled_elems.reshape((1, self.batch_size))

        # set the "order" property
        return shuffled_idxs.flatten()


class N4BatchShuffle(N2BatchShuffle):
    """
    4-batch shuffling
    """

    n_combining_rows = 4


class N8BatchShuffle(N2BatchShuffle):
    """
    4-batch shuffling
    """

    n_combining_rows = 8


shuffling_factory = {
    "none": DoNotShuffle,  # do not shuffle
    "sample": SampleShuffle,  # shuffle on a per-sample basis
    "batch": BatchShuffle,  # shuffle on a per-batch basis
    "batch-2": N2BatchShuffle,  # shuffle on a per-batch basis (mixing elements from 2 batches)
    "batch-4": N4BatchShuffle,  # shuffle on a per-batch basis (mixing elements from 4 batches)
    "batch-8": N8BatchShuffle,  # shuffle on a per-batch basis (mixing elements from 8 batches)
}


class DistributedShuffler:
    """
    Shuffle indexes distributed on multiple nodes
    here the indexes are calculated taking into account local rank
    """

    def __init__(self, shuffler, seed):

        self.shuffler = shuffler

        self.global_idxs_shuffled = self._calculate_global_idxs_shuffled(seed)
        self.local_idxs_shuffled = self._calculate_local_idxs_shuffled()
        self.local_idxs_sorted = sorted(self.local_idxs_shuffled)

    def _calculate_global_idxs_shuffled(self, seed):
        """
        Global order of shuffled indexes
        """
        return self.shuffler.apply(seed=seed)

    def _calculate_local_idxs_shuffled(self):
        """
        Local (per node) order of shuffled indexes
        """
        import horovod.keras as hvd

        _rank = hvd.rank()
        _size = hvd.size()

        n_local_samples = len(self.global_idxs_shuffled) // _size

        start_idx = _rank * n_local_samples
        end_idx = (_rank + 1) * n_local_samples

        return self.global_idxs_shuffled[start_idx:end_idx]
