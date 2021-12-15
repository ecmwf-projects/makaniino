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


class DataCache:
    """
    Data that is cached in memory and can be
    passed to a generator that can use it at will
    """

    def __init__(self, max_len=-1):

        self.max_len = max_len  # -1 means no max length

        self.cached_data = {}
        self.gidx = 0

    def get_data(self, idx):
        return self.cached_data[idx]

    def add_data(self, data):
        if self.gidx < self.max_len or self.max_len == -1:
            self.cached_data[self.gidx] = data
            self.gidx += 1

    def has_idx(self, idx):
        return idx in self.cached_data
