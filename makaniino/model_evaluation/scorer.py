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


class Scorer:
    """
    Return a score
    """

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, scores):
        raise NotImplementedError

    def __str__(self):
        return "Scorer"
