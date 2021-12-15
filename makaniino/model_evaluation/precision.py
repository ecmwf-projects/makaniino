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

from makaniino.model_evaluation.scorer import Scorer

logger = logging.getLogger(__name__)


class Precision(Scorer):
    """
    Takes a ground truth and a prediction field
    and return a score
    """

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        # evaluation index incremented each
        # time the eval method is called
        self.eval_idx = 0

    def __str__(self):
        return "Precision"

    def __call__(self, hit_miss_dict):

        # extract tp, fp, fn
        tp = hit_miss_dict["true_positives"]
        fp = hit_miss_dict["false_positives"]

        if len(tp) + len(fp) == 0:
            precision = 0
        else:
            precision = len(tp) / (len(tp) + len(fp))

        return precision
