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

from makaniino.model_evaluation.precision import Precision
from makaniino.model_evaluation.recall import Recall
from makaniino.model_evaluation.scorer import Scorer

logger = logging.getLogger(__name__)


class F1Scorer(Scorer):
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
        return "F1-Score"

    def __call__(self, hit_miss_dict):

        # calc precision
        precision_scorer = Precision()
        precision = precision_scorer(hit_miss_dict)

        # calc recall
        recall_scorer = Recall()
        recall = recall_scorer(hit_miss_dict)

        # calc f1-score
        if precision + recall != 0:
            f1_score = 2 * precision * recall / (precision + recall)
        else:
            f1_score = 0

        return f1_score
