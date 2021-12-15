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


from makaniino.model_evaluation.f1_scorer import F1Scorer
from makaniino.model_evaluation.precision import Precision
from makaniino.model_evaluation.recall import Recall

factory = {
    "f1_score": F1Scorer,
    "precision": Precision,
    "recall": Recall,
}
