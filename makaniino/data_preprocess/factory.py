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


from makaniino.data_preprocess.default import PreProcessorDefault
from makaniino.data_preprocess.tc_classes import PreProcessorTCClasses

factory = {
    "default": PreProcessorDefault,
    "tc_classes": PreProcessorTCClasses,
}
