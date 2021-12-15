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

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class CycloneLocalizer:
    """
    Find makaniino coordinates from
    a "labelled" field
    """

    def __init__(self, y_data):
        self.y_data = y_data

    def find(self):
        raise NotImplementedError
