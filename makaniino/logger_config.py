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

logging_format = "%(asctime)s; " "%(name)s; " "%(levelname)s; " "%(message)s"

root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

logging.basicConfig(
    level=logging.INFO, format=logging_format, datefmt="%Y-%m-%d %H:%M:%S"
)

# add stdout handler if not there already
sh = logging.StreamHandler()
sh.setLevel(logging.INFO)
sh.setFormatter(logging.Formatter(logging_format))
if all(type(s) != logging.StreamHandler for s in root_logger.handlers):
    root_logger.addHandler(sh)


def getLogger(name):
    """
    Get the logger
    """
    return logging.getLogger(name)
