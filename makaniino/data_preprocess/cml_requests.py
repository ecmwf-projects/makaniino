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

import copy
import logging
import os

import climetlab as cml
import dateutil.parser
from climetlab.core.settings import SETTINGS

logger = logging.getLogger("cml_requests")


class CMLRequest:
    """
    A generic configurable climetlab request
    """

    required = [
        "var_name",
        "start_date",
        "end_date",
    ]

    defaults = {"grid": [0.5, 0.5], "field_size": [361, 720]}

    def __init__(self, user_args):

        self.args = copy.deepcopy(user_args)

        # check for required args
        for arg in self.required:
            if not getattr(user_args, arg):
                raise RuntimeError(
                    f"Argument {arg} is required "
                    f"for request type {self.__class__.__name__}, but is NOT FOUND!"
                )

        # compose date string
        self.dates_str = self.get_date_string(user_args.start_date, user_args.end_date)

        # split comma separated var string
        vars = self.args.var_name.split(",")

        if len(vars) == 1:
            self.args.var_name = vars[0]
        else:
            self.args.var_name = vars

    @classmethod
    def from_dict(cls, conf_dict):

        # dummy configuration obj
        class ConfStruct:
            pass

        # dict config params
        conf_args = ConfStruct()
        for k, v in conf_dict.items():
            setattr(conf_args, k, v)

        return cls(conf_args)

    def __str__(self):
        _str = "\n{} Request Configuration:".format(self.__class__.__name__)
        _str += "\n"
        _str += "\n".join(
            ["  {}: {}".format(arg, getattr(self.args, arg)) for arg in self.required]
        )
        _str += "\n"
        _str += "\n".join(["  {}: {}".format(k, v) for k, v in self.defaults.items()])
        return _str

    @staticmethod
    def get_date_string(start_date, end_date):
        """
        Compose date retrieval string
        """

        date_start = dateutil.parser.parse(start_date)
        date_end = dateutil.parser.parse(end_date)

        start_date_string = date_start.strftime("%Y-%m-%d")
        end_date_string = date_end.strftime("%Y-%m-%d")

        date_string_all = start_date_string + "/to/" + end_date_string
        logger.info(f"date retrival string = {date_string_all}")

        return date_string_all

    def set_cache_path(self, cache_path):

        if not os.path.exists(cache_path):
            os.mkdir(cache_path)

        # SETTINGS["cache_directory"] = cache_path
        SETTINGS.set("cache-directory", cache_path)

    def load_source(self):
        """
        Call climetlab internally
        """
        logger.error(f"Base class {__class__.__name__} method called! Returning None..")
        return None


class CMLRequest_MARS(CMLRequest):
    """
    climetlab MARS request
    """

    required = CMLRequest.required + [
        "levtype",
        "type",
        "stream",
        "expver",
    ]

    def __init__(self, user_args):
        super().__init__(user_args)

    def load_source(self):
        data = cml.load_source(
            "mars",
            param=self.args.var_name,
            date=self.dates_str,
            levtype=self.args.levtype,
            type=self.args.type,
            stream=self.args.stream,
            expver=self.args.expver,
            grid=self.defaults["grid"],
            time="00:00/06:00/12:00/18:00",
        )

        return data


class CMLRequest_CDS(CMLRequest):
    """
    climetlab CDS request
    """

    required = CMLRequest.required + ["cds_database", "product_type"]

    def __init__(self, user_args):
        super().__init__(user_args)

    def load_source(self):

        data = cml.load_source(
            "cds",
            self.args.cds_database,
            variable=self.args.var_name,
            product_type=self.args.product_type,
            date=self.dates_str,
            grid=self.defaults["grid"],
            time="00:00/03:00/06:00/09:00/12:00/15:00/18:00/21:00",
        )

        return data


class CMLRequest_CDS_GRID025(CMLRequest_CDS):
    """
    climetlab CDS request with grid res [0.25, 0.25]
    """

    defaults = {"grid": [0.25, 0.25], "field_size": [721, 1440]}


cml_request_types = {
    "mars": CMLRequest_MARS,
    "cds": CMLRequest_CDS,
    "cds_grid025": CMLRequest_CDS_GRID025,
}
