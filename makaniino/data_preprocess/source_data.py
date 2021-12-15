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
import os

import jsonschema
import xarray as xr

from makaniino.data_preprocess.cml_requests import cml_request_types
from makaniino.utils.generic_utils import read_json

logger = logging.getLogger(__name__)


class SourceData:
    """
    Wrapper class that handles data to be pre-processed
    """

    schema_json = os.path.join(os.path.dirname(__file__), "schema_download.json")

    var_to_xr_map = {
        "100u": "u100",
        "100m_u_component_of_wind": "u100",
        "100v": "v100",
        "100m_v_component_of_wind": "v100",
        "surface_pressure": "sp",
        "sea_surface_temperature": "sst",
    }

    def __init__(self, config_json):

        # source data config
        self.config_json = config_json

        # all variables available for request
        self.all_vars_ = self.list_all_vars()

        # convenience structure var_name to request
        # *** {
        # ***   "var1": {"req_name": "req1", "req_idx": 0},
        # ***   "var2": {"req_name": "req1", "req_idx": 5},
        # ***   ...
        # *** }
        self.vars_to_req_map_ = self.build_vars_to_req_map()

        # Reference to open handles
        self.var_databases_ = None

        # the "opened" variables
        self.open_vars_ = None

        self.latitudes = None
        self.longitude = None

    @classmethod
    def from_json_path(cls, json_path):
        """
        From config path
        """

        # read config json
        config_json = read_json(json_path)

        # validate the json against the schema
        cls.validate_json(config_json)

        return cls(config_json)

    @classmethod
    def check_source_data(cls, database_paths):
        """
        Some checks on the source dta handled
        """

        # 1) All the data must have the same resolution!
        # NOTE: strong assumption!
        path_0 = list(database_paths.keys())[0]
        lat_size_0 = xr.open_mfdataset(
            database_paths[path_0], engine="cfgrib"
        ).latitude.values.size
        lon_size_0 = xr.open_mfdataset(
            database_paths[path_0], engine="cfgrib"
        ).longitude.values.size

        for var, path in list(database_paths.items())[1:]:

            ds = xr.open_mfdataset(path, engine="cfgrib")

            assert (
                ds.latitude.values.size == lat_size_0
            ), f"latitude of {var} has wrong shape {ds.latitude.values.size}"

            assert (
                ds.longitude.values.size == lon_size_0
            ), f"longitude of {var} has wrong shape {ds.longitude.values.size}"

    def list_all_vars(self):
        """
        All variables handled (potentially downloadable)
        according to the configuration
        """

        vars_ = []

        for req in self.config_json["requests"]:
            vars_.extend(req["var_name"].split(","))

        return vars_

    def build_vars_to_req_map(self):
        """
        Build a convenience map between
        variable name and corresponding request
        """

        vars_to_req_map_ = {}
        for rid, req in enumerate(self.config_json["requests"]):

            # variable in request rr
            vs_ = req["var_name"].split(",")

            for v_ in vs_:
                if v_ in vars_to_req_map_:
                    logger.error(
                        f"Variable {v_} is already in the list!, "
                        f"the same variable should not be present "
                        f"in more than 1 request!"
                    )
                    raise ValueError
                else:
                    vars_to_req_map_.update(
                        {v_: {"req_name": req["name"], "req_idx": rid}}
                    )
        return vars_to_req_map_

    def run_requests(self):
        """
        Run all the requests
        """

        requests = self.config_json["requests"]
        metadata = self.config_json["metadata"]

        # simply run all the requests in the JSON
        for request in requests:
            req = cml_request_types[request["source"]].from_dict(request)
            req.set_cache_path(metadata["cache_path"])
            req.load_source()

    def get_requests(self):
        """
        Get requests
        """
        return self.config_json["requests"]

    def open_handle(self, open_vars=None):
        """
        Open the handle of the source data (Xarray files)
        """

        self.open_vars_ = open_vars if open_vars else self.all_vars_

        # read source data metadata
        metadata = self.config_json["metadata"]

        self.var_databases_ = {}
        for var in self.open_vars_:

            # request corresponding to variable var
            req = self.get_request(self.vars_to_req_map_[var]["req_idx"])

            req_data = cml_request_types[req["source"]].from_dict(req)
            req_data.set_cache_path(metadata["cache_path"])
            # xarray_hdl = req_data.load_source().to_xarray()

            xarray_path = req_data.load_source().path
            self.var_databases_.update({var: xarray_path})

        # run some consistency checks on the data handled
        self.check_source_data(self.var_databases_)

    def __str__(self):
        return f"DataSource:\n{self.config_json}"

    @property
    def all_variables(self):
        """
        All the variables from all requests in the config
        """
        return self.all_vars_

    @property
    def open_variables(self):
        """
        Requested var at the time the data handle was open
        """
        return self.open_vars_

    def get_latitudes(self):
        """
        Latitudes
        """

        if self.latitudes is not None:
            return self.latitudes
        else:
            path = next(iter(self.var_databases_.values()))
            self.latitudes = self._open_ds_at_path(path).latitude.values
            return self.latitudes

    def get_longitudes(self):
        """
        Longitudes
        """

        if self.longitude is not None:
            return self.longitude
        else:
            path = next(iter(self.var_databases_.values()))
            self.longitude = self._open_ds_at_path(path).longitude.values
            return self.longitude

    def get_variable_at_time(self, var_name, time_stamp):
        """
        Get a variable at specified time
        """
        var_name_xr = self.var_to_xr_map.get(var_name, var_name)
        vals = (
            getattr(self._open_ds_at_path(self.var_databases_[var_name]), var_name_xr)
            .loc[time_stamp, :, :]
            .values
        )

        return vals

    def get_request(self, req_idx):
        """
        Get request from idx
        """
        return self.config_json["requests"][req_idx]

    def get_field_size(self):
        """
        Field size (they should be all the same)
        """
        return self.get_latitudes().size, self.get_longitudes().size

    def get_dataset_for_var(self, var_name):
        """
        Get DS handle this var belongs to
        """
        return self.var_databases_[var_name]

    @classmethod
    def validate_json(cls, js):
        """
        Do validation of a dictionary that has been loaded from (or will be written to) a JSON
        """
        _schema = read_json(cls.schema_json)
        jsonschema.validate(js, _schema, format_checker=jsonschema.FormatChecker())

    @staticmethod
    def _open_ds_at_path(path):
        """
        Simply open a xarray ds in path
        """
        ds = xr.open_mfdataset(path, engine="cfgrib")
        return ds
