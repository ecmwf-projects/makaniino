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

import json
import logging
from types import SimpleNamespace

logger = logging.getLogger(__name__)


class Params(SimpleNamespace):
    """
    Minimal class to mock "args" from user command line
    """

    pass


class Configurable:
    """
    Some basic configuration
    """

    # model config params
    default_params = {}

    def __init__(self, params):

        # user configuration (where all the configuration params
        # are expected to be defined as attributes)
        self.params = params

        # make sure that all the configurations
        # are consistent and complete as required
        self._check_user_config(params)

    @classmethod
    def from_params_dict(cls, params_dict):
        """
        Instantiate from dictionary of params
        """

        params = Params()

        for k, v in params_dict.items():
            setattr(params, k, v)

        return cls(params)

    @classmethod
    def get_default_params(cls):
        """
        Return the configuration options
        """
        return cls.default_params

    @classmethod
    def list_params(cls):
        """
        Return list of param descriptions
        """

        return {k: f"{v[1]} [{v[0]}]" for k, v in cls.default_params.items()}

    def get_current_params(self):
        """
        Return current config as dict
        """
        all_user_params = {arg: getattr(self.params, arg) for arg in vars(self.params)}

        return {k: v for k, v in all_user_params.items() if k in self.default_params}

    def get_param(self, param_name):
        """
        Return the value of a param
        """
        return getattr(self.params, param_name)

    def set_param(self, k, v):
        """
        Set a param
        """
        setattr(self.params, k, v)

    @classmethod
    def __str__(cls):
        return "\n".join(f"{k}: {v}" for k, v in cls.default_params.items())

    def _check_user_config(self, configs):
        """
        Make sure that the required user configs
        have been specified in configs
        """

        for conf in self.default_params.keys():

            try:
                assert getattr(configs, conf) is not None
            except AssertionError:
                logger.error(f"Configuration {conf} not specified but required!")

    @classmethod
    def config_type(cls, val):
        """
        Trying to find the correct type from string..
        """

        # rather permissive..
        try:
            _value = json.loads(str(val).lower())
        except (TypeError, json.decoder.JSONDecodeError):
            return str
        return type(_value)
