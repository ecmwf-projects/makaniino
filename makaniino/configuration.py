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

from makaniino import components

logger = logging.getLogger(__name__)


class Configuration:
    """
    A class that defines a configuration
    as several configured objects
    """

    def __init__(self, configurables_objs=None, configurables_classes=None):

        # NB configurables can be either objects or classes..
        self.configurables_objs = configurables_objs if configurables_objs else {}
        self.configurables_classes = (
            configurables_classes if configurables_classes else {}
        )

    @classmethod
    def from_classes(cls, configurables):
        """
        Expects a dictionary of Configurable classes
        """
        return cls(configurables_objs=None, configurables_classes=configurables)

    @classmethod
    def from_objects(cls, configurables):
        """
        Expects a dictionary of Configurable objects
        """
        return cls(
            configurables_objs=configurables,
            configurables_classes={k: type(v) for k, v in configurables.items()},
        )

    @classmethod
    def from_classes_json(cls, json_classes):

        configurables_classes = {k: components[v] for k, v in json_classes.items()}

        return cls(configurables_objs=None, configurables_classes=configurables_classes)

    def configure_all(self, args):
        """
        Configure all the classes in this configuration
        """
        self.configurables_objs = {
            name: conf_class(args)
            for name, conf_class in self.configurables_classes.items()
        }

    def get_configurable(self, config_name):
        """
        Get a configurable object
        """
        return self.configurables_objs.get(config_name)

    def get_all_params(self):
        """
        Only configurable objects have params
        """

        _params = {}
        for _, v in self.configurables_objs.items():
            _params.update(v.get_current_params())

        return _params

    def get_all_default_params(self):
        """
        Both configurable objects and classes can call default params
        """

        _params = {}
        for _, v in self.configurables_classes.items():
            _params.update(v.get_default_params())

        return _params

    def to_json(self):
        """
        A way to serialize this configuration
        """

        # find configurable class key in the configurables_map
        config_json = {}
        for k, v in self.configurables_objs.items():

            key = [kk for kk, vv in components.items() if vv == type(v)][0]

            config_json.update({k: {"name": key, "params": v.get_current_params()}})

        return config_json

    def to_file(self, f_name):
        """
        to a file
        """

        config_json = self.to_json()
        with open(f_name, "w") as f:
            json.dump(config_json, f, indent=4, sort_keys=True)

    def to_string(self):
        """
        to string
        """

        return json.dumps(self.to_json(), indent=4, sort_keys=True)

    @classmethod
    def from_json(cls, config_json):
        """
        Instantiate a configuration from a saved JSON configuration file
        """

        configurable_objs = {}

        for k, v in config_json.items():
            configurable_obj = components[v["name"]].from_params_dict(v["params"])
            configurable_objs.update({k: configurable_obj})

        return cls.from_objects(configurable_objs)

    @classmethod
    def from_file(cls, f_name):
        """
        Read a configuration from a configuration (JSON) file
        """
        with open(f_name, "r") as f:
            config_json = json.load(f)

        return cls.from_json(config_json)
