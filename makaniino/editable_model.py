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


class EditableLayer:
    """
    A layer that operates on the raw json representation
    of a keras layer
    """

    def __init__(self, lay_json):

        self.lay_json = lay_json

    def set_config(self, config_key, config_val):
        """
        Generic method to set a layer config
        """
        self.lay_json["config"][config_key] = config_val

    def set_source_layers(self, source_layers):
        """
        Set source layers
        """
        _in_layers = [[lay, 0, 0, {}] for lay in source_layers]
        self.lay_json["inbound_nodes"] = [_in_layers]

    def add_source_layer(self, source_layer):
        """
        Add a source layer
        """
        self.lay_json["inbound_nodes"][0].append([source_layer, 0, 0, {}])

    @property
    def name(self):
        return self.lay_json["name"]

    def to_string(self):
        return json.dumps(self.lay_json)

    def to_dict(self):
        return self.lay_json


class EditableModel:
    """
    Edits a model
    """

    def __init__(self, model_json):

        # json model
        self._model_json = model_json

        # layers to be edited
        self.layers = [
            EditableLayer(lay) for lay in self._model_json["config"]["layers"]
        ]

    def layer(self, layer_name):
        """
        Return the layer according to name
        """

        return next((lay for lay in self.layers if lay.name == layer_name), None)

    def remove_layers(self, lay_ids):
        """
        Remove layers (either ids or names)
        """

        for lay_id in lay_ids:

            if type(lay_id) == int:
                self.layers.pop(lay_id)

            elif type(lay_id) == str:
                lay = next(lay for lay in self.layers if lay.name == lay_id)
                self.layers.remove(lay)

    def set_input_layer(self, lay_name):
        self._model_json["config"]["input_layers"] = [[lay_name, 0, 0]]

    def set_output_layer(self, lay_name):
        self._model_json["config"]["output_layers"] = [[lay_name, 0, 0]]

    def _build(self):
        layers_json = [lay.to_dict() for lay in self.layers]
        self._model_json["config"]["layers"] = layers_json

    def to_string(self):
        self._build()
        return json.dumps(self._model_json)
