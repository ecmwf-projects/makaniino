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


class SchemaDescription(object):
    """
    Provide a description of a JSON schema for user consumption
    """

    @staticmethod
    def from_schema(schema, depth=0):

        # check for required children needed (for draft v4)
        children_req = schema.get("required", None)
        # print(json.dumps(schema, indent=2))

        if children_req is not None:
            for k in children_req:
                schema["properties"][k]["required_flag"] = True

        if "enum" in schema:
            return EnumSchemaDescription(schema, depth)

        if "oneOf" in schema:
            return OneOfSchemaDescription(schema, depth)

        if isinstance(schema["type"], list):
            return MultiTypeSchemaDescription(schema, depth)

        return SchemaDescription.type_lookup()[schema["type"]](schema, depth)

    @staticmethod
    def from_path(schema_path):
        """
        From JSON path
        """

        with open(schema_path, "r") as f:
            schema = json.load(f)

        return SchemaDescription.from_schema(schema)

    def __init__(self, schema, depth):

        self._description = schema.get("description", "(none)")
        self.title = schema.get("title", None)
        self.required = schema.get("required_flag", False)
        self.depth = depth

    def __bytes__(self):
        return str(self).encode("utf-8")

    def __str__(self):
        s = ""
        if self.title:
            s += "{}\n{}\n".format(self.title, ("-" * len(self.title)))

        if self.description:
            s += "# {}\n".format(self.description)
        s += self.details_string()
        if self.title:
            s += "\n--------------------------------------------------"
        return s

    @property
    def newline(self):
        return "\n" + "    " * self.depth

    @property
    def description(self):
        if self._description:
            return "{}{}".format(
                "(REQUIRED) " if self.required else "", self._description
            )
        else:
            return self.newline

    @staticmethod
    def type_lookup():
        # This is a method so that the types are defined when it is executed
        return {
            "object": ObjectSchemaDescription,
            "array": ArraySchemaDescription,
            "number": NumberSchemaDescription,
            "string": StringSchemaDescription,
            "integer": IntegerSchemaDescription,
            "boolean": BooleanSchemaDescription,
        }

    def details_string(self):
        raise NotImplementedError


class ObjectSchemaDescription(SchemaDescription):
    def __init__(self, schema, depth):
        super(ObjectSchemaDescription, self).__init__(schema, depth)

        self.properties = {
            k: SchemaDescription.from_schema(v, depth + 1)
            for k, v in schema.get("properties", {}).items()
        }
        self.properties.update(
            {
                k: SchemaDescription.from_schema(v, depth + 1)
                for k, v in schema.get("patternProperties", {}).items()
            }
        )

    def details_string(self):
        if len(self.properties) == 0:
            properties = ""
        else:
            properties = self.newline * 2
            properties += ",{0}{0}".format(self.newline).join(
                "    # {}{}    {}: {}".format(
                    "{}    # ".format(self.newline).join(
                        prop_schema.description.split("\n")
                    ),
                    self.newline,
                    prop,
                    prop_schema.details_string(),
                )
                for prop, prop_schema in self.properties.items()
            )
            properties += self.newline

        return "{{{}}}".format(properties)


class ArraySchemaDescription(SchemaDescription):
    def __init__(self, schema, depth):
        super(ArraySchemaDescription, self).__init__(schema, depth)

        if schema.get("items", None) is not None:
            self.items = SchemaDescription.from_schema(schema["items"], depth + 1)
        else:
            self.items = None

    def details_string(self):
        if self.items is not None:
            item = self.items.details_string()
            return "[{}, ...]".format(item)
        else:
            return "[]"


class NumberSchemaDescription(SchemaDescription):
    def details_string(self):
        return "<number>"


class IntegerSchemaDescription(SchemaDescription):
    def details_string(self):
        return "<integer>"


class BooleanSchemaDescription(SchemaDescription):
    def details_string(self):
        return "<boolean>"


class StringSchemaDescription(SchemaDescription):
    def __init__(self, schema, depth):
        super(StringSchemaDescription, self).__init__(schema, depth)
        self.format = schema.get("format", None)

    def details_string(self):
        if self.format is None:
            fmt = ""
        elif self.format == "date-time":
            fmt = " (rfc3339)"
        else:
            fmt = " {}".format(self.format)
        return "<string>" + fmt


class EnumSchemaDescription(SchemaDescription):
    def __init__(self, schema, depth):
        super(EnumSchemaDescription, self).__init__(schema, depth)
        self.values = schema["enum"]

    @staticmethod
    def value_string(v):
        if isinstance(v, str):
            return '"{}"'.format(v)
        else:
            return "{}".format(v)

    def details_string(self):

        if len(self.values) == 1:
            return self.value_string(self.values[0])
        else:
            return "{" + ", ".join(self.value_string(v) for v in self.values) + "}"


class OneOfSchemaDescription(SchemaDescription):
    def __init__(self, schema, depth):
        super(OneOfSchemaDescription, self).__init__(schema, depth)
        self.values = schema["oneOf"]

        # self.properties = {kk: k for kk, k in enumerate(self.values)}

        self.properties = {
            "<OPTION-{}>".format(k): SchemaDescription.from_schema(v, depth + 1)
            for k, v in enumerate(self.values)
        }

    @staticmethod
    def value_string(v):
        if isinstance(v, str):
            return '"{}"'.format(v)
        else:
            return "{}".format(v)

    def details_string(self):
        if len(self.properties) == 0:
            properties = ""
        else:
            properties = self.newline * 2
            properties += ",{0}{0}".format(self.newline).join(
                "    # {}{}    {}: {}".format(
                    "{}    # ".format(self.newline).join(
                        prop_schema.description.split("\n")
                    ),
                    self.newline,
                    prop,
                    prop_schema.details_string(),
                )
                for prop, prop_schema in sorted(self.properties.items())
            )
            properties += self.newline

        return "{{{}}}".format(properties)


class MultiTypeSchemaDescription(SchemaDescription):
    def __init__(self, schema, depth):
        super(MultiTypeSchemaDescription, self).__init__(schema, depth)
        self.sub_schemas = [
            self.type_lookup()[t](schema, depth) for t in schema["type"]
        ]

    @staticmethod
    def value_string(v):
        if isinstance(v, str):
            return '"{}"'.format(v)
        else:
            return "{}".format(v)

    def details_string(self):

        if len(self.sub_schemas) == 1:
            return self.sub_schemas[0].details_string()
        else:
            return "{" + ", ".join(s.details_string() for s in self.sub_schemas) + "}"
