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

import os

import jsonschema

from makaniino.data_preprocess import factory as preprocessing_factory
from makaniino.data_preprocess.base import logger
from makaniino.data_preprocess.source_data import SourceData
from makaniino.utils.generic_utils import read_json


class PreProcessingRunner:
    """
    Handles pre-processing requests from a config file
    """

    schema_json = os.path.join(os.path.dirname(__file__), "schema_preprocess.json")

    def __init__(self, config_json):

        # pre-process configuration
        self.config_json = config_json

    @classmethod
    def from_json_path(cls, json_path):
        """
        From config path
        """

        config_json = read_json(json_path)

        # validate the json against the schema
        cls.validate_json(config_json)

        return cls(config_json)

    def run_preprocessing(self):
        """
        Run the pre-processing
        """

        logger.info("Running pre-processing..")

        for pre_proc in self.config_json["datasets"]:

            # Lazily load the required source data
            source_data = SourceData.from_json_path(
                pre_proc["source_data"]["config_path"]
            )
            source_data.open_handle(open_vars=pre_proc["source_data"]["requested_vars"])

            for var in source_data.var_databases_.keys():
                print(f"****** VAR: {var}: \n{source_data.var_databases_[var]}\n")

            # Data processor configuration
            processor_class = preprocessing_factory.factory[pre_proc["pre_processor"]]
            data_processor = processor_class(
                source_data,
                pre_proc["output_dir"],
                include_latlon=pre_proc.get("include_latlon"),
                ibtracks_path=pre_proc.get("ibtracks_path"),
                process_ibtracks=pre_proc.get("process_ibtracks"),
                tracks_source_tag=pre_proc.get("tracks_source_tag"),
                center_mark=pre_proc.get("center_mark"),
                labelling_method=pre_proc.get("labelling_method"),
                n_procs=pre_proc.get("n_processes"),
                chunk_size=pre_proc.get("chunk_size"),
                field_size=pre_proc.get("field_size"),
            )

            # run the data pre-processing
            data_processor.run(pre_proc["start_date"], pre_proc["end_date"])

    @classmethod
    def validate_json(cls, js):
        """
        Do validation of a dictionary that has been loaded from (or will be written to) a JSON
        """
        _schema = read_json(cls.schema_json)
        jsonschema.validate(js, _schema, format_checker=jsonschema.FormatChecker())
