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

from makaniino.data_handling.data_handler import DataHandler_ZARR
from makaniino.models.cnn_fixed_filter import CNN_Fixed_Filter_Model
from makaniino.models.trans_learn_model import TRANS_LEARN_Model
from makaniino.models.unet_1hot import UNET_1HOT
from makaniino.models.unet_model import UNET_Model
from makaniino.models.unet_prefiltered import UNET_Prefiltered
from makaniino.training.trainer_parallel import TrainerParallel
from makaniino.training.trainer_serial import TrainerSerial

available_components = {
    "model": {
        "tl_model": TRANS_LEARN_Model,
        "unet_model": UNET_Model,
        "cnn_fixed_filter": CNN_Fixed_Filter_Model,
        "unet_prefiltered": UNET_Prefiltered,
        "unet_1hot": UNET_1HOT,
    },
    "data_handler": {
        "data_handler_zarr": DataHandler_ZARR,
    },
    "trainer": {"trainer_serial": TrainerSerial, "trainer_parallel": TrainerParallel},
}

components = {
    c_k: c_v for cg_k, cg_v in available_components.items() for c_k, c_v in cg_v.items()
}

components_flat_list = {
    cg_k: [k for k in cg_v.keys()] for cg_k, cg_v in available_components.items()
}

components_config_blocks = {
    c_k: {"name": c_k, "params": c_v.list_params()}
    for cg_k, cg_v in available_components.items()
    for c_k, c_v in cg_v.items()
}
