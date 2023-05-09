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

from setuptools import find_packages, setup

#    "tensorflow>=2.2.0",

# required packages
install_requirements = [
    "packaging==21.0",
    "numpy==1.19.5",
    "zarr==2.10.2",
    "mpi4py==3.1.1",
    "keras>=2.4.3",
    "tensorflow-gpu>=2.2.0",
    "horovod>=0.20.2",
    "matplotlib==3.4.3",
    "jsonschema==4.1.2",
    "climetlab==0.8.31",
    "jupyter==1.0.0",
    "scikit-learn==1.0.1",
]

# required packages for testing
test_requirements = ["pytest"]

setup(
    name="makaniino",
    version="0.1.0",
    description="Tropical Cyclone detection using Machine Learning",
    long_description="Tropical Cyclone detection using Machine Learning",
    keywords="tropical makaniino detection",
    author="NOAA, ECMWF",
    author_email="",
    license="TBD",
    packages=find_packages(),
    package_dir={"makaniino": "makaniino"},
    zip_safe=False,
    scripts=[
        "bin/mk-augment-data",
        "bin/mk-check-dataset",
        "bin/mk-diagnostics",
        "bin/mk-download",
        "bin/mk-preprocess",
        "bin/mk-evaluate-model",
        "bin/mk-format-download-config",
        "bin/mk-format-preprocess-config",
        "bin/mk-format-training-config",
        "bin/mk-keras-model",
        "bin/mk-predict",
        "bin/mk-train",
        "bin/mk-train-from-args",
        "bin/mk-version",
    ],
    include_package_data=True,
    install_requires=install_requirements,
    tests_require=test_requirements,
    test_suite="tests",
)
