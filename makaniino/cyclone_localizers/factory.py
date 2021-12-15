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

from makaniino.cyclone_localizers.localizer_dbscan import CycloneLocalizer_DBSCAN
from makaniino.cyclone_localizers.localizer_kmeans import CycloneLocalizer_Kmeans

cyclone_localizer_factory = {
    "kmeans": CycloneLocalizer_Kmeans,
    "dbscan": CycloneLocalizer_DBSCAN,
}
