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


from makaniino.tracks_sources.ecmwf_tracker import TracksSource_ECMWFTracks
from makaniino.tracks_sources.ibtracks import (
    TracksSource_IBTracks,
    TracksSource_IBTracks_All,
)

tracks_factory = {
    "ibtracks": TracksSource_IBTracks,
    "ibtracks_all": TracksSource_IBTracks_All,
    "ecmwf_tracker": TracksSource_ECMWFTracks,
}
