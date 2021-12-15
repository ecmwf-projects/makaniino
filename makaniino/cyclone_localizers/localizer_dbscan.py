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

import numpy as np
from sklearn.cluster import DBSCAN

from makaniino.cyclone_localizers.localizer import CycloneLocalizer, logger


class CycloneLocalizer_DBSCAN(CycloneLocalizer):
    """
    Uses DBSCAN algorithm
    """

    def __init__(self, y_data, label_threshold=0.6, cluster_crop_pxl=10):

        super().__init__(y_data)

        self.label_threshold = label_threshold

        # are around predicted cluster ctr, on which
        # the value of the prediction is sampled..
        self.cluster_crop_pxl = cluster_crop_pxl

    def find(self, label_radius=25):
        """
        Apply kmeans to find makaniino centers
        """

        # points above threshold
        eye_mask = np.argwhere(self.y_data > self.label_threshold)
        if eye_mask.shape[0] <= 1:
            logger.info("not enough points to attempt clustering..")
            return None

        eye_weights = self.y_data[eye_mask[:, 0], eye_mask[:, 1]]
        dbscan = DBSCAN(label_radius).fit(eye_mask, sample_weight=eye_weights)

        clusters = []
        for label in set(dbscan.labels_):

            if label != -1:
                label_idxs = np.where(dbscan.labels_ == label)
                cluster_x = np.mean(eye_mask[label_idxs, 0])
                cluster_y = np.mean(eye_mask[label_idxs, 1])
                clusters.append((cluster_x, cluster_y))

        return clusters
