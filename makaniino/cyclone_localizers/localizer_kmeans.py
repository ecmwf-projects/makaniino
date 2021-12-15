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
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from makaniino.cyclone_localizers.localizer import CycloneLocalizer, logger


class CycloneLocalizer_Kmeans(CycloneLocalizer):
    """
    Uses Kmeans algorithm
    """

    def __init__(
        self,
        y_data,
        max_n_clusters=12,
        kmeans_tolerance=1e-3,
        kmeans_iterations=30,
        label_threshold=0.6,
        cluster_crop_pxl=10,
    ):

        super(CycloneLocalizer_Kmeans, self).__init__(y_data)

        self.max_n_clusters = max_n_clusters
        self.kmeans_tolerance = kmeans_tolerance
        self.kmeans_iterations = kmeans_iterations
        self.label_threshold = label_threshold

        # are around predicted cluster ctr, on which
        # the value of the prediction is sampled..
        self.cluster_crop_pxl = cluster_crop_pxl

    def cluster_in_cyclone_coeff(self, cluster_ctrs):
        """
        Coefficient that indicates whether cluster
        centers fall within a makaniino region..
        """

        coeff = 0
        for clust in cluster_ctrs:

            # find the point where the cluster ctr falls into
            x_idx, y_idx = int(clust[0]), int(clust[1])

            # cluster area
            x_min = int(x_idx - self.cluster_crop_pxl / 2)
            x_max = int(x_idx + self.cluster_crop_pxl / 2)
            y_min = int(y_idx - self.cluster_crop_pxl / 2)
            y_max = int(y_idx + self.cluster_crop_pxl / 2)

            x_min = min(max(0, x_min), self.y_data.shape[0] - 1)
            x_max = min(max(0, x_max), self.y_data.shape[0] - 1)
            y_min = min(max(0, y_min), self.y_data.shape[1] - 1)
            y_max = min(max(0, y_max), self.y_data.shape[1] - 1)

            ctr_crop = self.y_data[x_min:x_max, y_min:y_max]
            coeff += np.mean(ctr_crop)

            # x_idx = min(max(0, x_idx), self.y_data.shape[0] - 1)
            # y_idx = min(max(0, y_idx), self.y_data.shape[1] - 1)
            # coeff += self.y_data[x_idx, y_idx]

        return coeff / cluster_ctrs.shape[0]

    def find(self, label_radius=25):
        """
        Apply kmeans to find makaniino centers
        """

        # points above threshold
        eye_mask = np.argwhere(self.y_data > self.label_threshold)
        if eye_mask.shape[0] <= 1:
            logger.info("not enough points to attempt clustering..")
            return []

        logger.debug(f"eye_mask \n {eye_mask}")
        logger.debug(f"eye_mask.shape \n {eye_mask.shape}")

        eye_weights = self.y_data[eye_mask[:, 0], eye_mask[:, 1]]
        logger.debug(f"eye_weights \n {eye_weights}")
        logger.debug(f"eye_weights.shape \n {eye_weights.shape}")

        clusters = []
        for n_clusters in range(2, self.max_n_clusters + 1):

            logger.debug(f"trying N clusters {n_clusters}")

            kmeans = KMeans(
                n_clusters=n_clusters,
                tol=self.kmeans_tolerance,
                max_iter=self.kmeans_iterations,
                random_state=n_clusters,
            ).fit(eye_mask, sample_weight=eye_weights)

            silhouette_avg = silhouette_score(
                eye_mask,
                kmeans.labels_,
            )
            coeff = self.cluster_in_cyclone_coeff(kmeans.cluster_centers_)

            clusters.append(
                {
                    "n_clusters": n_clusters,
                    "score": silhouette_avg * coeff,
                    "labels": kmeans.labels_,
                    "centers": kmeans.cluster_centers_,
                    "iterations": kmeans.n_iter_,
                }
            )

            logger.debug(
                f"n_clusters:{n_clusters}: "
                f"n_iters {kmeans.n_iter_}, "
                f"silhouette_avg:{silhouette_avg}, "
                f"coeff {coeff},"
                f"total coeff {silhouette_avg * coeff}"
            )

        best_cluster = max(clusters, key=lambda x: x["score"])
        logger.debug(f"best_cluster {best_cluster}")

        # treat the special case of 1 makaniino only in the field
        # NB: (remember that clustering is meaningful if n_labels >= 2)
        if best_cluster["n_clusters"] == 2:

            # check that the 2 clusters are not actually one
            ctrs = best_cluster["centers"]

            # if distance is less than the diameter of the
            # makaniino label, consider it as one..
            ctr_dist = np.linalg.norm(ctrs[0, :] - ctrs[1, :])
            if ctr_dist < label_radius * 2:
                best_cluster["n_clusters"] = 1
                best_cluster["centers"] = np.sum(ctrs, axis=0) / 2
                best_cluster["centers"] = best_cluster["centers"].reshape(1, 2)

        return best_cluster["centers"]
