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

import keras
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from makaniino.utils.debug_utils import find_layer_in_model

_colors = ["b", "k", "m", "r", "c"]


def normalize_np(data):

    max_ = np.max(data)
    min_ = np.min(data)

    if max_ == min_:
        return data / max_ if max_ else data
    else:
        return (data - min_) / (max_ - min_)


def plot_sample(
    input_field, ground_truth, output_dir="", title="sample", channel_idx=0
):
    fig_shape = (6, 10)

    # save sample picture
    fig = plt.figure(0, figsize=fig_shape)

    # ############## input_field
    input_field = input_field[:, :, channel_idx]
    fig.add_subplot(211)
    plt.imshow(normalize_np(input_field), cmap="gray")
    plt.title("Input field at t=0")

    # ############## labels
    fig.add_subplot(212)
    plt.imshow(ground_truth)
    plt.title("Ground truth")

    plt.savefig(
        os.path.join(output_dir, title.lower().replace(" ", "_") + "_sample.png")
    )
    plt.close()


def plot_prediction(
    input_field,
    prediction,
    ground_truth,
    cyc_coords_pxl_pred=None,
    cyc_coords_latlon_pred=None,
    cyc_coords_pxl_real=None,
    cyc_coords_latlon_real=None,
    cyc_classes=None,
    output_dir="",
    title="prediction",
    twod_only=False,
    plot_limits=None,
    vmin=None,
    vmax=None,
):
    fig_shape = (6, 10)

    # ######################################################################
    # ############################  2D  ####################################
    # ######################################################################

    # save sample picture
    fig = plt.figure(0, figsize=fig_shape)

    # ############## input_field
    ax = fig.add_subplot(311)
    plt.imshow(normalize_np(input_field), cmap="gray")

    if plot_limits:
        plt.xlim(plot_limits[0], plot_limits[1])
        plt.ylim(plot_limits[2], plot_limits[3])

    plt.title("Input field at t=0")

    # ############## labels
    ax = fig.add_subplot(312)
    plt.imshow(ground_truth, vmin=vmin, vmax=vmax)

    if plot_limits:
        plt.xlim(plot_limits[0], plot_limits[1])
        plt.ylim(plot_limits[2], plot_limits[3])

    if cyc_coords_pxl_real is not None:

        legend_string = []
        for cc, cyc in enumerate(cyc_coords_pxl_real):

            # point label
            lab_text = f"{cc}"

            # if tc classes provided, append it
            # to the label
            if cyc_classes is not None:
                lab_text += f" [{int(cyc_classes[0][cc])}]"

            ax.text(
                int(cyc[1]),
                int(cyc[0]),
                lab_text,
                fontsize=8,
                color="r",
                horizontalalignment="center",
                verticalalignment="center",
            )

            legend_string.append(
                f"{cc} "
                f"[{cyc_coords_latlon_real[cc, 0]:4.2f}, "
                f"{cyc_coords_latlon_real[cc, 1]:4.2f}]"
            )

        ax.text(
            0.025,
            0.05,
            "\n".join(legend_string),
            fontsize=8,
            color="k",
            bbox=dict(facecolor="white", alpha=1.0),
            horizontalalignment="left",
            verticalalignment="bottom",
            transform=ax.transAxes,
        )
    plt.title("Ground truth")

    # ############## prediction flat
    ax = fig.add_subplot(313)
    plt.imshow(prediction, vmin=vmin, vmax=vmax)

    if plot_limits:
        plt.xlim(plot_limits[0], plot_limits[1])
        plt.ylim(plot_limits[2], plot_limits[3])

    if cyc_coords_pxl_pred is not None:

        legend_string = []
        for cc, cyc in enumerate(cyc_coords_pxl_pred):
            ax.text(
                int(cyc[1]),
                int(cyc[0]),
                f"{cc}",
                fontsize=10,
                color="r",
                horizontalalignment="center",
                verticalalignment="center",
            )

            legend_string.append(
                f"{cc} "
                f"[{cyc_coords_latlon_pred[cc, 0]:4.2f}, "
                f"{cyc_coords_latlon_pred[cc, 1]:4.2f}]"
            )

        ax.text(
            0.025,
            0.05,
            "\n".join(legend_string),
            fontsize=8,
            color="k",
            bbox=dict(facecolor="white", alpha=1.0),
            horizontalalignment="left",
            verticalalignment="bottom",
            transform=ax.transAxes,
        )

    plt.title("Prediction")

    plt.savefig(os.path.join(output_dir, title.lower().replace(" ", "_") + "_2d.png"))
    plt.close()

    if not twod_only:
        # #######################################################################
        # ############################  3D  ####################################
        # #######################################################################

        fig = plt.figure(0, figsize=fig_shape)

        # ############## mesh
        x_len = ground_truth.shape[1]
        y_len = ground_truth.shape[0]
        print(f"x_len:{x_len}, y_len:{y_len}")
        X, Y = np.meshgrid(np.arange(x_len), np.arange(y_len))

        # ############## labels 3D
        ax = fig.add_subplot(211, projection=Axes3D.name)
        ax.plot_surface(
            X,
            Y,
            np.flip(ground_truth, axis=0),
            cmap=plt.cm.plasma,
            linewidth=0.5,
            edgecolors="k",
            antialiased=True,
        )
        ax.elev = 45
        ax.azim = -45

        ax.set_xlim([0, x_len])
        ax.set_ylim([0, y_len])
        ax.set_zlim([0, 2])

        plt.title("Ground truth - 3D")

        # ############## prediction 3D
        ax = fig.add_subplot(212, projection=Axes3D.name)
        ax.plot_surface(
            X,
            Y,
            np.flip(prediction, axis=0),
            cmap=plt.cm.plasma,
            linewidth=0.5,
            edgecolors="k",
            antialiased=True,
        )

        ax.elev = 45
        ax.azim = -45

        ax.set_xlim([0, x_len])
        ax.set_ylim([0, y_len])
        ax.set_zlim([0, 2])

        plt.title("Prediction - 3D")

        plt.savefig(
            os.path.join(output_dir, title.lower().replace(" ", "_") + "_3d.png")
        )
        # plt.show()
        plt.close()


def plot_filters(model, layer_name, model_name, out_path):
    """
    Plot filters of a layer
    """

    layer = find_layer_in_model(model, layer_name)

    # get weights
    try:
        filters, _ = layer.get_weights()
    except ValueError:
        # if there is no bias in the weights..
        filters = layer.get_weights()
        filters = np.asarray(filters[0])

    # filters shape:
    # (filter_x, filter_y, filter_depth, n_filters)
    filter_x, filter_y, filter_depth, n_filters = filters.shape

    # global normalization across all filters
    f_min = filters.min()
    f_max = filters.max()
    filters = (filters - f_min) / (f_max - f_min)

    # plot first few filters and first few depth
    n_filters_to_plot = min(10, n_filters)
    filter_depth_to_plot = min(5, filter_depth)

    # main frame
    fig = plt.figure()
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor="none", top=False, bottom=False, left=False, right=False)
    plt.grid(False)

    # title and axes
    plt.suptitle(
        f"Layer {layer_name} "
        f"filters of shape=({filter_x}, {filter_y}, {filter_depth}, {n_filters})"
    )
    plt.xlabel(f"{filter_depth_to_plot}/{filter_depth} filter 'depth' components")
    plt.ylabel(f"{n_filters_to_plot}/{n_filters} filters")

    idx = 1
    for i_filter in range(n_filters_to_plot):
        for i_depth in range(filter_depth_to_plot):
            # specify subplot and turn of axis
            ax = fig.add_subplot(n_filters_to_plot, filter_depth_to_plot, idx)
            ax.set_xticks([])
            ax.set_yticks([])
            plt.imshow(filters[:, :, i_depth, i_filter], cmap="gray")
            idx += 1

    # show the figure
    plt.savefig(os.path.join(out_path, model_name + "_filters.png"))
    plt.close()


def plot_feature_maps(model,
                      layer_name,
                      model_name,
                      out_path,
                      x_data,
                      max_features_plot=3):
    """
    Plot feature maps of a layer
    """

    layer = find_layer_in_model(model, layer_name)

    # get number of filters of this conv layer
    try:
        filters, _ = layer.get_weights()
    except ValueError:
        # if there is no bias in the weights..
        filters = layer.get_weights()
        filters = np.asarray(filters[0])

    _, _, _, n_filters = filters.shape

    layer_model = keras.models.Model(inputs=model.model.inputs, outputs=layer.output)

    feature_maps = layer_model.predict(x_data)
    print(f"layer_output.shape {feature_maps.shape}")

    # global normalization across all feature maps
    f_min = feature_maps.min()
    f_max = feature_maps.max()
    feature_maps = (feature_maps - f_min) / (f_max - f_min)

    # main frame
    fig = plt.figure()
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor="none", top=False, bottom=False, left=False, right=False)
    plt.grid(False)

    # title and axes
    plt.suptitle(
        f"First {max_features_plot * max_features_plot} "
        f"Feature Maps (row-order) of layer {layer_name} "
    )
    plt.xlabel("")
    plt.ylabel("")

    # plot max 64 feature maps..
    nmaps = min(max_features_plot, int(n_filters))

    if nmaps * nmaps > feature_maps.shape[-1]:
        print(
            f"Requested {nmaps * nmaps} plots, "
            f"but Feature map depth is {feature_maps.shape[-1]}. Skipping plots.."
        )
        return

    # only use the first sample of the batch
    idx_sample_in_batch = 0

    for idx_map in range(nmaps * nmaps):
        # specify subplot and turn of axis
        ax = fig.add_subplot(nmaps, nmaps, idx_map + 1)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.imshow(feature_maps[idx_sample_in_batch, :, :, idx_map], cmap="gray")

    # show the figure
    plt.savefig(os.path.join(out_path, model_name + "_maps.png"))
    plt.close()
