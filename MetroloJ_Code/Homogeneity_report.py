#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 15:32:09 2022

@author: bottimacintosh
"""

from itertools import product

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
from skimage import draw

"""
Paths for execution
"""

path5 = "/Users/bottimacintosh/Documents/M2_CMB/IBDML/Data/homogeneity/lame/homogeneite10zoom1-488.tif"


"""
Homogeneity report
Code tested on .tif file from homogeneity samples

Remarks:
- Centers' locations table (report): in progress
"""


# 1. report: normalized intensity report
def get_norm_intensity_matrix(path):  # used in both n.i.profile and centers' locations
    """get normalized intensity matrix (in %) : divide all the pixels' intensity by the max_intensity"""

    img = np.asanyarray(Image.open(path))

    max_intensity = np.max(img)
    # the rule of three : max_intensity->100%, pixel_intensity*100/max
    norm_intensity_profile = np.round(img / max_intensity * 100)
    return norm_intensity_profile


# get normalized intensity profile (plot)
def get_norm_intensity_profile(path):
    norm_intensity_profile = get_norm_intensity_matrix(path)

    fig = plt.figure()
    plt.imshow(norm_intensity_profile)
    plt.colorbar()
    plt.title("normalized intensity profile", figure=fig)

    return fig


# ex
get_norm_intensity_matrix(path5)
get_norm_intensity_profile(path5)

# 2. report : microscopy info (copied from c_v_report)

##


def get_microscopy_info(microscope_type, wavelength, Na, sampling_rate, pinhole):

    info_values = [microscope_type, wavelength, Na, sampling_rate, pinhole]
    info_labels = ["microscope type", "wavelength", "Na", "sampling rate", "pinhole"]

    # add units to info_labels
    info_units = ["", "(nm)", "", "pixel", "(airy units)"]

    # use format strings

    info_labels = [i + " " + info_units[info_labels.index(i)] for i in info_labels]

    microscopy_info_dict = {}
    microscopy_info_dict["labels"] = info_labels
    microscopy_info_dict["values"] = info_values

    microscopy_info = pd.DataFrame(microscopy_info_dict)
    return microscopy_info


# ex
get_microscopy_info("example", 1, 2, 3, 4)

# 3. report: centers' locations IN PROGRESS


# 4. report: intensity profiles


def get_pixel_values_of_line(path, x0, y0, xf, yf):

    image_2d = np.array(Image.open(path))

    rr, cc = np.array(draw.line(x0, y0, xf, yf))
    line_pixel_values = image_2d[rr, cc]
    return line_pixel_values


def get_xaxis(line):
    return np.linspace(-line.size // 2, line.size // 2, line.size) + 0.5


def get_intensity_plot(path):
    image_2d = np.array(Image.open(path))

    xmax, ymax = np.shape(image_2d)
    xmax = xmax - 1  # -1 : python starts from 0
    ymax = ymax - 1
    xmid = round(xmax / 2)
    ymid = round(ymax / 2)
    v_seg = get_pixelvalues_ofline(path, x0=0, y0=ymid, xf=xmax, yf=ymid)
    h_seg = get_pixel_values_of_line(path, x0=xmid, y0=0, xf=xmid, yf=ymax)
    diag_ud = get_pixel_values_of_line(
        path, x0=0, y0=0, xf=xmax, yf=ymax
    )  # diag up_down left right
    diag_du = get_pixel_values_of_line(
        path, x0=xmax, y0=0, xf=0, yf=ymax
    )  # diag down_up left right

    # plot
    fig = plt.figure()
    plt.plot(get_xaxis(v_seg), v_seg, color="b", label="v_seg", figure=fig)
    plt.plot(get_xaxis(h_seg), h_seg, color="g", label="h_seg", figure=fig)

    plt.plot(get_xaxis(diag_u_d), diag_ud, color="r", label="diag1", figure=fig)
    plt.plot(get_xaxis(diag_d_u), diag_du, color="y", label="diag2", figure=fig)

    plt.axvline(0, linestyle="--")
    plt.title("intensity profiles", figure=fig)
    plt.xlim(
        (min(get_xaxis(diag_u_d)) - 25, max(get_xaxis(diag_u_d)) + 25)
    )  # 25 subjective choice
    plt.legend()

    return fig


# ex
get_intensity_plot(path5)

# 5. report: profile statistics


def get_profile_statistics_table(path):

    img = np.asanyarray(Image.open(path))

    # 1. find the maximum intensity and the corresponding pixel.
    max_intensity = np.max(img)
    xx_max, yy_max = np.where(img == max_intensity)
    nb_pixels_with_max_intensity = xx_max.size
    # we chose only the first localization if the max intensity is in >1 pixels
    x_index_max_intensity = xx_max[0]
    y_index_max_intensity = yy_max[0]

    # nb_pixels_with_max_intensity = len(np.where(img==max_intensity)[0]) #optional
    # x_index_max_intensity=np.where(img==max_intensity)[0][0]
    # y_index_max_intensity=np.where(img==max_intensity)[1][0]

    max_found_at = [x_index_max_intensity, y_index_max_intensity]

    # 3 by 3 grid going through each corner and the middle of each line:

    # tl, um, tr
    # lm, cc, rm
    # bl, bm, br

    xx, yy = np.meshgrid([0, img.shape[0] // 2, -1], [0, img.shape[1] // 2, -1])
    max_intensities = img[xx, yy].flatten()
    max_intensities_relative = intensity / max_intensity

    # replace central values with max
    max_intensities[1, 1] = max_intensity
    max_intensities_relative[1, 1] = 1.0

    # build dictionnary
    profiles_statistics_dict = {}
    profiles_statistics_dict["location"] = [
        "top-left corner",
        "upper-middle pixel",
        "top-right corner",
        "left-middle pixel",
        f"maximum found at {max_found_at}",
        "right-middle pixel",
        "bottom-left corner",
        "bottom-middle pixel",
        "bottom-right corner",
    ]

    profiles_statistics_dict["intensity"] = max_intensities
    profiles_statistics_dict["intensity relative to max"] = max_intensities_relative

    profiles_statistics = pd.DataFrame(profiles_statistics_dict)
    return profiles_statistics


# ex
get_profile_statistics_table(path5)

"""
#r_ep_ort
"""


def get_homogeneity_report_elements(
    path, microscope_type, wavelength, Na, sampling_rate, pinhole
):

    # 1. get normalized intensity profile
    norm_intensity_profile = getnorm_intensity_profile(path)

    # 2. get microscopy info
    microscopy_info = getmicroscopy_info(
        microscope_type, wavelength, Na, sampling_rate, pinhole
    )

    # 3. get centers' locations

    # 4. get intensity profiles
    intensity_plot = getintensity_plot(path)  # 2nd fct from c_v file

    # 5. get profiles statistics
    profile_statistics_table = getprofile_statistics_table(path)

    homogeneity_report_components = [
        norm_intensity_profile,
        microscopy_info,
        intensity_plot,
        profile_statistics_table,
    ]

    return homogeneity_report_components


# ex1
homogeneity_report_elements_1 = gethomogeneity_report_elements(
    path5, "confocal", 460, 1.4, "1.0x1.0x1.0", 1
)
homogeneity_report_elements_1[0]
homogeneity_report_elements_1[1]
homogeneity_report_elements_1[2]
homogeneity_report_elements_1[3]


def save_homogeneity_report_elements(
    tiff_path, output_dir, microscope_type, wavelength, Na, sampling_rate, pinhole
):
    homogeneity_report_elements = get_homogeneity_report_elements(
        tiff_path, microscope_type, wavelength, Na, sampling_rate, pinhole
    )

    norm_intensity_profile = homogeneity_report_elements[0]
    microscopy_info = homogeneity_report_elements[1]
    intensity_plot = homogeneity_report_elements[2]
    profile_statistics_table = homogeneity_report_elements[3]

    # .png : normalized intensity profile and intesity profile plot
    norm_intensity_profile.savefig(
        output_dir + "_histnb_pixel_v_s_gray_scale.png",
        format="png",
        bbox_inches="tight",
    )
    intensity_plot.savefig(
        output_dir + "intensity_plot.png", format="png", bbox_inches="tight"
    )

    # .csv : microscopy info and profile_statistics_table
    microscopy_info.to_csv(output_dir + "microscopy_info.csv")
    profile_statistics_table.to_csv(output_dir + "profile_statistics_table.csv")


# ex1
output_dir_hom = "/users/bottimacintosh/documents/m2_cm_b/_i_bdml/metrolo_j-for-python/_homogeneity_output_files/"
save_homogeneity_report_elements(
    path5, output_dir_hom, "confocal", 460, 1.4, "1.0x1.0x1.0", 1
)

# ex1
output_dir_hom = "/Users/bottimacintosh/Documents/M2_CMB/IBDML/MetroloJ-for-python/Homogeneity_output_files/"
SaveHomogeneityReportElements(
    path5, output_dir_hom, "Confocal", 460, 1.4, "1.0x1.0x1.0", 1
)
