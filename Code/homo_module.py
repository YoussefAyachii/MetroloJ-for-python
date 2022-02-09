#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 15:32:09 2022

@author: bottimacintosh
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from skimage import draw

import common_module as cm  # local module


"""
Homogeneity report
Code tested on one image .tif file (from homogeneity samples)
"""


# 1. normalized intensity report


def get_norm_intensity_matrix(img):
    """
    get normalized intensity matrix: divide all the pixels' intensity
    by the maximum intensity.

    Parameters
    ----------
    img : np.array
        image on a 2d np.array format.

    Returns
    -------
    norm_intensity_profile : np.array
        2d np.array where pixel values are scaled by the max intensity of
        the original image.

    """

    max_intensity = np.max(img)
    # the rule of three : max_intensity->100%, pixel_intensity*100/max
    norm_intensity_profile = np.round(img/max_intensity * 100)
    return norm_intensity_profile


def get_norm_intensity_profile(img):
    """
    plots the normalized intensity profile of the image

    Parameters
    ----------
    img : np.arrray
        image on a 2d np.array format.

    Returns
    -------
    fig : matplotlib.figure.Figure
        returns the normalized intensity profile of the image

    """

    norm_intensity_profile = get_norm_intensity_matrix(img)

    fig = plt.figure()
    plt.imshow(norm_intensity_profile)
    plt.colorbar()
    plt.title("normalized intensity profile", figure=fig)

    return fig


"""
# ex:
img = get_images_from_multi_tiff(path_homo)[0]
get_norm_intensity_matrix(img)
get_norm_intensity_profile(img)
"""


# 2. report : microscopy info


def get_microscopy_info(
        Microscope_type, Wavelength, NA, Sampling_rate, Pinhole
        ):
    info_values = [Microscope_type, Wavelength,
                   NA, Sampling_rate, Pinhole]
    info_labels = ["Microscope type", "Wavelength",
                   "NA", "Sampling rate", "Pinhole"]

    # add units to info_labels
    info_units = ["", "(nm)", "", "pixel", "(airy units)"]
    info_labels = [i+" "+info_units[info_labels.index(i)] for i in info_labels]

    MicroscopyInfo_dict = {}
    MicroscopyInfo_dict["Labels"] = info_labels
    MicroscopyInfo_dict["Values"] = info_values

    MicroscopyInfo = pd.DataFrame(MicroscopyInfo_dict)
    return MicroscopyInfo


"""
# ex:
get_microscopy_info("example", 1, 2, 3, 4)
"""


# 3. centers' locations IN PROGRESS


# 4. intensity profiles


def get_pixel_values_of_line(img, x0, y0, xf, yf):
    """
    get the value of a line of pixels.
    the line defined by the user using the corresponding first and last
    pixel indices.

    Parameters
    ----------
    img : np.array.
        image on a 2d np.array format.
    x0 : int
        raw number of the starting pixel
    y0 : int
        column number of the starting pixel.
    xf : int
        raw number of the ending pixel.
    yf : int
        column number of the ending pixel.

    Returns
    -------
    line_pixel_values : np.array
        1d np.array representing the values of the chosen line of pixels.

    """
    rr, cc = np.array(draw.line(x0, y0, xf, yf))
    # line_pixel_values = [img[rr[i], cc[i]] for i in range(len(rr))]
    line_pixel_values = img[rr, cc]
    return line_pixel_values


"""
# ex:
img = get_images_from_multi_tiff(path_homo)[0]
get_pixel_values_of_line(img, 0, 0, 200, 200)
"""


def get_x_axis(y_axis):
    """
    get x axis values for the intensity plot.

    Parameters
    ----------
    y : np.array
        1d np.array representing the y axis values of the intensity plot.

    Returns
    -------
    x_axis : np.array
        x axis values for the intensity plot.

    """
    nb_pixels = len(y_axis)
    # center the pixel value vector around 0
    x_axis = np.arange(round(-nb_pixels/2), round(nb_pixels/2+1), 1)
    # the center of the matrix is 4 pixels not one
    x_axis = x_axis[x_axis != 0]
    return x_axis


"""
# ex:
img = get_images_from_multi_tiff(path_homo)[0]
img_vec = get_pixel_values_of_line(img, 0, 0, 200, 200)
get_x_axis(img_vec)
"""


def get_intensity_plot(img):
    """
    get the distribution of pixel intensities of the mid
    vertical, mid horizontal and the two diagonal lines of a given image.
    the vertical line y=0 on the plot represent to the image center.

    Parameters
    ----------
    img : np.array
        image on a 2d np.array format.

    Returns
    -------
    fig : matplotlib.figure.Figure
        distribution of pixel intensities of the mid vertical, mid horizontal
        and the two diagonal lines of a given image.
        the vertical line y=0 on the plot represent to the image center.

    """

    xmax, ymax = np.shape(img)
    xmax = xmax-1
    ymax = ymax-1
    xmid = round(xmax/2)
    ymid = round(ymax/2)
    # mid vertical pixel segment
    V_seg = get_pixel_values_of_line(img, x0=0, y0=ymid, xf=xmax, yf=ymid)
    # mid horizontal pixel segment
    H_seg = get_pixel_values_of_line(img, x0=xmid, y0=0, xf=xmid, yf=ymax)
    # diagonal UpDown Left Right
    diagUD = get_pixel_values_of_line(img, x0=0, y0=0, xf=xmax, yf=ymax)
    # diagonal DownUp Left Right
    diagDU = get_pixel_values_of_line(img, x0=xmax, y0=0, xf=0, yf=ymax)

    # plot
    fig = plt.figure()
    plt.plot(get_x_axis(V_seg), V_seg, color="b", label="V_seg", figure=fig)
    plt.plot(get_x_axis(H_seg), H_seg, color="g", label="H_seg", figure=fig)

    plt.plot(get_x_axis(diagUD), diagUD, color="r", label="Diag1", figure=fig)
    plt.plot(get_x_axis(diagDU), diagDU, color="y", label="Diag2", figure=fig)

    plt.axvline(0, linestyle='--')
    plt.title("Intensity Profiles", figure=fig)
    plt.xlim((min(get_x_axis(diagUD))-25, max(get_x_axis(diagUD))+25))
    plt.legend()

    return fig


"""
# ex:
img = get_images_from_multi_tiff(path_homo)[0]
type(get_intensity_plot(img))
"""

# 5. profile statistics


def get_profile_statistics_table(img):
    """
    given an image in a 2d np.array format, this function return the pixel
    intensity values of 9 specific pixels and their ratio over the maximum
    intensity. The 9 concerned pixels are:
        - top-left corner
        - upper-middle pixel
        - top-right corner
        - left-middle pixel
        - maximum intensity pixel
        - right-middle pixel
        - bottom-left corner
        - bottom-middle pixel
        - bottom-right corner

    Parameters
    ----------
    img : np.array
        image on a 2d np.array format.

    Returns
    -------
    profiles_statistics : pd.DataFrame
        dataframe showing the intensity values of the concerned 9 pixels and
        their ratio over the maximum intensity value of the array.

    """

    # find the maximum intensity and the corresponding pixel.
    max_intensity = np.max(img)
    xx_max, yy_max = np.where(img == max_intensity)

    # if max intensity is in >1 pixels, we chose only the first localization
    x_index_max_intensity = xx_max[0]
    y_index_max_intensity = yy_max[0]

    max_found_at = [x_index_max_intensity, y_index_max_intensity]

    # 3 by 3 grid going through each corner and the middle of each line:

    # tl, um, tr
    # lm, cc, rm
    # bl, bm, br

    xx, yy = np.meshgrid([0, img.shape[0] // 2, -1],
                         [0, img.shape[1] // 2, -1])
    max_intensities = img[xx, yy].flatten()
    max_intensities_relative = max_intensities/max_intensity

    # replace central pixel value with max intensity
    max_intensities[4] = max_intensity
    max_intensities_relative[4] = 1.0

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
    profiles_statistics_dict["intensity relative to max"] = \
        max_intensities_relative

    profiles_statistics = pd.DataFrame(profiles_statistics_dict)
    return profiles_statistics


"""
# ex:
img = get_images_from_multi_tiff(path_homo)[0]
get_profile_statistics_table(img)
"""


"""
REPORT
"""


def get_homogeneity_report_elements(
        path, Microscope_type, Wavelength, NA, Sampling_rate, Pinhole
        ):
    """
    Generate the different componenent of the homogeneity report and stock them
    in a list.

    Parameters
    ----------
    path : str
        path of .tif single image file.
    Microscope_type : str

    Wavelength : float
        In nm.
    NA : int or float
        Numerical aperture.
    Sampling_rate : str
        In number of pixels. Ex: "1.0x1.0x1.0".
    Pinhole : int or float
        In airy units.

    Returns
    -------
    HomogeneityReportComponents : list
        List of all the homogeneity report components:
            1. normalized intensity profile of the image.
            2. microscopy info dataframe
            3. intensity plot of the mid horizontal, mid vertical and the two
            diagonal lines of the image.
            4. dataframe showing the intensity values of 9 specific pixels and
            their ratio over the maximum intensity value of the array.


    """
    img = cm.get_images_from_multi_tiff(path)[0]

    # 1. get normalized intensity profile
    NormIntensityProfile = get_norm_intensity_profile(img)

    # 2. get microscopy info
    MicroscopyInfo = get_microscopy_info(
        Microscope_type, Wavelength, NA, Sampling_rate, Pinhole
        )

    # 3. get centers' locations

    # 4. get intensity profiles
    IntensityPlot = get_intensity_plot(img)

    # 5. get profiles statistics
    ProfileStatisticsTable = get_profile_statistics_table(img)

    HomogeneityReportComponents = [NormIntensityProfile,
                                   MicroscopyInfo,
                                   IntensityPlot,
                                   ProfileStatisticsTable]

    return HomogeneityReportComponents


"""
# ex1:
Homogeneity_report_elements_1 = get_homogeneity_report_elements(
    path_homo, "Confocal", 460, 1.4, "1.0x1.0x1.0", 1
    )
Homogeneity_report_elements_1[0]
Homogeneity_report_elements_1[1]
Homogeneity_report_elements_1[2]
Homogeneity_report_elements_1[3]
"""


def save_homogeneity_report_elements(
        tiff_path, output_dir, Microscope_type,
        Wavelength, NA, Sampling_rate, Pinhole
        ):

    HomogeneityReportElements = get_homogeneity_report_elements(
        tiff_path, Microscope_type, Wavelength, NA, Sampling_rate, Pinhole
        )

    NormIntensityProfile = HomogeneityReportElements[0]
    MicroscopyInfo = HomogeneityReportElements[1]
    IntensityPlot = HomogeneityReportElements[2]
    ProfileStatisticsTable = HomogeneityReportElements[3]

    # .png : Normalized Intensity Profile and Intesity Profile plot
    NormIntensityProfile.savefig(
        output_dir+"HistNbPixelVSGrayScale.png",
        format="png",
        bbox_inches='tight'
        )
    IntensityPlot.savefig(
        output_dir+"IntensityPlot.png", format="png", bbox_inches="tight"
                         )

    # .csv : Microscopy Info and ProfileStatisticsTable
    MicroscopyInfo.to_csv(output_dir+"MicroscopyInfo.csv")
    ProfileStatisticsTable.to_csv(output_dir+"ProfileStatisticsTable.csv")


"""
# ex1:
save_homogeneity_report_elements(
    path_homo, output_dir_hom, "Confocal", 460, 1.4, "1.0x1.0x1.0", 1
    )
"""
