#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Thu Jan 27 15:32:09 2022
@author: Youssef Ayachi

The present module aims to reproduce the homogeneity report generated by
MetroloJ, an ImageJ plugin.
Given a single image .tif file, this module will produce the following
elements:
    1. normalized intensity profile of the image.
    2. microscopy info dataframe
    3. intensity plot: distribution of pixel values of the the mid horizontal,
    mid vertical and the two diagonal lines of the image.
    4. dataframe showing the intensity values of 9 specific pixels and
    their ratio over the maximum intensity value of the array.

Note: Code tested on one image .tif file (from homogeneity samples)
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from skimage import draw
from skimage.measure import regionprops

from .common import get_images_from_multi_tiff, get_microscopy_info


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


def get_max_intensity_region_table(img):
    """
    this function finds the max intensity area of the given image
    in order to figure out the number of pixels,the center of mass and
    the max intensity of the corresponding area.

    Parameters
    ----------
    img : np.array.
        2d np.array.

    Returns
    -------
    center_of_mass: dict
        dict encolsing the number of pixels, the coordinates of the
        center of mass of the and the max intensity value of the max intensity
        area of the provided image.

    """

    max_intensity = np.max(img)

    # define the maximum intensity
    threshold_value = max_intensity-1

    # label pixels with max intesity values: binary matrix.
    labeled_foreground = (img > threshold_value).astype(int)

    # identify the region of max intensity
    properties = regionprops(labeled_foreground, img)

    # identify the center of mass of the max intensity area
    center_of_mass = (int(properties[0].centroid[0]),
                      int(properties[0].centroid[1]))

    # number of pixels of max intensity region
    nb_pixels = properties[0].area

    # organize info in dataframe
    max_region_info = {
        "nb pixels": [nb_pixels],
        "center of mass": [center_of_mass],
        "max intensity": [max_intensity]
        }

    return max_region_info


def get_norm_intensity_profile(img, save_path=""):
    """
    plots the normalized intensity profile of the image.
    the center of mass of the max intensity area is marked in red.
    If save_path is not empty, the generated figure will be saved as png in
    the provided path.

    Parameters
    ----------
    img : np.arrray
        image on a 2d np.array format.
    save_path : str, optional
        path to save the generated figure including filename.
        The default is "".

    Returns
    -------
    fig : matplotlib.figure.Figure
        returns the normalized intensity profile of the image with
        the center of mass of the max intensity area marked in red.

    """

    # normalized intensity array of the given image
    norm_intensity_profile = get_norm_intensity_matrix(img)
    # coordinates of center of mass of mac intensity area
    x_mass, y_mass = get_max_intensity_region_table(img)["center of mass"][0]

    # figure construction
    fig, ax = plt.subplots()
    ax.scatter(y_mass, x_mass, s=60, color="r", marker='+')
    plt.imshow(norm_intensity_profile)
    plt.colorbar()
    plt.title("normalized intensity profile", figure=fig)
    if save_path:
        plt.savefig(str(save_path),
                    bbox_inches='tight')

    return fig


# 3. intensity profiles


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


def get_x_axis(y_axis):
    """
    get x axis values for the intensity plot given y values.

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


def get_intensity_plot(img, save_path=""):
    """
    get the distribution of pixel intensities of the mid
    vertical, mid horizontal and the two diagonal lines of a given image.
    the vertical line y=0 on the plot represent to the image center.
    If save_path is not empty, the generated figure will be saved as png in
    the provided path.

    Parameters
    ----------
    img : np.array
        image on a 2d np.array format.
    save_path : str, optional
        path to save the generated figure inluding file name.
        The default is "".

    Returns
    -------
    fig : matplotlib.figure.Figure
        distribution of pixel intensities of the mid vertical, mid horizontal
        and the two diagonal lines of a given image.
        the vertical line y=0 on the plot represent to the image center.

    fig_data : dict
        dict representing the data used to generate the fig.
        the 8 keys are organised by pair with x axis and y axis data:
            - x_axis_V_seg and y_axis_V_seg
            - x_axis_H_seg and y_axis_H_seg
            - x_axis_diagUD and y_axis_diagUD
            - x_axis_diagDU and y_axis_diagDU

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

    # plot data into pandas array
    fig_data = {}
    fig_data["x_axis_V_seg"] = get_x_axis(V_seg)
    fig_data["y_axis_V_seg"] = V_seg

    fig_data["x_axis_H_seg"] = get_x_axis(H_seg)
    fig_data["y_axis_H_seg"] = H_seg

    fig_data["x_axis_diagUD"] = get_x_axis(diagUD)
    fig_data["y_axis_diagUD"] = diagUD

    fig_data["x_axis_diagDU"] = get_x_axis(diagDU)
    fig_data["y_axis_diagDU"] = diagDU

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

    if save_path:
        plt.savefig(str(save_path),
                    bbox_inches='tight')

    return fig, fig_data


# 4. profile statistics


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
    profiles_statistics : dict
        dict showing the intensity values of the concerned 9 pixels and
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

    return profiles_statistics_dict


# all report elements functions.


def cv_report(
        tiff_path,
        output_dir=None,
        microscope_type="NA",
        wavelength="NA",
        NA="NA",
        sampling_rate="NA",
        pinhole="NA"
        ):
    """
    Generate the different componenent of the homogeneity report and stock them
    in a list.

    Parameters
    ----------
    tiff_path : str
        path of .tif single image file.
    output_dir : str, optional
        if specified, all returns are saved in the corresponding directory.
        default is None.
    microscope_type : str
        type of microscope.
    wavelength : float
        In nm.
    NA : int or float
        Numerical aperture.
    sampling_rate : str
        In number of pixels. Ex: "1.0x1.0x1.0".
    pinhole : int or float
        In airy units.

    Returns
    -------
    homo_report_elements : list
        List of all the homogeneity report elements:
            1. normalized intensity profile of the image.
            2. np.array used to generate the normalized intensity profile.
            3. microscopy info dataframe
            4. intensity plot of the mid horizontal, mid vertical and the two
            diagonal lines of the image.
            5. np.array used to generate the intensity plot of the mid
            horizontal, mid vertical and the two diagonal lines of the image.
            6. dataframe showing the intensity values of 9 specific pixels and
            their ratio over the maximum intensity value of the array.


    """
    # we assume that .tif images for homogeneity carry one single image
    img = get_images_from_multi_tiff(tiff_path)

    # 1. get normalized intensity profile
    norm_intensity_profile = get_norm_intensity_profile(img)
    norm_intensity_data = get_norm_intensity_matrix(img)

    # 2. get microscopy info
    microscopy_info_table = get_microscopy_info(
        microscope_type, wavelength, NA, sampling_rate, pinhole
        )

    # 3. get centers' locations
    max_intensity_region_table = get_max_intensity_region_table(img)

    # 4. get intensity profiles
    intensity_plot, intensity_plot_data = get_intensity_plot(img)

    # 5. get profiles statistics
    profile_stat_table = get_profile_statistics_table(img)

    homo_report_elements = [
        norm_intensity_profile,
        pd.DataFrame(norm_intensity_data),
        pd.DataFrame(max_intensity_region_table),
        microscopy_info_table,
        intensity_plot,
        pd.DataFrame(intensity_plot_data),
        pd.DataFrame(profile_stat_table)
        ]

    if output_dir is not None:
        # .png : Normalized Intensity Profile and Intesity Profile plot
        norm_intensity_profile.savefig(
            output_dir+"norm_intensity_profile.png",
            format="png",
            bbox_inches='tight'
            )
        intensity_plot.savefig(
            output_dir+"intensity_plot.png", format="png", bbox_inches="tight"
                            )

        # .csv : Microscopy Info and profile_stat_table
        norm_intensity_data.to_csv(output_dir+"norm_intensity_data.csv")
        max_intensity_region_table.to_csv(output_dir+"max_region_table.csv")
        microscopy_info_table.to_csv(output_dir+"microscopy_info_table.csv")
        intensity_plot_data.to_csv(output_dir+"intensity_plot_data.csv")
        profile_stat_table.to_csv(output_dir+"profile_stat_table.csv")

    else:
        return homo_report_elements
