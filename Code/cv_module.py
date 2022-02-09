#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 14:23:09 2022

@author:YoussefAyachi
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.morphology import closing, square
from skimage.draw import polygon_perimeter

from PIL import Image

import common_module as cm

"""
CV report:
Code tested on one or multi image .tif file (from homogeneity and cv samples)
"""


# Get ROI (default central 20% of the original image) for a given 2d image


def get_roi_default(tiff_data):
    """
    Select the default Region Of Interest (ROI) from the initial image,
    e.i. select the central 20% of the whole np.array and return it.
    The returned arrays, one per image, are enclosed in a list.

    Parameters
    ----------
    tiff_data : list
        tiff_data is a list of np.arrays representing the image data.

    Returns
    -------
    list : list
        list of 2 components:
            1. dict enclosing info about the ROI
            2. list of ROIs pictures to display
    """

    ROI_info = {}
    ROI_data = []
    ROI_nb_pixels_list = []
    ROI_start_pixel_list = []
    ROI_end_pixel_list = []
    ROI_Original_ratio_list = []

    for i in range(len(tiff_data)):

        x, y = tiff_data[i].shape
        h, w = x*0.4, y*0.4

        startx, endx = int(x//2 - h//2), int(x//2 + h//2)
        starty, endy = int(y//2 - w//2), int(y//2 + w//2)

        ROI_data_temp = tiff_data[i][startx:endx, starty:endy]
        ROI_start_pixel = [startx, starty]
        ROI_end_pixel = [endx, endy]

        ROI_nb_pixels = ROI_data_temp.shape

        ROI_nb_pixels_list.append(ROI_nb_pixels)
        ROI_start_pixel_list.append(ROI_start_pixel)
        ROI_end_pixel_list.append(ROI_end_pixel)
        ROI_Original_ratio_list.append("20%")

        ROI_data.append(ROI_data_temp)

    # dict enclosing info about the ROI
    ROI_info["ROI_nb_pixels"] = ROI_nb_pixels_list
    ROI_info["ROI_start_pixel"] = ROI_start_pixel_list
    ROI_info["ROI_end_pixel"] = ROI_end_pixel_list
    ROI_info["ROI_Original_ratio"] = ROI_Original_ratio_list
    ROI_info = pd.DataFrame(ROI_info)

    return ROI_info, ROI_data


"""
# ex1: one image in tiff file
img = cm.get_images_from_multi_tiff(path_homo)
roi_df, roi_array = get_roi_default(img)

# ex1: 2 images in tiff file
img = cm.get_images_from_multi_tiff(path_cv)
roi_df, roi_array = get_roi_default(img)
"""


# 2. Compute CV


def get_segmented_image(img):
    """
    Given a 2D np.array, it replaces all the pixels with an intensity below
    a threshold otsu value by 0 as well as artifacts connected to image border.

    Parameters
    ----------
    img : np.array
        Original image in a 2D format.

    Returns
    -------
    img : np.array
        2D np.array where only pixels with significant intensity are given
        non null values.

    """
    # apply threshold
    thresh = threshold_otsu(img)
    # boolean matrice: True represent the pixels of interest
    bw = closing(img > thresh, square(3))
    # remove artifacts connected to image border
    cleared = clear_border(bw)

    # get segmented image
    xtot, ytot = np.shape(img)

    for i in range(xtot):
        for j in range(ytot):
            if not cleared[i, j]:
                img[i, j] = 0
    return img


"""
# ex:
img = cm.get_images_from_multi_tiff(path_cv)[0]
get_segmented_image(img)
"""


def get_non_zero_vec_from_seg_image(img):
    """
    Given a segmented image, it transforms the matrix to a non null vector.

    Parameters
    ----------
    img : np.array
        a segmented image in a 2D np.array format

    Returns
    -------
    non_zero_vec : np.array
        One dimension np.array of all pixels with postive intensity

    """
    non_zero_vec = img[img != 0]
    return non_zero_vec


"""
# ex:
img = cm.get_images_from_multi_tiff(path_cv)[0]
get_non_zero_vec_from_seg_image(img)
"""


def get_cv_table_global(tiff_data):
    """
    For each np.arrays of the given list, it computes the Coefficient of
    Variation (CV) of the central 20% (ROI).

    Parameters
    ----------
    tiff_data : list
        List of np.arrays

    Returns
    -------
    CV_df : pd.Data.Frame
        Dataframe enclosing info about the pixels with significant intensities
        of the segemented ROI of each given np.array:
            1. standard deviation
            2. mean
            3. number of pixels
            4. Coefficient of Variation (CV)
            5. Normalized CV: CV relative to min value.

    """
    std_intensity_list = []
    mean_intensity_list = []
    nb_pixels_list = []
    cv_list = []

    for i in range(len(tiff_data)):
        img_temp = get_segmented_image(tiff_data[i])
        ball_intensity_vec_temp = get_non_zero_vec_from_seg_image(img_temp)
        # Statistics
        std_intensity_temp = np.std(ball_intensity_vec_temp)
        mean_intensity_temp = np.mean(ball_intensity_vec_temp)
        nb_pixels_temp = len(ball_intensity_vec_temp)
        cv_temp = std_intensity_temp/mean_intensity_temp

        std_intensity_list.append(std_intensity_temp)
        mean_intensity_list.append(mean_intensity_temp)
        nb_pixels_list.append(nb_pixels_temp)
        cv_list.append(cv_temp)

    CV_normalized = np.divide(cv_list, min(cv_list))

    CV_dict = {"Standard deviation": std_intensity_list,
               "Average": mean_intensity_list,
               "Nb pixels": nb_pixels_list,
               "CV": cv_list,
               "CVs relative to min value": CV_normalized
               }

    CV_df = pd.DataFrame(CV_dict)

    return CV_df


"""
# ex1: one image in tiff file
img = cm.get_images_from_multi_tiff(path_cv)
get_cv_table_global(img)

# ex2: 2 images in tiff file
img = cm.get_images_from_multi_tiff(path_cv)
get_cv_table_global(img)
"""


# 3. Report: Get Tiff images with ROIs marked on them.


def get_original_with_marked_roi(tiff_data):
    """
    Add a rectangle on the original image which representing the ROI, i.e.
    the central 20%.

    Parameters
    ----------
    tiff_data : list
        list of images in a 2d np.array format.

    Returns
    -------
    image_list : list
        list of images of type PIL.Image.Image.

    """

    image_list = []
    roi_info, roi_arrays = get_roi_default(tiff_data)

    for i in range(len(tiff_data)):
        x0, y0 = roi_info["ROI_start_pixel"][i]
        xf, yf = roi_info["ROI_end_pixel"][i]

        roi_temp = tiff_data[i]

        rr, cc = polygon_perimeter([x0, x0, xf, xf],
                                   [y0, yf, yf, y0],
                                   shape=roi_temp.shape,
                                   clip=True)
        roi_temp[rr, cc] = 255
        image_list.append(Image.fromarray(roi_temp, mode="L"))

    return image_list


"""
# ex1: one image in tiff file
img = cm.get_images_from_multi_tiff(path_cv)
get_original_with_marked_roi(img)[0]

# ex2: 2 images in tiff file
img = cm.get_images_from_multi_tiff(path_cv)
get_original_with_marked_roi(img)[0]
get_original_with_marked_roi(img)[1]
"""


# 4. Get histogram : nb of pixels per intensity values


def get_hist_data(img):
    """
    Construct a non zero vector from the segmented ROI of a given image array
    and return a dataframe organizing the number of pixels per intensity value.

    Parameters
    ----------
    img : np.array
        Original 2D image.

    Returns
    -------
    count_df : pd.Data.Frame
        Dataframe with two columns:
            1. intensity values:
            2. nb of pixels

    """

    # convert matrix to one vector
    ball_intensity_vec = get_segmented_image(img)
    ball_intensity_vec.flatten()
    ball_intensity_vec = ball_intensity_vec[ball_intensity_vec != 0]
    np.ndarray.sort(ball_intensity_vec)

    # build a dataframe
    intensity_value, nb_pixel = np.unique(ball_intensity_vec,
                                          return_counts=True)

    return intensity_value, nb_pixel


"""
# ex:
img = cm.get_images_from_multi_tiff(path_cv)
roi_df, roi_arrays = get_roi_default(img)
get_hist_data(roi_arrays[0])
"""


def get_hist_nbpixel_vs_grayintensity(tiff_data):
    """
    For a given list of images in np.array format, return a histogram
    of the number of pixels per gray intensity of each of the given arrays.

    Parameters
    ----------
    tiff_data : list
        List of np.arrays.

    Returns
    -------
    fig : plot
        Histogram of the number of pixels per gray intensity.
    """

    fig = plt.figure()
    colors = ["r", "g", "b", "c", "m", "y", "k", "w"]

    roi_arrays = get_roi_default(tiff_data)[1]
    for i in range(len(roi_arrays)):
        hist_x, hist_y = get_hist_data(roi_arrays[i])
        plt.plot(
            hist_x, hist_y, marker=".", markersize=0.2, color=colors[i],
            label="ROI " + str(i), linewidth=0.8, figure=fig
            )

    plt.title("Intensity histogram", figure=fig)
    plt.xlim((0, 256))
    plt.xlabel("Gray levels")
    plt.ylabel("Nb Pixels")
    plt.legend()
    plt.title("Intensity histogram", figure=fig)

    return fig


"""
# ex:
img = cm.get_images_from_multi_tiff(path_cv)
get_hist_nbpixel_vs_grayintensity(img)
"""

"""
Generate CV report Given 1 .tif file enclosing one or more 2D images.
"""


def get_cv_report_elements(
        tiff_data, Microscope_type, Wavelength, NA, Sampling_rate, Pinhole
        ):
    """
    Generate the different componenent of the CV report and stock them in
    a list.

    Parameters
    ----------
    tiff_data : list
        list of images in a 2D np.arrays format.
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
    CVReportComponents : list
        List of all the CV report components:
            1. original images with ROIs marked on them.
            2. microscopy info dataframe
            3. histogram of the number of pixels per gray intensity value
            for all the images
            4. Dataframe enclosing info about the pixels with significant
            intensities of the segemented ROI of each given np.array.
    """

    roi_data = tiff_data

    # Get Histogram : Nbpixel VS Gray scale
    HistNbPixelVSGrayScale = get_hist_nbpixel_vs_grayintensity(roi_data)

    # Get Images with Marked ROIs on them
    img_original_marked_roi = get_original_with_marked_roi(roi_data)

    # Get Microscope info dataframe
    MicroscopyInfo = cm.get_microscopy_info(
        Microscope_type, Wavelength, NA, Sampling_rate, Pinhole
        )

    # Get CV table
    CV = get_cv_table_global(roi_data)
    CVReportComponents = [img_original_marked_roi,
                          MicroscopyInfo,
                          HistNbPixelVSGrayScale,
                          CV]

    return CVReportComponents


"""
# ex1:
img = cm.get_images_from_multi_tiff(path_cv)
CVreport_elements_1 = get_cv_report_elements(img, "Confocal", 460,
                                             1.4, "1.0x1.0x1.0", 1)
CVreport_elements_1[0]
CVreport_elements_1[1]
CVreport_elements_1[2]
CVreport_elements_1[3]
"""


def save_cv_report_elements(
        tiff_path, output_dir, Microscope_type, Wavelength,
        NA, Sampling_rate, Pinhole
        ):
    """
    Save the different elements of the CV componenent in a chosen directory.

    Parameters
    ----------
    tiff_path : str
        .tif file path. .tif file must contain one or more 2D images.
    output_dir : str
        Output directory path.
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
    1.Save as .png: original images with ROIs marked on them.
    2.Save as .csv: microscopy info dataframe.
    3.Save as .png: histogram of the number of pixels per gray intensity value
    for all the images.
    4.Save as .csv: Dataframe enclosing info about the pixels with significant
    intensities of the segemented ROI of each given np.array.

    """

    tiff_data = cm.get_images_from_multi_tiff(tiff_path)

    CVReportElements_temp = get_cv_report_elements(
        tiff_data, Microscope_type, Wavelength, NA, Sampling_rate, Pinhole
        )

    OriginalWithMarkedROIs = CVReportElements_temp[0]
    MicroscopyInfo = CVReportElements_temp[1]
    HistNbPixelVSGrayScale = CVReportElements_temp[2]
    CV_df = CVReportElements_temp[3]

    if len(OriginalWithMarkedROIs) > 1:
        for i in range(len(OriginalWithMarkedROIs)):
            OriginalWithMarkedROIs[i].save(
                fp=output_dir+str(i)+".ROI.png", format="PNG"
                )
    else:
        OriginalWithMarkedROIs[0].save(
            fp=output_dir+"ROI.png", format="PNG")

    MicroscopyInfo.to_csv(output_dir+"MicroscopyInfo.csv")
    HistNbPixelVSGrayScale.savefig(output_dir+"Hist.png",
                                   format="png",
                                   bbox_inches='tight')
    CV_df.to_csv(output_dir+"CV.csv")


"""
# ex1
save_cv_report_elements(path_cv, output_path_dir,
                        "Confocal", 460, 1.4,
                        "1.0x1.0x1.0", 1)
"""
