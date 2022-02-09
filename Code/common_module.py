#!/usr/bin/env python3

"""

requirement modules

"""

import numpy as np
import pandas as pd
from PIL import Image


def get_images_from_multi_tiff(path):
    """
    Import .tif file from a given path

    Parameters
    ----------
    path : str
        .tif file path.

    Returns
    -------
    tiff_images : list
        retrns list of np.arrays enclosed in the .tif file.

    """
    img = Image.open(path)
    tiff_images = []
    for i in range(img.n_frames):
        img.seek(i)
        tiff_images.append(np.array(img))
    return tiff_images


def get_microscopy_info(
        Microscope_type, Wavelength, NA, Sampling_rate, Pinhole
        ):
    """
    Organize microscopy info provided by the user into a dataframe.

    Parameters
    ----------
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
    MicroscopyInfo : pd.DataFrame
        dataframe resuming microscopy informations.
        2 columns: Labels and Values

    """

    info_values = [Microscope_type, Wavelength, NA, Sampling_rate, Pinhole]
    info_labels = ["Microscope type",
                   "Wavelength (nm)",
                   "Nmerical Aperture",
                   "Sampling rate (pixel)",
                   "Pinhole (airy units)"]

    # organize result in a dataframe
    MicroscopyInfo_dict = {}
    MicroscopyInfo_dict["Labels"] = info_labels
    MicroscopyInfo_dict["Values"] = info_values

    MicroscopyInfo = pd.DataFrame(MicroscopyInfo_dict)

    return MicroscopyInfo


"""
ex:
get_microscopy_info("Confocal", 460.0, 1.4, "1.0x1.0x1.0", 1.0)
"""
