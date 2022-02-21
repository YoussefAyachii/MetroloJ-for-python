#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 09:45:57 2022
@author: Youssef

This file comprises functions that are not finished or whose goal is unclear.
These functions may be included later in one of the main modules.
"""


from PIL import Image
from PIL.TiffTags import TAGS


def get_metadata(path, info_key="ImageDescription"):
    """
    This function aim to extract the sample name of the image from a given .tif
    file.

    Ambiguity 1: the available examples do not provide sample names.
    Ambiguity 2: chose the right info_key that corresponds to the sample name.
    Ambiguity 3: does one single .tif file provide sample names of each image?

    Parameters
    ----------
    path : TYPE
        DESCRIPTION.
    info_key : TYPE, optional
        DESCRIPTION. The default is "ImageDescription".

    Returns
    -------
    None.

    """
    img = Image.open(path)
    meta_dict = {TAGS[key]: img.tag[key] for key in img.tag_v2}
    return meta_dict[info_key]


"""
path_cv="/Users/Youssef/Documents/IBDML/Data/CV/cv.comparatif.tif"
get_metadata(path_cv)
"""
