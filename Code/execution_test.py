#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 14:35:39 2022

@author: Youssef
"""

# import local modules
import common_module as cm
import cv_module as cv
import homo_module as homo


# save cv report elements
path_cv = "/Users/Youssef/Documents/IBDML/Data/CV/cv.comparatif.tif"
output_path_cv = "/Users/Youssef/Documents/IBDML/MetroloJ-for-python/CV_output_files/"

cv.save_cv_report_elements(
    tiff_path=path_cv,
    output_dir=output_path_cv,
    Microscope_type="confocal",
    Wavelength=458,
    NA=1,
    Sampling_rate="1x1x1",
    Pinhole=1)


# save homogeneiety report elements
path_homo = "/Users/Youssef/Documents/IBDML/Data/homogeneity/lame/homogeneite10zoom1-488.tif"
output_path_homo = "/Users/Youssef/Documents/IBDML/MetroloJ-for-python/Homogeneity_output_files/"

homo.save_homogeneity_report_elements(
    tiff_path=path_homo,
    output_dir=output_path_homo,
    Microscope_type="confocal",
    Wavelength=458,
    NA=1,
    Sampling_rate="1x1x1",
    Pinhole=1)













# paths for test
path_homo = "/Users/Youssef/Documents/IBDML/Data/homogeneity/lame/homogeneite10zoom1-488.tif"
output_dir_hom = "/Users/Youssef/Documents/IBDML/MetroloJ-for-python/Homogeneity_output_files/"

# metroloJ-for-python modules


cm.get_images_from_multi_tiff(path_cv)[0]



