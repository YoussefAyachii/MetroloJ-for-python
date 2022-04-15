#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Thu Feb 17 13:49:45 2022
@author: Youssef

This module aims to read and extract metadata from -almost- any given image
format using bioformats package for python.
To extract the needed metadata, follow the bioformat scehema documentation (
https://www.openmicroscopy.org/Schemas/Documentation/ \
    Generated/OME-2016-06/ome.html)

Note that python-bioformats package uses python-javabridge to start and
interact with a Java virtual machine.

Mandatory commands: (from https://pythonhosted.org/python-bioformats/)
    pip install javabridge
    pip install python-bioformats

Remark: Code tested only on google colab
"""


import javabridge
import bioformats
javabridge.start_vm(class_path=bioformats.JARS)


def get_tiff_omexml(tiff_path):
    """
    This function returns the metadata of the tiff file.
    The metadata is stored in a bioformats specific format.
    Information from the metadata will be extracted then -from the returned
    variable- following bioformats schema documentation.

    Parameters
    ----------
    tiff_path : str
        path of the tiff file.

    Returns
    -------
    tiff_omexml : bioformats.omexml.OMEXML
        metadata of the chosen tiff file.

    """
    tiff_xml = bioformats.get_omexml_metadata(path=tiff_path)
    tiff_omexml = bioformats.OMEXML(tiff_xml)
    return tiff_omexml
