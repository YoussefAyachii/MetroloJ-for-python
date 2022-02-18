#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 13:49:45 2022
@author: Youssef

This module aims to read and extract metadata from -almost- any given image
format using bioformats package for python.
Note that python-bioformats package uses python-javabridge to start and
interact with a Java virtual machine.
https://pythonhosted.org/python-bioformats/
"""

""" Waiting until installing jdr locally
pip install javabridge
pip install python-bioformats

import numpy as np
import javabridge
import bioformats
javabridge.start_vm(class_path=bioformats.JARS)


path_coloc = "/Users/Youssef/Documents/IBDML/Data/homogeneity/lame/homogeneite10zoom06-405.lsm"
coloc_xml = bioformats.get_omexml_metadata(path=path_coloc)
coloc_omexml = bioformats.OMEXML(coloc_xml)
dir(coloc_omexml)
print(coloc_omexml.image().Name)
"""
