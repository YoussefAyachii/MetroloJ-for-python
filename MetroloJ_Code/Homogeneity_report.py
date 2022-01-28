#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 15:32:09 2022

@author: bottimacintosh
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#pip install aicsimageio
#pip install 'Pillow==8.0.1' #aicsimageio need specific version of PIL
from PIL import Image
from skimage import draw
#pip install 'aicsimageio[base-imageio]'

"""
Paths for execution
"""
path5="/Users/bottimacintosh/Documents/M2_CMB/IBDML/Data/homogeneity/lame/homogeneite10zoom1-488.tif"


"""
Homogeneity report 
Code tested on .tif file from homogeneity samples

Remarks: 
- Centers' locations table (report): in progress
"""

#1. Report: Normalized Intensity Report
#Get normalized intensity matrix (in %) : divide all the pixels' intensity by the max_intensity
def GetNormIntensityMatrix(path): #used in both N.I.Profile and Centers' Locations
    img = np.asanyarray(Image.open(path))
    
    max_intensity=np.max(img)
    norm_intensity_profile=np.round(img/max_intensity * 100) #the rule of three : max_intensity->100%, pixel_intensity*100/max
    return norm_intensity_profile

#Get normalized intensity profile (plot)
def GetNormIntensityProfile(path):
    norm_intensity_profile=GetNormIntensityMatrix(path)

    plt.imshow(norm_intensity_profile)
    plt.colorbar ( )
    plt.title("Normalized intensity profile")

GetNormIntensityMatrix(path5)
GetNormIntensityProfile(path5)

#2. Report : Microscopy Info (copied from CV_report)
def GetMicroscopyInfo(Microscope_type, Wavelength, NA, Sampling_rate, Pinhole):
    info_values=[Microscope_type, Wavelength, NA, Sampling_rate, Pinhole]
    info_labels=["Microscope type", "Wavelength", "NA", "Sampling rate", "Pinhole"]

    #add units to info_labels
    info_units=["","(nm)","","pixel","(airy units)"]
    info_labels=[i+" "+info_units[info_labels.index(i)] for i in info_labels]
    
    MicroscopyInfo_dict={}
    MicroscopyInfo_dict["Labels"]=info_labels
    MicroscopyInfo_dict["Values"]=info_values
    
    MicroscopyInfo=pd.DataFrame(MicroscopyInfo_dict)
    return MicroscopyInfo

GetMicroscopyInfo("example",1,2,3,4)

#3. Report: Centers' locations IN PROGRESS


#4. Report: Intensity Profiles 

def GetPixelValuesOfLine(image_2d, x0, y0, xf, yf):
    rr, cc= np.array(draw.line(x0, y0, xf, yf))
    line_pixel_values=[image_2d[rr[i],cc[i]] for i in range(len(rr))]
    return line_pixel_values

def GetIntensityPlot(image_2d):
    
    xmax, ymax=np.shape(image_2d)
    xmax=xmax-1 #-1 : python starts from 0
    ymax=ymax-1
    xmid=round(xmax/2) 
    ymid=round(ymax/2)
    V_seg=GetPixelValuesOfLine(image_2d, x0=0, y0=ymid, xf=xmax, yf=ymid)
    H_seg=GetPixelValuesOfLine(image_2d, x0=xmid, y0=0, xf=xmid, yf=ymax)
    diagUD=GetPixelValuesOfLine(image_2d, x0=0, y0=0, xf=xmax, yf=ymax) #diag UpDown Left Right
    diagDU=GetPixelValuesOfLine(image_2d, x0=xmax, y0=0, xf=0, yf=ymax) #diag DownUp Left Right
    
    #x.axes
    def GetXAxis(line):
        nb_pixels=len(line)
        x_axis=list(np.arange(round(-nb_pixels/2), round(nb_pixels/2+1), 1))
        x_axis.remove(0) #the center of the matrix is 4 pixels not one 
        return x_axis
    
    #plot
    plt.plot(GetXAxis(V_seg), V_seg, color = 'b',label = 'V_seg')
    plt.plot(GetXAxis(H_seg), H_seg, color = 'g',label = 'H_seg')

    plt.plot(GetXAxis(diagUD), diagUD, color='r', label= 'Diag1')
    plt.plot(GetXAxis(diagDU), diagDU, color='y', label= 'Diag2')

    plt.axvline(0, linestyle='--')  
    plt.title("Intensity Profiles ")
    plt.xlim((min(GetXAxis(diagUD))-25,max(GetXAxis(diagUD))+25)) #25 subjective choice
    plt.legend()

GetIntensityPlot(GetImagesFromeMultiTiff(path5)[0]) #GetImagesFromeMultiTiff() from CV_report

#5. Report: Profile statistics

def GetProfileStatisticsTable(path):
    img=np.asanyarray(Image.open(path))
    
    #1. find the maximum intensity and the corresponding pixel.
    max_intensity=np.max(img)
    nb_pixels_with_max_intensity=len(np.where(img==max_intensity)[0]) #optional
    #we chose only the first localization if the max intensity is in >1 pixels
    x_index_max_intensity=np.where(img==max_intensity)[0][0]
    y_index_max_intensity=np.where(img==max_intensity)[1][0]
    max_found_at=[x_index_max_intensity, y_index_max_intensity]
    relative_to_max=1.0
    
    #2. Top-left corner intensity and its ratio over max_intensity
    TL_x_index, TL_y_index = [0,0]
    TL_intensity=img[TL_x_index, TL_y_index]
    TL_relative=TL_intensity/max_intensity
    
    #3. Top-right corner intensity and its ratio over max_intensity
    TR_x_index, TR_y_index = [0, np.shape(img)[1]-1]
    TR_intensity=img[TR_x_index, TR_y_index]
    TR_relative=TR_intensity/max_intensity
    
    #4. Bottom-left corner intensity and its ratio over max_intensity
    BL_x_index, BL_y_index = [np.shape(img)[0]-1, 0]
    BL_intensity=img[BL_x_index, BL_y_index]
    BL_relative=BL_intensity/max_intensity
    
    #5. Bottom-right corner intensity and its ratio over max_intensity
    BR_x_index, BR_y_index = [np.shape(img)[0]-1, np.shape(img)[1]-1]
    BR_intensity=img[BR_x_index, BR_y_index]
    BR_relative=BR_intensity/max_intensity
    
    #6. Upper - Middle pixel 
    UM_x_index, UM_y_index = [0, round(np.shape(img)[1]/2)]
    UM_intensity=img[UM_x_index, UM_y_index]
    UM_relative=UM_intensity/max_intensity
    
    #7. Bottom - Middle pixel 
    BM_x_index, BM_y_index = [np.shape(img)[0]-1, round(np.shape(img)[1]/2)]
    BM_intensity=img[BM_x_index, BM_y_index]
    BM_relative=BM_intensity/max_intensity
    
    #8. Left - Middle pixel
    LM_x_index, LM_y_index = [round(np.shape(img)[0]/2), 0]
    LM_intensity=img[LM_x_index, LM_y_index]
    LM_relative=LM_intensity/max_intensity
    
    #9. Right Middle pixel 
    RM_x_index, RM_y_index = [round(np.shape(img)[0]/2), np.shape(img)[1]-1]
    RM_intensity=img[RM_x_index, RM_y_index]
    RM_relative=RM_intensity/max_intensity
    
    
    #build dictionnary 
    ProfilesStatistics_dict={}
    ProfilesStatistics_dict["Location"]=["Maximum found at "+str(max_found_at),
                    "Top-Left corner",
                    "Top-Right corner",
                    "Bottom-Left corner",
                    "Bottom-Right corner",
                    "Upper-Middle pixel",
                    "Bottom-Middle pixel",
                    "Left-Middle pixel",
                    "Right-Middle pixel"]
    ProfilesStatistics_dict["Intensity"]=[max_intensity,
                                     TL_intensity,
                                     TR_intensity,
                                     BL_intensity,
                                     BR_intensity,
                                     UM_intensity,
                                     BM_intensity,
                                     LM_intensity,
                                     RM_intensity]
    ProfilesStatistics_dict["Intensity Relative to Max"]=[relative_to_max,
                                                     TL_relative,
                                                     TR_relative,
                                                     BL_relative,
                                                     BR_relative,
                                                     UM_relative,
                                                     BM_relative,
                                                     LM_relative,
                                                     RM_relative]
                                     
    ProfilesStatistics=pd.DataFrame(ProfilesStatistics_dict)
    return ProfilesStatistics
    
GetProfileStatisticsTable(path5)

