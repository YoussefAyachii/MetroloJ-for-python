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

    fig=plt.figure()
    plt.imshow(norm_intensity_profile)
    plt.colorbar ( )
    plt.title("Normalized intensity profile", figure=fig)

    return fig

#ex
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

#ex
GetMicroscopyInfo("example",1,2,3,4)

#3. Report: Centers' locations IN PROGRESS


#4. Report: Intensity Profiles

def GetPixelValuesOfLine(path, x0, y0, xf, yf):
    image_2d=np.array(Image.open(path))
    rr, cc= np.array(draw.line(x0, y0, xf, yf))
    line_pixel_values=[image_2d[rr[i],cc[i]] for i in range(len(rr))]
    return line_pixel_values

def GetXAxis(line):
        nb_pixels=len(line)
        x_axis=list(np.arange(round(-nb_pixels/2), round(nb_pixels/2+1), 1))
        x_axis.remove(0) #the center of the matrix is 4 pixels not one
        return x_axis


def GetIntensityPlot(path):
    image_2d=np.array(Image.open(path))

    xmax, ymax=np.shape(image_2d)
    xmax=xmax-1 #-1 : python starts from 0
    ymax=ymax-1
    xmid=round(xmax/2)
    ymid=round(ymax/2)
    V_seg=GetPixelValuesOfLine(path, x0=0, y0=ymid, xf=xmax, yf=ymid)
    H_seg=GetPixelValuesOfLine(path, x0=xmid, y0=0, xf=xmid, yf=ymax)
    diagUD=GetPixelValuesOfLine(path, x0=0, y0=0, xf=xmax, yf=ymax) #diag UpDown Left Right
    diagDU=GetPixelValuesOfLine(path, x0=xmax, y0=0, xf=0, yf=ymax) #diag DownUp Left Right


    #plot
    fig = plt.figure()
    plt.plot(GetXAxis(V_seg), V_seg, color = 'b',label = 'V_seg', figure=fig)
    plt.plot(GetXAxis(H_seg), H_seg, color = 'g',label = 'H_seg', figure=fig)

    plt.plot(GetXAxis(diagUD), diagUD, color='r', label= 'Diag1', figure=fig)
    plt.plot(GetXAxis(diagDU), diagDU, color='y', label= 'Diag2', figure=fig)

    plt.axvline(0, linestyle='--')
    plt.title("Intensity Profiles", figure=fig)
    plt.xlim((min(GetXAxis(diagUD))-25,max(GetXAxis(diagUD))+25)) #25 subjective choice
    plt.legend()

    return fig

#ex
GetIntensityPlot(path5)

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

#ex
GetProfileStatisticsTable(path5)

"""
#REPORT
"""

def GetHomogeneityReportElements(path, Microscope_type, Wavelength, NA, Sampling_rate, Pinhole):

    #1. Get Normalized Intensity Profile
    NormIntensityProfile=GetNormIntensityProfile(path)

    #2. Get Microscopy Info
    MicroscopyInfo=GetMicroscopyInfo(Microscope_type, Wavelength, NA, Sampling_rate, Pinhole)

    #3. Get Centers' locations

    #4. Get Intensity Profiles
    IntensityPlot=GetIntensityPlot(path) #2nd fct from CV file

    #5. Get Profiles Statistics
    ProfileStatisticsTable=GetProfileStatisticsTable(path)

    HomogeneityReportComponents=[NormIntensityProfile, MicroscopyInfo, IntensityPlot, ProfileStatisticsTable]

    return HomogeneityReportComponents


#ex1
Homogeneity_report_elements_1=GetHomogeneityReportElements(path5, "Confocal", 460, 1.4, "1.0x1.0x1.0", 1 )
Homogeneity_report_elements_1[0]
Homogeneity_report_elements_1[1]
Homogeneity_report_elements_1[2]
Homogeneity_report_elements_1[3]

def SaveHomogeneityReportElements(tiff_path, output_dir, Microscope_type, Wavelength, NA, Sampling_rate, Pinhole):
    HomogeneityReportElements = GetHomogeneityReportElements(tiff_path, Microscope_type, Wavelength, NA, Sampling_rate, Pinhole)

    NormIntensityProfile = HomogeneityReportElements[0]
    MicroscopyInfo = HomogeneityReportElements[1]
    IntensityPlot = HomogeneityReportElements [2]
    ProfileStatisticsTable = HomogeneityReportElements [3]

    #.png : Normalized Intensity Profile and Intesity Profile plot
    NormIntensityProfile.savefig(output_dir+"HistNbPixelVSGrayScale.png", format="png", bbox_inches='tight')
    IntensityPlot.savefig(output_dir+"IntensityPlot.png", format="png", bbox_inches="tight")

    #.csv : Microscopy Info and ProfileStatisticsTable
    MicroscopyInfo.to_csv(output_dir+"MicroscopyInfo.csv")
    ProfileStatisticsTable.to_csv(output_dir+"ProfileStatisticsTable.csv")

#ex1
output_dir_hom="/Users/bottimacintosh/Documents/M2_CMB/IBDML/MetroloJ-for-python/Homogeneity_output_files/"
SaveHomogeneityReportElements(path5, output_dir_hom, "Confocal", 460, 1.4, "1.0x1.0x1.0", 1 )


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
