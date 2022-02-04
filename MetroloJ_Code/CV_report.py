#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 14:23:09 2022

@author: bottimacintosh
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import matplotlib.patches as mpatches

from skimage import data
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square

# pip install aicsimageio
# pip install 'Pillow==8.0.1' #aicsimageio need specific version of PIL
from PIL import Image
from aicsimageio import AICSImage, imread_dask
from aicsimageio import readers
#pip install 'aicsimageio[base-imageio]'
from collections import Counter

from skimage.filters import threshold_otsu
import math

import cv2

"""
Paths for execution
"""
path0="/Users/bottimacintosh/Downloads/pup3.tif"
path1="/Users/bottimacintosh/Documents/M2_CMB/IBDML/Data/CV/cv.comparatif.tif"
pathCV="/Users/bottimacintosh/Documents/M2_CMB/IBDML/Data/CV/cv.comparatif.tif"
path_dir="/Users/bottimacintosh/Desktop/test1.png"

"""
CV Report 
- In progress : Histograms 
- GetImagesFromeMultiTiff() , to replace by 
"""

# 1. Import .tiff containing the acquisitions made with the PMTs to analyze
# Use paths as inputs


def GetImagesFromeMultiTiff(path):
    img = Image.open(path)
    tiff_images = []
    for i in range(img.n_frames):
        img.seek(i)
        tiff_images.append(np.array(img))
    return tiff_images


# ex
img = GetImagesFromeMultiTiff(path1)[0]
GetImagesFromeMultiTiff(path1)[1]


# 2. Get ROI (default central 20% of the original image) for a given 2d image

def GetROIDefault(tiff_data):
    ROI_info = {}
    ROI_img = []

    ROI_nb_pixels_list = []
    ROI_start_pixel_list = []
    ROI_end_pixel_list = []
    ROI_Original_ratio_list = []
    
    if type(tiff_data)==list :
        
        for i in range(len(tiff_data)):
            
            x, y = tiff_data[i].shape
            h, w = x*0.4, y*0.4
        
            startx, endx = int(x//2 - h//2), int(x//2 + h//2)
            starty, endy = int(y//2 - w//2), int(y//2 + w//2)
        
            ROI_data = tiff_data[i][startx:endx, starty:endy]
            ROI_start_pixel = [startx, starty]
            ROI_end_pixel = [endx, endy]
        
            ROI_nb_pixels = ROI_data.shape
            
            ROI_nb_pixels_list.append(ROI_nb_pixels)
            ROI_start_pixel_list.append(ROI_start_pixel)
            ROI_end_pixel_list.append(ROI_end_pixel)
            ROI_Original_ratio_list.append("20%")
        
            ROI_img.append(Image.fromarray(ROI_data))
    else: 
        x, y = tiff_data.shape
        h, w = x*0.4, y*0.4
    
        startx, endx = int(x//2 - h//2), int(x//2 + h//2)
        starty, endy = int(y//2 - w//2), int(y//2 + w//2)
    
        ROI_data = tiff_data[startx:endx, starty:endy]
        ROI_start_pixel = [startx, starty]
        ROI_end_pixel = [endx, endy]
    
        ROI_nb_pixels = ROI_data.shape
        
        ROI_nb_pixels_list.append(ROI_nb_pixels)
        ROI_start_pixel_list.append(ROI_start_pixel)
        ROI_end_pixel_list.append(ROI_end_pixel)
        ROI_Original_ratio_list.append("20%")

        ROI_img=Image.fromarray(ROI_data)

    
    # 3 outputs :
    
    # a. dataframe
    ROI_info["ROI_nb_pixels"] = ROI_nb_pixels_list
    ROI_info["ROI_start_pixel"] = ROI_start_pixel_list
    ROI_info["ROI_end_pixel"] = ROI_end_pixel_list
    ROI_info["ROI_Original_ratio"] = ROI_Original_ratio_list
    ROI_info = pd.DataFrame(ROI_info)
    
    # b. ROI images: ROI_pil_list
    
    # c. ROI data
    

    return [ROI_info, ROI_img, tiff_data]

# ex1: one image in tiff file
img = GetImagesFromeMultiTiff(path1)[0]
GetROIDefault(img)[0]  # df
GetROIDefault(img)[1] # image
GetROIDefault(img)[2]  # matrix

# ex1: 2 images in tiff file
img = GetImagesFromeMultiTiff(path1)
GetROIDefault(img)[0]  # df
GetROIDefault(img)[1][0]
GetROIDefault(img)[1][1] # image
GetROIDefault(img)[2]  # matrices


# 3. Compute CV

def GetCVTableRegions(img): # to delete ? 
    thresh = threshold_otsu(img)
    bw = closing(img > thresh, square(3))
    
    # remove artifacts connected to image border
    cleared = clear_border(bw)
    
    # label image regions : Only billes pixels are !=0
    # return a Labeled array, where all connected regions are assigned the same integer value.
    label_image = label(cleared)
        
    region_dim=[]
    region_nb_pixels=[]
    region_start_pixel=[]
    region_end_pixel=[]
    region_center=[]
    region_max=[]
    region_min=[]
    mean_intensity=[]
    std_intensity=[]
    cv=[]
    
    for region_temp in regionprops(label_image, img):
 
        if region_temp.area >= 100:
            x0, y0, xf, yf = region_temp.bbox
            #outputs
            region_dim.append(str([xf-x0,yf-y0]))
            region_nb_pixels.append(region_temp.area)
            region_start_pixel.append(str([x0,y0]))
            region_end_pixel.append(str([xf,yf]))
            region_center .append(region_temp.centroid)
            region_max.append(region_temp.intensity_max)
            region_min.append(region_temp.intensity_min)
            
            
            
                
            region_matrix_to_vec=(region_temp.image_intensity).flatten()
            ball_intensity=np.delete(region_matrix_to_vec, np.where(region_matrix_to_vec==0))
            
            std_temp = np.std(ball_intensity)
            std_intensity.append(std_temp)

            mean_temp=np.mean(ball_intensity)
            mean_intensity.append(mean_temp)
            
            cv.append(std_temp/mean_temp)        
    
    cv_min=np.min(cv)
    cv_relative_to_min_value=np.divide(cv,cv_min)
    
    #Global roi values : merge the regions and get the resukts 
    
    
    cv_dict={"region_dim":region_dim,"region_nb_pixels":region_nb_pixels,
    "region_start_pixel":region_start_pixel,"region_end_pixel":region_end_pixel,
    "region_center":region_center, "mean_intensity":mean_intensity, "std_intensity":std_intensity,
    "CV":cv, "CVs_relative_to_min_value":cv_relative_to_min_value}                  ,
              
    cv_tab=pd.DataFrame(cv_dict)
    
    return cv_tab

#ex
img=GetImagesFromeMultiTiff(path1)[1]
img=GetROIDefault(img)[2]
pd.set_option("display.max_rows", None, "display.max_columns", None)


def GetSegmentedImage(img):
    thresh = threshold_otsu(img)
    bw = closing(img > thresh, square(3))
        
        # remove artifacts connected to image border
    cleared = clear_border(bw)
    xtot, ytot = np.shape(img)
    for i in range(xtot):
        for j in range(ytot):
            if cleared[i,j]==False :
                img[i,j]=0 
    return img

    
def GetNonZeroVecFromSegImage(img):
    return img[img != 0]

def GetCVTableGlobal(tiff_data):
    
    std_intensity_list = []
    mean_intensity_list = []
    nb_pixels_list = []
    cv_list = []
    
    if type(tiff_data)==list: 
        for i in range(len(tiff_data)):
            img_temp=GetSegmentedImage(tiff_data[i])
            ball_intensity_vec_temp=GetNonZeroVecFromSegImage(img_temp)
            # Statistics
            std_intensity_temp = np.std(ball_intensity_vec_temp)
            mean_intensity_temp = np.mean(ball_intensity_vec_temp)
            nb_pixels_temp = len(ball_intensity_vec_temp)
            cv_temp = std_intensity_temp/mean_intensity_temp
            
            std_intensity_list.append(std_intensity_temp)
            mean_intensity_list.append(mean_intensity_temp)
            nb_pixels_list.append(nb_pixels_temp)
            cv_list.append(cv_temp)
    
    else: 
        img_temp=GetSegmentedImage(tiff_data)
        ball_intensity_vec_temp=GetNonZeroVecFromSegImage(img_temp)
        # Statistics
        std_intensity_temp = np.std(ball_intensity_vec_temp)
        mean_intensity_temp = np.mean(ball_intensity_vec_temp)
        nb_pixels_temp = len(ball_intensity_vec_temp)
        cv_temp = std_intensity_temp/mean_intensity_temp
        
        std_intensity_list.append(std_intensity_temp)
        mean_intensity_list.append(mean_intensity_temp)
        nb_pixels_list.append(nb_pixels_temp)
        cv_list.append(cv_temp)
    
    
    CV_normalized=np.divide(cv_list,min(cv_list))
    
    
    CV_dict={"Standard deviation":std_intensity_list,
            "Average":mean_intensity_list,
            "Nb pixels": nb_pixels_list,
            "CV":cv_list ,
            "CVs relative to min value":CV_normalized
            }
    CV_df=pd.DataFrame(CV_dict)

    return CV_df 

# ex
img=GetImagesFromeMultiTiff(path1)
img=GetROIDefault(img)[2]

GetSegmentedImage(img)
GetNonZeroVecFromSegImage(img)
GetCVTableGlobal(img)

#4. Report : Get Tiff images with ROIs marked on them. 

def GetOriginalWithMarkedROI(tiff_data):
    image_list = []
    if type(tiff_data)==list :
        roi = GetROIDefault(tiff_data)
        roi_info = roi[0]
        roi_data = roi [2]
        for i in range(len(tiff_data)):
            x0,y0 = roi_info["ROI_start_pixel"][i]
            xf,yf = roi_info["ROI_end_pixel"][i]
            
            roi_temp=roi_data[i]
            cv2.rectangle(roi_temp, (x0, y0), (xf, yf), (255, 0, 0), 1)
            image_list.append(Image.fromarray(roi_temp, mode="L"))
    
    else:
        roi = GetROIDefault(tiff_data)
        roi_info = roi[0]
        roi_data = roi [2]

        x0,y0 = roi_info["ROI_start_pixel"][0]
        xf,yf = roi_info["ROI_end_pixel"][0]
    
        cv2.rectangle(roi_data, (x0, y0), (xf, yf), (255, 0, 0), 1)
        image_list.append(Image.fromarray(roi_data, mode="L"))
    
    return image_list

# Ex 
img=GetImagesFromeMultiTiff(path1)[1]
GetOriginalWithMarkedROI(img)[0]

# Ex 2 
img=GetImagesFromeMultiTiff(path1)
GetOriginalWithMarkedROI(img)[0]
GetOriginalWithMarkedROI(img)[1]

#5. Report : Microscopie info 

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
GetMicroscopyInfo("Confocal",460.0,1.4,"1.0x1.0x1.0", 1.0)


# 6. get histogram : nb of pixels per intensity values

def GetHistData(img):
    
    roi = GetROIDefault(img)

    roi_data = roi[2]

    roi_info = roi[0]
    x0,y0 = roi_info["ROI_start_pixel"][0]
    xf,yf = roi_info["ROI_end_pixel"][0]
    h,w = (xf-x0),(yf-y0)
    
    # convert matrix to one vector
    img=GetSegmentedImage(img)
    ball_intensity_vec=GetNonZeroVecFromSegImage(img)
    
    # build a dataframe 
    count_df = pd.DataFrame.from_dict(Counter(ball_intensity_vec), orient='index').reset_index()
    count_df = count_df.rename(columns={'index':'intensity', 0:'count'})
    count_df = count_df.sort_values(by="intensity", axis=0, ascending=True)

    return count_df

def GetHistNbPixelVSGrayScale(tiff_data): 
    fig = plt.figure()
    colors=["r","g","b","y"]
    if type(tiff_data)==list : #if tiff comprises more than 1 array
        for i in range(len(tiff_data)):
            count_df=GetHistData(tiff_data[i])
            plt.plot(count_df['intensity'], count_df["count"] ,marker=".", markersize=0.2,  color=colors[i], label= "ROI "+str(i), linewidth=0.8,  figure=fig)
    else:
        count_df=GetHistData(tiff_data)
        fig = plt.figure()
        plt.plot(count_df['intensity'], count_df["count"] ,marker=".", markersize=0.2,  color="b", label= "ROI", linewidth=0.8,  figure=fig)
      
    
    plt.title("Intensity histogram", figure=fig)
    plt.xlim((0,256)) 
    plt.xlabel("Gray levels")
    plt.ylabel("Nb Pixels")
    plt.legend()
    plt.title("Intensity histogram", figure=fig)
    plt.show() #to pythn verification
    
    return fig

# ex 
img=GetImagesFromeMultiTiff(path1)[0]
GetHistData(img)

tiff_data=GetImagesFromeMultiTiff(path1)
GetHistNbPixelVSGrayScale(tiff_data)

"""
REPORT
"""
# below only apply on 1 image tiff -> to generalize on multi tiff file


def GetCVReportElements(tiff_data,
                        Microscope_type,
                        Wavelength,
                        NA,
                        Sampling_rate,
                        Pinhole):
        
    

    roi_data = GetROIDefault(tiff_data)[2]
   
    # Get Histogram : Nbpixel VS Gray scale
    HistNbPixelVSGrayScale = GetHistNbPixelVSGrayScale(roi_data)

    # Get Images with Marked ROIs on them
    img_original_marked_roi = GetOriginalWithMarkedROI(roi_data)

    # Get Microscope info dataframe
    MicroscopyInfo = GetMicroscopyInfo(Microscope_type,
                                   Wavelength,
                                   NA,
                                   Sampling_rate,
                                   Pinhole)



    # Get CV table
    CV = GetCVTableGlobal(roi_data)
    CVReportComponents = [img_original_marked_roi,
                          MicroscopyInfo,
                          HistNbPixelVSGrayScale,
                          CV]

    return CVReportComponents

#ex1
img=GetImagesFromeMultiTiff(path1)
CVreport_elements_1=GetCVReportElements(img, "Confocal", 460, 1.4, "1.0x1.0x1.0", 1)
CVreport_elements_1[0]
CVreport_elements_1[1]
CVreport_elements_1[2]
CVreport_elements_1[3]

#add GetCVReportFromPath after resolving the multiple tiff issue

def SaveCVReportElements(tiff_path,
                         output_dir,
                         Microscope_type, Wavelength,
                         NA, Sampling_rate, Pinhole):

    tiff_data=GetImagesFromeMultiTiff(tiff_path)
    

    CVReportElements_temp = GetCVReportElements(tiff_data,
                                           Microscope_type,
                                           Wavelength,
                                           NA, Sampling_rate, Pinhole)
            
    OriginalWithMarkedROIs=CVReportElements_temp[0]
    MicroscopyInfo=CVReportElements_temp[1]
    HistNbPixelVSGrayScale=CVReportElements_temp[2]
    CV_df=CVReportElements_temp[3]

    
    if len(OriginalWithMarkedROIs)>1:
        for i in range(len(OriginalWithMarkedROIs)):
            OriginalWithMarkedROIs[i].save(fp=output_dir+str(i)+".ROI.png", format="PNG")
    else: 
        OriginalWithMarkedROIs[0].save(fp=output_dir+"ROI.png", format="PNG")
    
    MicroscopyInfo.to_csv(output_dir+"MicroscopyInfo.csv")
    HistNbPixelVSGrayScale.savefig(output_dir+"Hist.png", format="png", bbox_inches='tight')
    CV_df.to_csv(output_dir+"CV.csv")

    # Image independent outputs

#ex1
output_path_dir="/Users/bottimacintosh/Documents/M2_CMB/IBDML/MetroloJ-for-python/CV_output_files/"
SaveCVReportElements(path1, output_path_dir, "Confocal", 460, 1.4, "1.0x1.0x1.0", 1 )    
    







        
    