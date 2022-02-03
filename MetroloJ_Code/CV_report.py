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
path1="/Users/Youssef/Documents/IBDML/Data/CV/cv.comparatif.tif"
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

def GetROIDefault(img):
    x, y = img.shape
    h, w = x*0.4, y*0.4

    startx, endx = int(x//2 - h//2), int(x//2 + h//2)
    starty, endy = int(y//2 - w//2), int(y//2 + w//2)

    ROI_data = img[startx:endx, starty:endy]
    ROI_start_pixel = [startx, starty]
    ROI_end_pixel = [endx, endy]

    ROI_nb_pixels = [x, y]

    # 2 outputs :

    # a. dataframe
    ROI_info = {}
    ROI_info["Original_image_dim"] = [x, y]
    ROI_info["ROI_nb_pixels"] = ROI_nb_pixels
    ROI_info["ROI_start_pixel"] = ROI_start_pixel
    ROI_info["ROI_end_pixel"] = ROI_end_pixel
    ROI_info["ROI_Original_ratio"] = ["25%", "-"]
    ROI_info = pd.DataFrame(ROI_info)

    # b. image: ROI_Pil
    ROI_Pil = Image.fromarray(ROI_data)

    return [ROI_info, ROI_Pil, ROI_data]

# ex
img = GetImagesFromeMultiTiff(path1)[0]
GetROIDefault(img)[0]  # df
GetROIDefault(img)[1]  # image
GetROIDefault(img)[2]  # matrix


# 3. Compute CV

def GetCVTable(img):
    thresh = threshold_otsu(img)
    bw = closing(img > thresh, square(3))
    
    # remove artifacts connected to image border
    cleared = clear_border(bw)
    Image.fromarray(cleared)
    
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
print(GetCVTable(img))

#4. Report : Get Tiff images with ROIs marked on them. 

def GetOriginalWithMarkedROI(img):
    
    img_info = GetROIDefault(img)[0]
    x0,y0 = img_info["ROI_start_pixel"]
    xf,yf = img_info["ROI_end_pixel"]

    cv2.rectangle(img, (x0, y0), (xf, yf), (255, 0, 0), 1)
    image_temp_toshow=Image.fromarray(img, mode="L")
    
    return image_temp_toshow

# Ex 
img=GetImagesFromeMultiTiff(path1)[0]
GetOriginalWithMarkedROI(img)

# Ex 2 
img=GetImagesFromeMultiTiff(path1)[1]
GetOriginalWithMarkedROI(img)


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

def GetHistNbPixelVSGrayScale(img):

    roi = GetROIDefault(img)

    roi_data = roi[2]

    roi_info = roi[0]
    x0,y0 = roi_info["ROI_start_pixel"]
    xf,yf = roi_info["ROI_end_pixel"]
    h,w = (xf-x0),(yf-y0)
    
    fig = plt.figure()
    roi_data_flat=list(roi_data.flatten())  # convert matrix to one vector
    count_df = pd.DataFrame.from_dict(Counter(roi_data_flat), orient='index').reset_index()
    count_df = count_df.rename(columns={'index':'intensity', 0:'count'})
    count_df = count_df.sort_values(by="intensity", axis=0, ascending=True)
    plt.plot(count_df['intensity'], count_df["count"] ,marker=".", markersize=0.2,  color="b", label= 'ROI pixels', linewidth=0.8,  figure=fig)


    plt.title("Intensity histogram", figure=fig)
    plt.xlim((0,256)) 
    plt.xlabel("Gray levels")
    plt.ylabel("Nb Pixels")
    plt.legend()
    plt.title("Intensity histogram", figure=fig)
    plt.show() #to pythn verification

    return fig

# ex 
img=GetImagesFromeMultiTiff(path1)[1]
GetHistNbPixelVSGrayScale(img)



"""
REPORT
"""
# below only apply on 1 image tiff -> to generalize on multi tiff file 
def GetCVReportElements (img, Microscope_type, Wavelength, NA, Sampling_rate, Pinhole):
    #Get image from path; 2d image from path; 
    original_data = img
    roi_data = GetROIDefault(original_data)[2]

    # Get Images with Marked ROIs on them
    img_original_marked_roi = GetOriginalWithMarkedROI(original_data)
    
    #Get Microscope info dataframe 
    MicroscopyInfo=GetMicroscopyInfo(Microscope_type, Wavelength, NA, Sampling_rate, Pinhole)
    
    #Get Histogram : Nbpixel VS Gray scale
    HistNbPixelVSGrayScale=GetHistNbPixelVSGrayScale(roi_data)
    
    #Get CV table
    CV=GetCVTable(roi_data)
    CVReportComponents=[img_original_marked_roi,MicroscopyInfo ,  HistNbPixelVSGrayScale, CV]
    
    return CVReportComponents

#ex1
img=GetImagesFromeMultiTiff(path1)[1]
CVreport_elements_1=GetCVReportElements(img, "Confocal", 460, 1.4, "1.0x1.0x1.0", 1)
CVreport_elements_1[0]
CVreport_elements_1[1]
CVreport_elements_1[2]
CVreport_elements_1[3]

#add GetCVReportFromPath after resolving the multiple tiff issue

def SaveCVReportElements(tiff_path, output_dir, x, y, h, w, Microscope_type, Wavelength, NA, Sampling_rate, Pinhole):
    CVReportElements=GetCVReportElements(tiff_path, x, y, h, w, Microscope_type, Wavelength, NA, Sampling_rate, Pinhole)
    
    OriginalWithMarkedROIs=CVReportElements[0]
    MicroscopyInfo=CVReportElements[1]
    HistNbPixelVSGrayScale=CVReportElements[2]
    CV_df=CVReportElements[3]
    
    
    
    #.png : Images with marked ROIs
    if len(OriginalWithMarkedROIs)>1 : 
        for i in range(len(OriginalWithMarkedROIs)):
            output_path_temp=output_dir+"Image_"+str(i)            
            OriginalWithMarkedROIs[i].save(fp=output_path_temp+".png", format="PNG")
    else:
        output_path=output_dir            
        OriginalWithMarkedROIs[i].save(fp=output_path+".png", format="PNG")
    
    CV_df.to_csv(output_dir+"CV.csv")
    MicroscopyInfo.to_csv(output_dir+"MicroscopyInfo.csv")
    
    HistNbPixelVSGrayScale.savefig(output_dir+"HistNbPixelVSGrayScale.png", format="png", bbox_inches='tight') 
    
#ex1
output_path_dir="/Users/bottimacintosh/Documents/M2_CMB/IBDML/MetroloJ-for-python/CV_output_files/"
SaveCVReportElements(path1, output_path_dir, 100, 100, 300, 300, "Confocal", 460, 1.4, "1.0x1.0x1.0", 1 )    
    







        
    