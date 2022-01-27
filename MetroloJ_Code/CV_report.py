#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 14:23:09 2022

@author: bottimacintosh
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.image as mi
import matplotlib
import cv2

#pip install aicsimageio
#pip install 'Pillow==8.0.1' #aicsimageio need specific version of PIL
from PIL import Image
from aicsimageio import AICSImage, imread_dask
from aicsimageio import readers
#pip install 'aicsimageio[base-imageio]'

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
"""

#1. Import a stack of images containing the acquisitions made with the PMTs to analyze
#Use paths as inputs


def GetImagesFromeMultiTiff(path):
    img = Image.open(path)
    tiff_images = []
    for i in range(img.n_frames):
        img.seek(i)
        tiff_images.append(np.array(img))
    return tiff_images

path1="/Users/bottimacintosh/Documents/M2_CMB/IBDML/Data/CV/cv.comparatif.tif"
tiff_images_data=GetImagesFromeMultiTiff(path1)

#2. Get same ROI for each region

def GetROI(image_data, x, y, h , w, origin="TL"):
    if origin=="TL": #top left
        ROI=image_data[y:y+w, x:x+h]
    if origin=="TR": #top right
        ROI=image_data[y:y-w, x:x+h]
    if origin=="BL": #bottom left
        ROI=image_data[y:y+w, x:x-h]     
    if origin=="BR": #bottom right
        ROI=image_data[y:y-w, x:x-h]     
    return ROI

def GetROIsFromMultipleTiff(path, x, y, h , w, origin="TL"):
    tiff_images_data=GetImagesFromeMultiTiff(path)
    
    ROIs_images_data=[]
    for i in range(len(tiff_images_data)):
        ROIs_images_data.append(GetROI(tiff_images_data[i], x,y,w,h,origin))
        
    return ROIs_images_data

ROIs_images_data=GetROIsFromMultipleTiff(path1, 100,100, 300, 300)

#3. Compute CV 

def GetColorInfo(path):
    #import the tiff as AISCI file, then use .dims["C"] to verify if its colored 
    tiff_images_aicsi=AICSImage(path, reader=readers.TiffReader)
    
    #0 B&W or 1 Colored
    color=0 if tiff_images_aicsi.dims["C"][0]==1 else 1

    return color

GetColorInfo(path0)
GetColorInfo(path1)

def GetCVforBW(path, sample_names="NA"):
    tiff_images_data=GetImagesFromeMultiTiff(path)
    
    avg_vec=[]
    sd_vec=[]
    nb_pixels_vec=[] # what is the number of pixels ?
    CV_vec=[]
    
    for i in range(len(tiff_images_data)):
     
        avg_intesnity=np.mean(tiff_images_data[i]) 
    
        sd_intensity=np.std(tiff_images_data[i])
    
        nb_pixels="?"    #to specify here
    
        CV=sd_intensity/avg_intesnity
        
        avg_vec.append(avg_intesnity)
        sd_vec.append(sd_intensity)
        nb_pixels_vec.append(nb_pixels)
        CV_vec.append(CV)
    
    CV_min=min(CV_vec) 
    CV_vec_normalized=[j/CV_min for j in CV_vec]
    
    CV_df={"Standard deviation":sd_vec,
           "Average":avg_vec,
           "Nb pixels": nb_pixels_vec,
           "CV":CV_vec ,
           "CVs relative to min value":CV_vec_normalized
           }
     
    CV_df=pd.DataFrame(CV_df) if sample_names=="NA" else pd.DataFrame(CV_df, sample_names)
    
    return CV_df 

GetCVforBW(path1)


def GetCVforRGB(path): #Useful ?? Are we working on RGB images ? 
    #Treating each of R, G and B as a single B&W file 
    #THIS FUNCTION NOT YET TESTED 
    tiff_images_aicsi=AICSImage(path, reader=readers.TiffReader)
    tiff_images_data=tiff_images_aicsi.data

    CV_dict={}
    
    for z in range(len(tiff_images_data)): #get 3 CV for each image
        #Strategy: Images of a multi tiff are considered by Aicsi as plans (z)
        #z here does not represent the xyz dimension but the img nb in the tiff file 
        R_channel= tiff_images_aicsi[:,0,z,:,:] #Get R grey scale of the image  
        G_channel= tiff_images_aicsi[:,1,z,:,:] #Aics lib always return order TCZYX
        B_channel= tiff_images_aicsi[:,2,z,:,:]
        
        channels_temp=[R_channel,G_channel,B_channel]
        
        mean_temp=[mean(i) for i in channels_temp]
        std_temp=[std(i) for j in channels_temp]
        #add pixel column : np_pixels (see same function for BW images)
        
        CV_temp=[std_temp[h]/mean_temp[h] for h in range(len(channels_temp))]
        
        CV_dict["Tiff_Image_"+str(z)]=CV_temp
    
    CV_df=pd.DataFrame(CV_dict, index=["R","G","B"])
    
    #Get normalized Cvs 
    CV_min_vec=CV_df.min(axis='index') #min of each row, return [minR, minG, minB]
    CV_df.div(CV_min_vec, axis='columns') #divide each col by [minR, minG, minB]
    
    return CV_df
        
def GetCV(path):
    color=GetColorInfo(path)
    if color==0 : #B&W
        CV=GetCVforBW(path)
    else:
        CV=GetCVforRGB(path)
    return CV

GetCV(path1)


#4. Report : Get Tiff images with ROIs marked on them. 

def GetOriginalWithROIs(path, x, y, h, w, output_dir, output_name, output_format="jpeg" ,origin="TL"):
    tiff_images_data=GetImagesFromeMultiTiff(path)
    image_list=[]
    output_path=output_dir+output_name
    for i in range(len(tiff_images_data)):
        image_temp=tiff_images_data[i]
        cv2.rectangle(image_temp, (x, y), (h, w), (255, 0, 0), 1)
        image_list.append(image_temp)
        cv2.imwrite(output_path+str(i)+"."+output_format, image_temp)
    
    return(image_list)
    
image1_with_roi=GetOriginalWithROIs(path1, 100, 100, 100+300, 100+200,
                                    output_dir="/Users/bottimacintosh/Desktop/",
                                    output_name="tests",
                                    output_format="png")    

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



#6. Report: Get Histogram : nb of pixels per intensity values
# In Progress

image0=GetROIsFromMultipleTiff(path1, 10, 10, 30, 20)[0]
image_flat=list(image0.flatten())
plt.hist(image_flat, linestyle='dashed', color="green",bins=256)
plt.xlim(xmin=0, xmax = 255)

"""
Sheet paper
"""
aa=AICSImage("/Users/bottimacintosh/Documents/M2_CMB/IBDML/Data/CV/cv.comparatif.tif", reader=readers.TiffReader)
path4="/Users/bottimacintosh/Desktop/imm.png"


