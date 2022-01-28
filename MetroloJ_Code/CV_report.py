#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 14:23:09 2022

@author: bottimacintosh
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

#pip install aicsimageio
#pip install 'Pillow==8.0.1' #aicsimageio need specific version of PIL
from PIL import Image
from aicsimageio import AICSImage, imread_dask
from aicsimageio import readers
#pip install 'aicsimageio[base-imageio]'
from collections import Counter


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

#1. Import a stack of images containing the acquisitions made with the PMTs to analyze
#Use paths as inputs


def GetImagesFromeMultiTiff(path):
    img = Image.open(path)
    tiff_images = []
    for i in range(img.n_frames):
        img.seek(i)
        tiff_images.append(np.array(img))
    return tiff_images

#ex
GetImagesFromeMultiTiff(path1)

#2. Get same ROI for each image in the same tiff

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

#ex
GetROIsFromMultipleTiff(path1, 100,100, 300, 300)

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
        
        mean_temp=[np.mean(i) for i in channels_temp]
        std_temp=[np.std(j) for j in channels_temp]
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

#ex1
GetCVforBW(path1)
GetCV(path1)


#4. Report : Get Tiff images with ROIs marked on them. 

def GetOriginalWithMarkedROIs(path, x, y, h, w, mode, output_dir, output_name, output_format="jpeg" ,origin="TL"):
    tiff_images_data=GetImagesFromeMultiTiff(path)
    
    if len(tiff_images_data)>1 :
        image_list=[]
        output_path=output_dir+output_name
        for i in range(len(tiff_images_data)):
            image_temp_data=tiff_images_data[i]
            cv2.rectangle(image_temp_data, (x, y), (h, w), (255, 0, 0), 1)
            image_temp_toshow=Image.fromarray(image_temp_data, mode=mode)
            image_list.append(image_temp_toshow)
            cv2.imwrite(output_path+str(i)+"."+output_format, image_temp_data)
 
    else:
        output_path=output_dir+output_name
        image_temp_data=tiff_images_data[0]
        cv2.rectangle(image_temp_data, (x, y), (h, w), (255, 0, 0), 1)
        image_temp_toshow=Image.fromarray(image_temp_data, mode=mode)
        image_list=image_temp_toshow
        cv2.imwrite(output_path+"."+output_format, image_temp_data)
        
    
    return image_list

#ex1    
image1_with_roi=GetOriginalWithMarkedROIs(path1, 100, 100, 100+300, 100+200,mode="L", #mode="RGB" if colored img
                                    output_dir="/Users/bottimacintosh/Desktop/",
                                    output_name="tests",
                                    output_format="png")
image1_with_roi[0]
image1_with_roi[1]

#ex2
GetOriginalWithMarkedROIs(path0, 0, 0, 100+300, 100+200,mode="L", 
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

#ex
GetMicroscopyInfo("Confocal",460.0,1.4,"1.0x1.0x1.0", 1.0)


#6. Report: Get Histogram : nb of pixels per intensity values
# In Progress

def GetHistNbPixelVSGrayScale(image_2d):
    
    image_flat=list(image_2d.flatten()) #convert matrix to one vector
    count_df=pd.DataFrame.from_dict(Counter(image_flat), orient='index').reset_index()
    count_df=count_df.rename(columns={'index':'intensity', 0:'count'})
    count_df=count_df.sort_values(by="intensity", axis=0, ascending=True)

    plt.plot(count_df['intensity'], count_df["count"] ,marker=".", markersize=0.2,  color='b', label= 'ROI pixels', linewidth=0.8)
    
    plt.title("Intensity histogram")
    plt.xlim((0,256)) #25 subjective choice
    plt.xlabel("Gray levels")
    plt.ylabel("Nb Pixels")
    plt.legend()

#ex1
img1=GetImagesFromeMultiTiff(path1)[0]
GetHistNbPixelVSGrayScale(img1)
#ex2
ex=np.random.randint(255, size=(100, 100, 10), dtype=np.uint8)
GetHistNbPixelVSGrayScale(ex)
#ex3
image0=GetImagesFromeMultiTiff(path1)[0]
GetHistNbPixelVSGrayScale(image0)
