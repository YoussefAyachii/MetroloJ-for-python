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

def GetCVforBW(path):
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
     
    CV_df=pd.DataFrame(CV_df) 
    
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
        
def GetCV(path, sample_names="NA"):
    color=GetColorInfo(path)
    if color==0 : #B&W
        CV=GetCVforBW(path)
    else:
        CV=GetCVforRGB(path)
    
    #If desired, add names for each image of the Tiff
    if sample_names!="NA" :
        CV.index=sample_names
        
    return CV

#ex1
GetCVforBW(path1)
GetCV(path1)


#4. Report : Get Tiff images with ROIs marked on them. 

def GetOriginalWithMarkedROIs(path, x, y, h, w, origin="TL"):
    tiff_images_data=GetImagesFromeMultiTiff(path)
    mode_temp="RGB" if GetColorInfo(path)==1 else "L"
    
    if len(tiff_images_data)>1 :        
        image_list=[]
        for i in range(len(tiff_images_data)):
            
            image_temp_data=tiff_images_data[i]
            cv2.rectangle(image_temp_data, (x, y), (h, w), (255, 0, 0), 1)
            image_temp_toshow=Image.fromarray(image_temp_data, mode=mode_temp)
            image_list.append(image_temp_toshow)
 
    else:
        image_temp_data=tiff_images_data[0]
        cv2.rectangle(image_temp_data, (x, y), (h, w), (255, 0, 0), 1)
        image_temp_toshow=Image.fromarray(image_temp_data, mode=mode_temp)
        image_list=image_temp_toshow
        
    
    return image_list

#ex1    
image1_with_roi=GetOriginalWithMarkedROIs(path1, 100, 100, 100+300, 100+200) #mode="RGB" if colored img
                                    
image1_with_roi[0]
image1_with_roi[1]

#ex2
GetOriginalWithMarkedROIs(path0, 0, 0, 100+300, 100+200)


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

def GetHistNbPixelVSGrayScale(path, x=0, y=0, h=0, w=0):
    tiff_images_data=GetImagesFromeMultiTiff(path)
    
    #if x,y,h and w == 0 : GetHist on the whole image. 
    #if  x,y,h and w specified : GetHist of specidfied ROI
    if x==0 and y==0 : 
        tiff_images_data=GetImagesFromeMultiTiff(path)
    else :
        tiff_images_data=GetImagesFromeMultiTiff(path)
        for i in range(len(tiff_images_data)):
            tiff_images_data[i]=tiff_images_data[i][x:(x+h),y:(y+h)]
        
    #build histograms for each image of the tiff   
    if len(tiff_images_data)>1 :
        count_df_list=[]
        fig = plt.figure()
        for i in range(len(tiff_images_data)):
            image_2d_temp=tiff_images_data[i]
    
            image_flat_temp=list(image_2d_temp.flatten()) #convert matrix to one vector
            count_df_temp=pd.DataFrame.from_dict(Counter(image_flat_temp), orient='index').reset_index()
            count_df_temp=count_df_temp.rename(columns={'index':'intensity', 0:'count'})
            count_df_temp=count_df_temp.sort_values(by="intensity", axis=0, ascending=True)
            count_df_list.append(count_df_temp)
            
        colours=['r','g','b','k']
        for j in range(len(tiff_images_data)) :
            plt.plot(count_df_list[j]['intensity'], count_df_list[j]["count"] ,marker=".", markersize=0.2,  color=colours[j], label= 'ROI pixels '+str(j), linewidth=0.8,  figure=fig)
            

    else : 
        fig = plt.figure()
        image_2d=tiff_images_data[0]
        image_flat=list(image_2d.flatten()) #convert matrix to one vector
        count_df=pd.DataFrame.from_dict(Counter(image_flat), orient='index').reset_index()
        count_df=count_df.rename(columns={'index':'intensity', 0:'count'})
        count_df=count_df.sort_values(by="intensity", axis=0, ascending=True)
        plt.plot(count_df['intensity'], count_df["count"] ,marker=".", markersize=0.2,  color="b", label= 'ROI pixels', linewidth=0.8,  figure=fig)


    plt.title("Intensity histogram", figure=fig)
    plt.xlim((0,256)) 
    plt.xlabel("Gray levels")
    plt.ylabel("Nb Pixels")
    plt.legend()
    plt.title("Intensity histogram", figure=fig)
    plt.show() #to pythn verification

    return fig

#ex1
GetHistNbPixelVSGrayScale(path1, 100,100,200,200)
#ex2
GetHistNbPixelVSGrayScale(path1)


"""
REPORT
"""

def GetCVReportElements (path, x, y, h, w, Microscope_type, Wavelength, NA, Sampling_rate, Pinhole):
    #Get image from path; 2d image from path; 
    ImagesFromeMultiTiff=GetImagesFromeMultiTiff(path)
    
    if len(ImagesFromeMultiTiff)>1 :
        OriginalWithMarkedROIs_list=[]
        
        for i in range(len(ImagesFromeMultiTiff)):
            #Get Images with Marked ROIs on them
            OriginalWithMarkedROIs_temp=GetOriginalWithMarkedROIs(path, x, y, h, w, origin="TL")[i]
            OriginalWithMarkedROIs_list.append(OriginalWithMarkedROIs_temp)
        
        #CV
        CV=GetCV(path)
            
        #Microscope info dataframe 
        MicroscopyInfo=GetMicroscopyInfo(Microscope_type, Wavelength, NA, Sampling_rate, Pinhole)
        
        #Histogram : Nbpixel VS Gray scale
        HistNbPixelVSGrayScale=GetHistNbPixelVSGrayScale(path)
        
        #Output for report
        CVReportComponents=[OriginalWithMarkedROIs_list, MicroscopyInfo, HistNbPixelVSGrayScale, CV]
        
    else :
        #Get Images with Marked ROIs on them
        OriginalWithMarkedROIs=GetOriginalWithMarkedROIs(path, x, y, h, w, origin="TL")
        
        #Get Microscope info dataframe 
        MicroscopyInfo=GetMicroscopyInfo(Microscope_type, Wavelength, NA, Sampling_rate, Pinhole)
        
        #Get Histogram : Nbpixel VS Gray scale
        HistNbPixelVSGrayScale=GetHistNbPixelVSGrayScale(path)
        
        #Get CV table
        CV=GetCV(path)
        CVReportComponents=[OriginalWithMarkedROIs,MicroscopyInfo ,  HistNbPixelVSGrayScale, CV]
    
    return CVReportComponents

#ex1
CVreport_elements_1=GetCVReportElements(path1, 100, 100, 300, 300, "Confocal", 460, 1.4, "1.0x1.0x1.0", 1)
CVreport_elements_1[0][0]
CVreport_elements_1[0][1]
CVreport_elements_1[1]
CVreport_elements_1[2]
CVreport_elements_1[3]

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
    
"""
Get default ROI : seg seuillage 
"""
img_2d=GetImagesFromeMultiTiff(path1)[0]
Image.fromarray(img_2d)
img=img_2d

#1. segmentation : threshold to detect billes
#Thresholding is used to create a binary image from a grayscale image
from skimage.filters import threshold_otsu
import math
thresh=threshold_otsu(img)
binary = img > thresh

#2. ROI default : central 20% of the image 
def GetROIDefault(img):
    x,y = img.shape
    
    perc=20 #We want to get the central 20% of the input image
    q=math.sqrt(perc/100)    
    
    output_height_size=int((q*100)*x//100)
    output_width_size=int((q*100)*y//100)
    
    startx = x//2-(output_height_size//2)
    starty = y//2-(output_width_size//2) 
    
    ROI_data=img[startx:startx+output_height_size, starty:starty+output_width_size]
    
    ROI_start_pixel=[startx,starty]
    ROI_end_pixel=[startx+output_height_size,starty+output_width_size]
    
    xtot,ytot=np.shape(ROI_data)
    ROI_nb_pixels=[xtot, ytot]
    
    #2 outputs :
        #1. dict : ROI infos
    ROI_info={}
    ROI_info["Original_image_dim"]=np.shape(img)
    ROI_info["ROI_nb_pixels"]=ROI_nb_pixels
    ROI_info["ROI_start_pixel"]=ROI_start_pixel
    ROI_info["ROI_end_pixel"]=ROI_end_pixel
    ROI_info=pd.DataFrame(ROI_info)
        
        #2. image : ROI and ROI numpy matrix 
    ROI_Pil=Image.fromarray(ROI_data)   
    
    return [ROI_info, ROI_Pil, ROI_data]

#ex
GetROIDefault(img)[0]["ROI_start_pixel"][0]
GetROIDefault(img)[1]
GetROIDefault(img)[2]


#Udated function : 
def GetOriginalWithMarkedROIsDefault(img):
    
    ROI_info=GetROIDefault(img)[0]
    x0,y0=ROI_info["ROI_start_pixel"][0], ROI_info["ROI_start_pixel"][1]
    h,w=x0+ROI_info["ROI_nb_pixels"][0], y0+ROI_info["ROI_nb_pixels"][1]
    
    cv2.rectangle(img, (x0, y0), (h, w), (255, 0, 0), 1)
    img_with_ROI=Image.fromarray(img, mode="L") #2D images: no need for mode="RGB" option 
    image_list=img_with_ROI

    return image_list    

#ex
GetOriginalWithMarkedROIsDefault(img)


#ex complete execution 

image=GetImagesFromeMultiTiff(path1)[0]
image=np.array(GetOriginalWithMarkedROIsDefault(img))


thresh = threshold_otsu(image)
binary = image > thresh

fig, axes = plt.subplots(ncols=3, figsize=(8, 2.5))
ax = axes.ravel()
ax[0] = plt.subplot(1, 3, 1)
ax[1] = plt.subplot(1, 3, 2)
ax[2] = plt.subplot(1, 3, 3, sharex=ax[0], sharey=ax[0])

ax[0].imshow(image, cmap=plt.cm.gray)
ax[0].set_title('Original')
ax[0].axis('off')

ax[1].hist(image.ravel(), bins=256)
ax[1].set_title('Histogram')
ax[1].axvline(thresh, color='r')

ax[2].imshow(binary, cmap=plt.cm.gray)
ax[2].set_title('Thresholded')
ax[2].axis('off')

plt.show()







    