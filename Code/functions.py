#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 11:07:27 2022

@author: bottimacintosh
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#pip install aicsimageio
#pip install 'Pillow==8.0.1' #aicsimageio need specific version of PIL
from PIL import Image
from aicsimageio import AICSImage, imread_dask

import time

#image example
photo = Image.open("/Users/bottimacintosh/Downloads/pup3.tif", mode="r")
photo_data = np.array(photo) 
photo_aicsi=AICSImage("/Users/bottimacintosh/Downloads/pup3.tif", mode="r")
#An ndarray is a multidimensional container of items (matrices) of the same type and size.


class metroloJ:
    
    def __init__ (self, image_path, colors, z, 
                  microscopeType, emission_length, num_aperture, pin_aperture):
        
        self.image_path=image_path
        self.couleur=bool(couleur) #image b&w {0} ; image colors/RGB {1}, needed to specify the C in dimensions 
        self.z=bool(z) #z : axis Z exist? Enter True/False
        
        self.microscopeType=microscopeType
        self.emission_length=emission_length
        self.num_aperture=num_aperture
        self.pin_aperture=pin_aperture
    
    
    
    image_aicsi=AICSImage(image_path, mode="r") #import image to AICSImage
    image_data=image.data ##convert image to np.ndarrry
    #use image_aicsi to get .shape, .dims, .. 
    #use image_data when working on np.ndarray
        
        
        
        

    
#Get ROI image using the selected pixel coordinates (x,y)
#Chose the height (h), width (w) of the ROI
#The image origin in use may also vary: top light (TL) is the default value.
def GetROI (image_aicsi, x, y, h , w, origin="TL"):   

    image_data=image_aicsi.data #always order : STCZYX
    image_dim=len(image_aicsi.shape)
       
    if origin=="TL" : #origin = top left
        if  image_dim==2:
            ROI=image_data[y:y+w, x:x+h]
        elif image_dim==3:
            ROI=image_data[:, y:y+w, x:x+h]
        elif image_dim==4:
            ROI=image_data[:, :, y:y+w, x:x+h]
        elif  image_dim==5:
            ROI=image_data[:, :, :, y:y+w, x:x+h]
        elif  image_dim==6:
            ROI=image_data[:, :, :, :, y:y+w, x:x+h]
            
    elif origin=="TR" : #origin = top right
        if  image_dim==2:
            ROI=image_data[y:y-w, x:x+h]
        elif image_dim==3:
            ROI=image_data[:, y:y-w, x:x+h]
        elif image_dim==4:
            ROI=image_data[:, :, y:y-w, x:x+h]
        elif  image_dim==5:
            ROI=image_data[:, :, :, y:y-w, x:x+h]
        elif  image_dim==6:
            ROI=image_data[:, :, :, :, y:y-w, x:x+h]
        
    elif origin=="BL" : #origin = bottom left
        if  image_dim==2:
            ROI=image_data[y:y+w, x:x-h]
        elif image_dim==3:
            ROI=image_data[:, y:y+w, x:x-h]
        elif image_dim==4:
            ROI=image_data[:, :, y:y+w, x:x-h]
        elif  image_dim==5:
            ROI=image_data[:, :, :, y:y+w, x:x-h]
        elif  image_dim==6:
            ROI=image_data[:, :, :, :, y:y+w, x:x-h]
            
    elif origin=="BR": #origin = bottom right
        if  image_dim==2:
            ROI=image_data[y:y-w, x:x-h]
        elif image_dim==3:
            ROI=image_data[:, y:y-w, x:x-h]
        elif image_dim==4:
            ROI=image_data[:, :, y:y-w, x:x-h]
        elif  image_dim==5:
            ROI=image_data[:, :, :, y:y-w, x:x-h]
        elif  image_dim==6:
            ROI=image_data[:, :, :, :, y:y-w, x:x-h]
            
    if origin not in ["TL","TR", "BL","BR" ]:
        print(" 'origin' param can only get 'TL', 'TR', 'BL' or 'BR'")
    
    return ROI

   
def GetDimLabelAndValue(ROI): #!!! ROI have to be AICSImage type not np.ndarray
    #we know that dimension C exist (using if GetColorInfo=1)
    ROI_dim=len(ROI.shape)

    #Get a list : X,Y,Z,.. of the ROI
    ROI_dim_labels=str(ROI.dims).replace("Dimensions ","") 
    ROI_dim_labels=list( ''.join([c for c in ROI_dim_labels if c.isupper()]) )
    ROI_dim_labels.reverse() # X, Y, .. (right order)
    
    #Get a list of the values of: X,Y,Z,.. of the ROI
    ROI_dim_values=list(ROI.shape)
    ROI_dim_values.reverse() # 256, 256, ..

    dim_matrix=[ROI_dim_labels, ROI_dim_values]

    return(dim_matrix)

"""
The gray level histogram for each of the channels is simply the gray level histogram
 of the Red channel, the Green channel and the Blue channel separately. 
In a colour RGB image, the number of channels is 3.
1. Extract the image RGB comonents
2. In a colour RGB image, the number of channels is 3.This can be viewed as 3 images
packed together in one "3D" matrix. To work on the Red channel, all that you have
to do is use slicing to obtain just one of those matrices.
"""


def GetColorInfo(ROI):
    #0 B&W or 1 Colored
    try:
        a=list(ROI.dims["C"])[0]
        color=1 #color
    except:
        color=0 #B&W
    return color


def GetRGBChannels(ROI): #!!! ROI have to be AICSImage type not np.ndarray
    #we know that dimension C exist (using : if GetColorInfo==1)
    ROI_dim=len(ROI.shape)

    #Get a list : X,Y,Z,.. of the ROI
    ROI_dim_labels=GetDimLabelAndValue(ROI)[0]
    
    #Get a list of the values of: X,Y,Z,.. of the ROI
    ROI_dim_values=GetDimLabelAndValue(ROI)[1]
    
    #Get R,G,B channels given the dim of ROI and the position of C in XYZCT
    C_index=ROI_dim_labels.index("C")
    ROI_data=ROI.data
    
    if ROI_dim==3:
        if C_index==0: #if for ex CXY
            #RGBChannels list: [ R channel, G channel, B channel]
            RGBChannels_list=[ROI_data[i,:,:] for i in range(3)] #3 for 3 channels RGB
        elif C_index==1: 
            RGBChannels_list=[ROI_data[:,i,:] for i in range(3)]
        elif C_index==2: 
            RGBChannels_list=[ROI_data[:,:,i] for i in range(3)]
     
            
    if ROI_dim==4:
        if C_index==0: 
            RGBChannels_list=[ROI_data[i,:,:,:] for i in range(3)]
        elif C_index==1: 
            RGBChannels_list=[ROI_data[:,i,:,:] for i in range(3)]
        elif C_index==2: 
            RGBChannels_list=[ROI_data[:,:,i,:] for i in range(3)]
        elif C_index==3: 
            RGBChannels_list=[ROI_data[:,:,:,i] for i in range(3)]
            
    if ROI_dim==5:
        if C_index==0: 
            RGBChannels_list=[ROI_data[i,:,:,:,:] for i in range(3)]            
        elif C_index==1: 
            RGBChannels_list=[ROI_data[:,i,:,:,:] for i in range(3)]                         
        elif C_index==2: 
            RGBChannels_list=[ROI_data[:,:,i,:,:] for i in range(3)]                             
        elif C_index==3: 
            RGBChannels_list=[ROI_data[:,:,:,i,:] for i in range(3)]                             
        elif C_index==4: 
            RGBChannels_list=[ROI_data[:,:,:,:,i] for i in range(3)]

        return(RGBChannels_list)
    
    if ROI_dim==6:
        if C_index==0: 
            RGBChannels_list=[ROI_data[i,:,:,:,:,:] for i in range(3)]            
        elif C_index==1: 
            RGBChannels_list=[ROI_data[:,i,:,:,:,:] for i in range(3)]                         
        elif C_index==2: 
            RGBChannels_list=[ROI_data[:,:,i,:,:,:] for i in range(3)]                             
        elif C_index==3: 
            RGBChannels_list=[ROI_data[:,:,:,i,:,:] for i in range(3)]                             
        elif C_index==4: 
            RGBChannels_list=[ROI_data[:,:,:,:,i,:] for i in range(3)]
        elif C_index==5: 
            RGBChannels_list=[ROI_data[:,:,:,:,:,i] for i in range(3)]

        return(RGBChannels_list)
        
    return(RGB_Channels_list)


"""
GET CV
for each image of a stack (r,g,b/x,y,z), get the mean and sd of the grey levels, 
i.e. for each of the 3 matrices r g b of the ROI get the mean and sd. 
"""
def GetCVforRGB(ROI):
    RGB_list=GetRGBChannels(ROI)
     
    avg_intesnity=[np.mean(i) for i in RGB_list]
    
    sd_intensity=[np.std(j) for j in RGB_list]
    
    CV=[sd_intensity[k]/avg_intesnity[k] for k in range(len(RGB_list))]
    
    CV_min=min(CV)
    CV_normalized=[l/CV_min for l in CV]
    
    return [CV, CV_normalized]

def GetCVforBW(ROI):
    Channel=ROI.data
     
    avg_intesnity=np.mean(Channel) 
    
    sd_intensity=np.std(Channel)
    
    CV=sd_intensity/avg_intesnity
    
    CV_min=min(CV) #problemmmmmmmmm in BW , how to normalize
    CV_normalized=[l/CV_min for l in CV]
    
    return [CV, CV_normalized]



def GetCV(ROI, normalized=1):
    
    if GetColorInfo(ROI)==1 : #if dimension C exist
        Channels=GetRGBChannels(ROI)
        CV_normalized=GetCVforRGB(ROI)[1]
        CV_non_normalized=GetCVforRGB(ROI)[0]
        
    if GetColorInfo(ROI)==0 :  #if dimension C does not exist
        Channels=ROI.data #a unique channel (and not 3 as in RGB)
        CV_normalized=GetCVforBW(ROI)[1]
        CV_non_normalized=GetCVforBW(ROI)[0]
    
    # The user can chose to print the normalized or non normalized Cv values
    if bool(normalized)==True :
        CV=CV_normalized
    else :
        CV=CV_non_normalized
        
    return CV


def GetImageFromROI(ROI): 
    ROI_data=ROI.data
    img=Image.fromarray(ROI_data, 'RGB') 
    img.show()
    


"""
Execution
"""
ex=np.random.randint(255, size=(100, 100, 10), dtype=np.uint8)
ex_data=ex.data

ex2=AICSImage("/Users/bottimacintosh/Documents/M2_CMB/IBDML/coalignement.czi")
ex2_data=ex2.data

ex3=ex2_data[:,:,:]
np.shape(ex3)
ex3_squeeze=(ex3.astype(np.uint8))
Image.fromarray(ex3_squeeze, mode="BGR;32")


start_time = time.time()
print(GetCV(ex))
print("--- %s seconds ---" % (time.time() - start_time))


import czifile
from skimage import io
img = AICSImage('/Users/bottimacintosh/Documents/M2_CMB/IBDML/coalignement.czi')
img_data=img.data # returns 6D STCZYX numpy array
list(img.dims["Y"]) # returns value of dimension Y
len(img.shape)







e=GetDimDict(ex2)
type(e)


dim_labels=str(ex2.dims).replace("Dimensions ","")
dim_labels=list( ''.join([c for c in dim_labels if c.isupper()]) )
dim_labels.reverse() # X, Y, ..

C_inde=dim_labels.index("C")
ROI_data=ROI.data

ex2_open = Image.open('')
im.show()


""" 
Homogeneity
"""


#get diagonals of a matrix/2d image
def GetDiagonalVecs(image_2d):
    photo_diagUpDown=np.diagonal(image_2d) #use numpy to get diag. (from left up to bottom right)
    
    photo_diagDownUp=np.ones((1,len(photo_diagUpDown)))[0]
    increase_values=list(range(len(photo_diagUpDown)))
    decrease_values=increase_values
    decrease_values.reverse()
    for i in range(len(photo_diagUpDown)):
        #start from last row - first column toward first row - last column
        x_temp=decrease_values[i]
        y_temp=increase_values[i] 
        
        pixel_temp=image_2d[x_temp,y_temp]
        photo_diagDownUp[i]=pixel_temp
    diags=[photo_diagUpDown , photo_diagDownUp]
    return diags

def GetVHsegments(image_2d):
    row_mid_index,col_mid_index=round(np.shape(image_2d)[0]/2),round(np.shape(image_2d)[1]/2)
    photo_V=image_2d[row_mid_index,:]
    photo_H=image_2d[:,col_mid_index]
    VH_seg=[photo_V, photo_H]
    return VH_seg

def GetIntensityPlot(image_2d):
    
    V_seg=GetVHsegments(photo_data)[0]
    H_seg=GetVHsegments(photo_data)[1]
    #get the x axis for VH segs
    nb_pixels_seg=len(V_seg)
    x_axis_seg=list(np.arange(-nb_pixels_seg/2, nb_pixels_seg/2+1, 1))
    x_axis_seg.remove(0) #as the center of the matrix is 4 pixels not one 
    
    
    photo_diag1=GetDiagonalVecs(photo_data)[0]
    photo_diag2=GetDiagonalVecs(photo_data)[1]
    #get the x axis for VH segs
    nb_pixels_diag=len(photo_diag1)
    x_axis_diag=list(np.arange(-nb_pixels_diag/2, nb_pixels_diag/2+1, 1))
    x_axis_diag.remove(0) # same
    
    #plots
    plt.plot(x_axis_seg, V_seg, marker = '+', color = 'b',label = 'V_seg')
    plt.plot(x_axis_seg, H_seg, marker = '+', color = 'g',label = 'H_seg')

    plt.plot(x_axis_diag, photo_diag1,marker='.', color='r', label= 'Diag1')
    plt.plot(x_axis_diag, photo_diag2,marker='.', color='r', label= 'Diag2')

    plt.axvline(0, linestyle='--')    
    plt.legend()


# Profile statistics

def GetProfileStatisticsTable(image_2d):
    
    #1. find the maximum intensity and the corresponding pixel.
    max_intensity=np.max(image_2d)
    nb_pixels_with_max_intensity=len(np.where(image_2d==max_intensity)[0]) #added by me
    #we chose only the first localization if the max intensity is in >1 pixels
    x_index_max_intensity=np.where(image_2d==max_intensity)[0][0]
    y_index_max_intensity=np.where(image_2d==max_intensity)[1][0]
    max_found_at=[x_index_max_intensity, y_index_max_intensity]
    relative_to_max=1.0
    
    #2. Top-left corner intensity and its ratio over max_intensity
    TL_x_index, TL_y_index = [0,0]
    TL_intensity=image_2d[TL_x_index, TL_y_index]
    TL_relative=TL_intensity/max_intensity
    
    #3. Top-right corner intensity and its ratio over max_intensity
    TR_x_index, TR_y_index = [0, np.shape(image_2d)[1]-1]
    TR_intensity=image_2d[TR_x_index, TR_y_index]
    TR_relative=TR_intensity/max_intensity
    
    #4. Bottom-left corner intensity and its ratio over max_intensity
    BL_x_index, BL_y_index = [np.shape(image_2d)[0]-1, 0]
    BL_intensity=image_2d[BL_x_index, BL_y_index]
    BL_relative=BL_intensity/max_intensity
    
    #5. Bottom-right corner intensity and its ratio over max_intensity
    BR_x_index, BR_y_index = [np.shape(image_2d)[0]-1, np.shape(image_2d)[1]-1]
    BR_intensity=image_2d[BR_x_index, BR_y_index]
    BR_relative=BR_intensity/max_intensity
    
    #6. Upper - Middle pixel 
    UM_x_index, UM_y_index = [0, round(np.shape(image_2d)[1]/2)]
    UM_intensity=image_2d[UM_x_index, UM_y_index]
    UM_relative=UM_intensity/max_intensity
    
    #7. Bottom - Middle pixel 
    BM_x_index, BM_y_index = [np.shape(image_2d)[0]-1, round(np.shape(image_2d)[1]/2)]
    BM_intensity=image_2d[BM_x_index, BM_y_index]
    BM_relative=BM_intensity/max_intensity
    
    #8. Left - Middle pixel
    LM_x_index, LM_y_index = [round(np.shape(image_2d)[0]/2), 0]
    LM_intensity=image_2d[LM_x_index, LM_y_index]
    LM_relative=LM_intensity/max_intensity
    
    #9. Right Middle pixel 
    RM_x_index, RM_y_index = [round(np.shape(image_2d)[0]/2), np.shape(image_2d)[1]-1]
    RM_intensity=image_2d[RM_x_index, RM_y_index]
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

#Get normalized intensity matrix (in %) : divide all the pixels' intensity by the max_intensity
def GetNormIntensityMatrix(image_2d): #used in both N.I.Profile and Centers' Locations
    max_intensity=np.max(image_2d)
    norm_intensity_profile=image_2d/max_intensity * 100 #the rule of three : max_intensity->100%, pixel_intensity*100/max
    return norm_intensity_profile

#Get normalized intensity profile (plot)
def GetNormIntensityProfile(image_2d):
    norm_intensity_profile=GetNormIntensityMatrix(image_2d)

    plt.imshow(norm_intensity_profile)
    plt.colorbar ( )
    plt.title("Normalized intensity profile")

#Get the centers' locations 
import imageio as iio
from skimage import filters
from skimage.measure import regionprops

# pip install --upgrade imutils
 import cv2

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]

CentersLocations={}
#columns: 
#"Image centre", "Center of intensity", "Center of max intensity", "Center of 100% zone"]
    
#Execution on photo_data

GetIntensityPlot(photo_data)

GetProfileStatisticsTable(photo_data)

GetMicroscopyInfo("Confocal", 405.0, 0.6, "1.0x1.0x1.0" , 1.0 )

GetNormIntensityProfile(photo_data)

