# MetroloJ-for-python
MetroloJ is an ImageJ software plugin that helps keeping track of microscopes' performances by extracting 4 relevant estimators out of standardized images, acquired from test samples[1]. \
The present project aims to provide MetroloJ functionalities as a python package named *metroloj*.

## *metroloj* package
*metroloj* is a pip package that provides functions to microscope images through image analysis assessment.

The first module of the *metroloj* package, named `common.py`, is developed to convert a microscope image file to a numpy array, i.e. a matrix of pixel intensity values. Besides, each image quality assessment is performed on the numpy array of the corresponding digital image.

In order to determine microscopes' health through image quality assessment, *metroloj* package carries in addition two modules: `cv.py` and `homo.py`. The `cv` and `homo` modules enclose Python functions that retrieve respectively the coefficient of variation and the illumination homogeneity indicators of the given microscope image. Both are indicators used to assess a fluorescence microscope's health, as detailed afterwards.

### *metroloj*: homo module
Controlling the homogeneity of illumination is one of the easiest but also one of the most important metrology tests to control the health of fluorescence microscopes. As we are testing the illumination homogeneity, the reference sample to use consists simply in a uniformly labelled sample, i.g. a simple fluorescent plastic slide. Two approaches are used to quantify microscopes' homogeneity illumination. The first approach consists in retrieving the intensity, known also as the gray levels, of 8 location specific pixels as well as localizing the pixel with the maximum grey scale value across the whole image. These results are then provided within a table comprising the pixel location column, the pixel intensity value and an additional column representing the relative intensity value of each of the 9 pixels with respect to the maximum intensity value. On the other hand, the second approach consists in retrieving lines of pixels from the digital image. These pixel sets correspond to the 4 symmetry axes of the image, all passing through the image center. The resulting grey scale values are then presented within a histogram.
This way, metrologists can easily determine whether the microscope illumination is homogenous or not by simply analyzing the obtained results.

### *metroloj*: cv module
Detectors play an essential role in fluorescence microscopy as they are responsible for the quantification process of the fluorochromes' emitted light. Moreover, detectors participate in extending the microscopes' limits and their performance is crucial for acquiring high quality images. Hence, there is no doubt that detectors' limit, known as sensitivity, is one of most important features of any imaging system. Detectors' sensitivity is defined as the efficiency of the detector to detect light. Consequently, the higher is the detector sensitivity, the better is the quality of the acquired image.
The sensitivity of a microscope detector is evaluated by computing the coefficient of variation (CV) indicator on an image of specific beads. 
Moreover, only beads that are inside the central 20% of the image are considered when computing the CV indicator. Furthermore, only beads' pixels are then taken into account in computing the detector's CV value. In this context, the `cv.py` *metroloj* module enables to label specifically these beads' pixels (in red) on the studied microscope image.
Basically, the CV indicator is defined as the ratio of the standard deviation ($\sigma$) over the average ($\mu$) of the bead's pixels intensity:

$$
cv = \sigma/\mu
$$

It's important to note that different types of beads are used to access different detectors' indicators and beads that are used to determine CV values are known as Rainbow beads with a diameter of 10 µm. The standard deviation, the  average intensity and the CV values, together with the total number of the bead's pixels are computed using the functions of the `cv.py` *metroloj* module and are provided within a table. In case of a multiple image file, this table carry an additional column representing the ratio of the corresponding cv value over the min cv value of the table. Moreover, this module provides the original images with labeled beads, i.e. images where bead's specific pixels are marked.
To have a better description of the sensitivity of the microscope detector, one may also consider the distribution of the bead's pixel values. Thus, the `cv.py` module enables to plot the bead's pixel values on a histogram. Moreover, when working with multiple images in a single file, the `cv.py` module enables to return the beads' pixel values distribution of each image in the same histogram. This makes it easier to analyze the detector sensitivity, especially when using confocal microscopy where images from the same file are simply images of the same sample but of different layers with respect to the z-axis.

# Requirements
In a virtual environment, run the following command:
```python3 -m pip install -r requirements.txt```

# References
[1] https://imagejdocu.tudor.lu/plugin/analysis/metroloj/start
