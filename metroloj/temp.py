import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from skimage import data
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.color import label2rgb
from skimage.draw import polygon_perimeter, circle_perimeter

path_cv = "/Users/Youssef/Documents/IBDML/Data/CV/cv.comparatif.tif"


def get_marked_roi_and_label_3d(tiff_data, output_dir=None):
    """
    This function:
    - mark by diffrent color the pixels that are considered when computing
    the cv value, i.e. having a higher intensity than the threshold otsu
    value.
    - mark by a red rectangle the region of pixels having an intensity higher
    than a threshold otsu
    which are used
    - mark by a white rectangle the roi region, i.e. the central 20% region
    of the inputed image.

    Parameters
    ----------
    tiff_data : numpay.ndarray
        3d np.array representing the image data.
    output_dir : str, optional
        Output directory path. The default is None.

    Returns
    -------
    fig_list : list
        1D list of figures of type matplotlib.figure.Figure.

    """
    fig_list = []
    for i in range(len(tiff_data)):
        image = tiff_data[i].astype(np.uint8)
        thresh = threshold_otsu(image)
        bw = closing(image > thresh, square(3))

        # limit the labelization to the roi region
        # roi coordinates
        roi_info, roi_arrays = get_roi_default(tiff_data)
        roi_minr, roi_minc = roi_info["ROI_start_pixel"][0]
        roi_maxr, roi_maxc = roi_info["ROI_end_pixel"][0]

        # remove outside roi region
        cleared = clear_border(bw, buffer_size=int((512/2)-roi_minr))

        # label image regions
        label_image = label(cleared)

        # set background to transparent
        image_label_overlay = label2rgb(label_image, image=image, bg_label=0)

        # mark roi on image_label_overlay
        rr, cc = polygon_perimeter([roi_minr, roi_minr, roi_maxr, roi_maxr],
                                   [roi_minc, roi_maxc, roi_maxc, roi_minc],
                                   shape=tiff_data[i].shape,
                                   clip=True)
        image_label_overlay[rr, cc, :] = 255

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(image_label_overlay)

        for region in regionprops(label_image):
            minr, minc, maxr, maxc = region.bbox
            # take regions with large enough areas and inside the ROI
            if region.area >= 100:
                # draw rectangle around segmented coins
                rect = mpatches.Rectangle((minc, minr), maxc - minc,
                                          maxr - minr, fill=False,
                                          edgecolor='red', linewidth=2)
                ax.add_patch(rect)

        ax.set_axis_off()
        plt.tight_layout()
        plt.show()
        # plt.axis(off)
        fig_list.append(fig)

        if output_dir is not None:
            plt.savefig(output_dir+"test"+str(i)+".png",
                        bbox_inches='tight',
                        pad_inches=0)
    return fig_list


"""
tiff_data = cm.get_images_from_multi_tiff(path_cv)
output_dir = "/Users/Youssef/Desktop/"
figure = get_marked_roi_and_label_3d(tiff_data)
figure[0]
figure[1]
figure = get_marked_roi_and_label_3d(tiff_data, output_dir)
"""


def get_marked_roi_and_label_2d(tiff_data):

    fig_list = []

    original_img_temp = tiff_data[i]
    img_temp = original_img_temp

    # define intensity threshold
    thresh = threshold_otsu(img_temp)

    # boolean matrice: True represent the pixels of interest
    bw = closing(img_temp > thresh, square(3))

    # limit the labelization to the roi region
    # a. roi coordinates
    roi_info, roi_arrays = get_roi_default(tiff_data)
    roi_minr, roi_minc = roi_info["ROI_start_pixel"][0]
    roi_maxr, roi_maxc = roi_info["ROI_end_pixel"][0]
    # b. remove outside roi region
    cleared = clear_border(bw, buffer_size=int((512/2)-roi_minr))

    # c. label image regions
    label_image = label(cleared)

    # mark labeled regions
    for region in regionprops(label_image):
        minr, minc, maxr, maxc = region.bbox
        r_center, c_center = region.centroid
        r_center, c_center = int(r_center), int(c_center)
        # take regions with large enough areas
        if region.area >= 100:
            # draw rectangle around segmented coins
            rr_ball, cc_ball = circle_perimeter(r_center,
                                                c_center,
                                                radius=int((maxc-minc)/2))
            original_img_temp[rr_ball, cc_ball] = 255

    # mark roi
    rr, cc = polygon_perimeter([roi_minr, roi_minr, roi_maxr, roi_maxr],
                               [roi_minc, roi_maxc, roi_maxc, roi_minc],
                               shape=tiff_data[i].shape,
                               clip=True)
    original_img_temp[rr, cc] = 255

    fig_list.append(original_img_temp.astype(np.uint8))

    return fig_list


"""
tiff_data = cm.get_images_from_multi_tiff(path_cv)
img0 = tiff_data[0]
img1 = tiff_data[1]

img_marked_roi_and_label0 = get_marked_roi_and_label(img0)
img_marked_roi_and_label1 = get_marked_roi_and_label(img1)

Image.fromarray(img_marked_roi_and_label0)
Image.fromarray(img_marked_roi_and_label1)

get_marked_roi_and_label(tiff_data)
"""
