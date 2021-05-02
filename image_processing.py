import numpy as np
import cv2
import glob
import os
from matplotlib import pyplot as plt

def sort_function(filename):
    filename = filename.split('\\')[1]
    return int(filename[:-4])

image_directory = 'static/images/*'

files = sorted(glob.glob(image_directory), key=sort_function)

selected_images = files[:]

temporal_window_size = len(selected_images)
im_to_denoise_index = temporal_window_size//2

if temporal_window_size % 2 == 0:
    temporal_window_size -= 1

image_list = []

for image in selected_images:
    img = cv2.imread(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized_img = cv2.resize(gray, (90, 60))
    image_list.append(resized_img)

dst = cv2.fastNlMeansDenoisingMulti(image_list, im_to_denoise_index, temporal_window_size, None, 4, 7, 35)
# adaptive gaussian thresholding
thresh = cv2.adaptiveThreshold(dst, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7, 2)
thresh2 = cv2.adaptiveThreshold(dst, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7, 4)

filename = selected_images[im_to_denoise_index]
filename = filename[:-4]

cv2.imwrite(filename + str('-thresholded.jpg'), dst)    # denoising multi
cv2.imwrite(filename + str('-thresholded1.jpg'), thresh)    # denoising multi with adaptive threshold
cv2.imwrite(filename + str('-thresholded2.jpg'), thresh2)   # denoising multi with adaptive threshold 2


# thresholding without nlmeans
img = cv2.imread(selected_images[im_to_denoise_index])
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
resized_img = cv2.resize(gray, (90, 60))

thresh3 = cv2.fastNlMeansDenoising(resized_img, None, 5, 7, 35)
thresh4 = cv2.adaptiveThreshold(thresh3, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7, 2)
thresh5 = cv2.adaptiveThreshold(thresh3, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7, 4)

cv2.imwrite(filename + str('-thresholded3.jpg'), thresh3)   # denoising single
cv2.imwrite(filename + str('-thresholded4.jpg'), thresh4)   # denoising single with adaptive threshold
cv2.imwrite(filename + str('-thresholded5.jpg'), thresh5)   # denoising single with adaptive threshold 2

# kernel_3x3 = np.ones((3, 3), np.float32) / 9
# # We apply the filter and display the image
# blurred = cv2.filter2D(thresh3, -1, kernel_3x3)

kernel = np.array([[-1,-1,-1], [-1, 9,-1], [-1,-1,-1]])
thresh6 = cv2.filter2D(thresh3, -1, kernel)
cv2.imwrite(filename + str('-thresholded6.jpg'), thresh6)   # sharpening single

kernel2 = np.array([[1, 1, 1], [1, -7, 1], [1, 1, 1]])
thresh7 = cv2.filter2D(thresh3, -1, kernel2)
cv2.imwrite(filename + str('-thresholded7.jpg'), thresh7)   # sharpening single 2
