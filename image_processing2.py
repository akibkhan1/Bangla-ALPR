import numpy as np
import cv2
import glob
import os
from matplotlib import pyplot as plt

def sort_function(filename):
    filename = filename.split('\\')[1]
    return int(filename[:-4])

def process_image(platecount):
    # image_directory = 'static/images/*jpg'
    image = 'static/images/22.jpg'

    # files = sorted(glob.glob(image_directory), key=sort_function)

    # selected_images = files[:]

    # if len(selected_images) == 0:
    #     return

    # temporal_window_size = len(selected_images)
    # im_to_denoise_index = temporal_window_size//2

    # if temporal_window_size % 2 == 0:
    #     temporal_window_size -= 1

    # image_list = []

    # for image in selected_images:
    #     img = cv2.imread(image)
    #     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #     resized_img = cv2.resize(gray, (90, 60))
    #     image_list.append(resized_img)
    #     os.remove(image)

    # dst = cv2.fastNlMeansDenoisingMulti(image_list, im_to_denoise_index, temporal_window_size, None, 4, 7, 35)
    # adaptive gaussian thresholding
    dst = cv2.imread(image)
    gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    resized_img = cv2.resize(gray, (100, 200), interpolation=cv2.INTER_CUBIC)
    thresh = cv2.adaptiveThreshold(resized_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7, 2)
    thresh2 = cv2.adaptiveThreshold(resized_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7, 4)

    # filename = selected_images[im_to_denoise_index]
    # filename = filename[:-4]

    save_directory = "static/processed_images/"

    cv2.imwrite(save_directory + str(platecount) + str('-thresholded.jpg'), resized_img)    # denoising multi
    cv2.imwrite(save_directory + str(platecount) + str('-thresholded1.jpg'), thresh)    # denoising multi with adaptive threshold
    cv2.imwrite(save_directory + str(platecount) + str('-thresholded2.jpg'), thresh2)   # denoising multi with adaptive threshold 2

    kernel = np.array([[-1,-1,-1], [-1, 9,-1], [-1,-1,-1]])
    thresh6 = cv2.filter2D(resized_img, -1, kernel)
    cv2.imwrite(save_directory + str(platecount) + str('-thresholded3.jpg'), thresh6)   # sharpening single

    kernel2 = np.array([[1, 1, 1], [1, -7, 1], [1, 1, 1]])
    thresh7 = cv2.filter2D(resized_img, -1, kernel2)
    cv2.imwrite(save_directory + str(platecount) + str('-thresholded4.jpg'), thresh7)   # sharpening single 2

if __name__ == "__main__":
    process_image(1)