import numpy as np
import cv2
import glob
import pytesseract 

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def sort_function(filename):
    filename = filename.split('\\')[1]
    # print(filename)
    return int(filename[0])

def recognize_plate(image):
    box = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    # gray = cv2.cvtColor(box, cv2.COLOR_RGB2GRAY)
    # resize image to three times as large as original for better readability
    gray = cv2.resize(box, None, fx = 8, fy = 8, interpolation = cv2.INTER_CUBIC)
    # perform gaussian blur to smoothen image
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    #cv2.imshow("Gray", gray)
    #cv2.waitKey(0)
    # threshold the image using Otsus method to preprocess for tesseract
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    # cv2.imshow("Otsu Threshold", thresh)
    # cv2.waitKey(0)
    # create rectangular kernel for dilation
    rect_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    # apply dilation to make regions more clear
    dilation = cv2.dilate(thresh, rect_kern, iterations = 1)
    # cv2.imshow("Dilation", dilation)
    # cv2.waitKey(0)
    # find contours of regions of interest within license plate
    try:
        contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    except:
        ret_img, contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # sort contours left-to-right
    sorted_contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])
    # print(len(sorted_contours))
    # create copy of gray image
    im2 = gray.copy()
    # create blank string to hold license plate number
    plate_num = ""
    count = 1
    # loop through contours and find individual letters and numbers in license plate
    for cnt in sorted_contours:
        x,y,w,h = cv2.boundingRect(cnt)
        height, width = im2.shape
        # if height of box is not tall enough relative to total height then skip
        # if height / float(h) > 6: continue

        # ratio = h / float(w)
        # # if height to width ratio is less than 1.5 skip
        # if ratio < 1.5: continue

        # if width is not wide enough relative to total width then skip
        if width / float(w) > 15: continue

        # area = h * w
        # # if area is less than 100 pixels skip
        # if area < 400: continue
        # print("Passed")
        # draw the rectangle
        rect = cv2.rectangle(im2, (x,y), (x+w, y+h), (0,255,0),2)
        # cv2.imshow('rect', rect)
        # cv2.waitKey(0)
        # grab character region of image
        roi = thresh[y-5:y+h+5, x-5:x+w+5]
        # perfrom bitwise not to flip image to black text on white background
        roi = cv2.bitwise_not(roi)
        # perform another blur on character region
        # print(cnt)
        # roi = cv2.medianBlur(roi, 5)
        crop_img = im2[y-5:y+h+5, x-5:x+w+5]
        filename = 'static/cropped_imgs/'+str(count)+'.jpg'
        cv2.imwrite(filename, crop_img)
        count += 1
        try:
            text = pytesseract.image_to_string(roi, lang='ben')
            # clean tesseract text by removing any unwanted blank spaces
            # clean_text = re.sub('[\W_]+', '', text)
            # plate_num += clean_text
            # print(text)
            plate_num += text
        except: 
            text = None
    if plate_num != None:
        print("License Plate #: ", plate_num)
    cv2.imshow("Character's Segmented", im2)
    cv2.waitKey(0)
    return plate_num

if __name__ == '__main__':

    image_directory = 'static/images/22.jpg'

    # files = sorted(glob.glob(image_directory), key=sort_function)
    # print(len(files))

    # for image in files:
    #     # text = pytesseract.image_to_string(image, lang='ben')
    #     # print(text)
    #     output = recognize_plate(image)
    output = recognize_plate(image_directory)
    print(output)