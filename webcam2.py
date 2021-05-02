# from detect import *
# import cv2
# import pickle
# import os
# import glob
# import shutil

import cv2
import numpy as np
import time
from image_processing2 import process_image

class VideoCamera(object):
    def __init__(self, filepath):
        self.imcap = cv2.VideoCapture(filepath)
        self.imcap.set(3, 480) # set width as 480
        self.imcap.set(4, 480) # set height as 480
        self.detected_frame = 1
        self.prev_framecount = 1
        self.framecount = 1
        self.platecount = 0
        self.model, self.classes, self.output_layers = self.load_yolo()

    def load_yolo(self):
        # net = cv2.dnn.readNet("yolov3-tiny.weights", "yolov3-tiny.cfg")
        net = cv2.dnn.readNet("best-yolov3-tiny-2.weights", "yolov3-tiny3-1cls.cfg")
        classes = []
        # with open("coco.names", "r") as f:
        with open("lp.names", "r") as f:
            classes = [line.strip() for line in f.readlines()]

        layers_names = net.getLayerNames()
        output_layers = [layers_names[i[0]-1] for i in net.getUnconnectedOutLayers()]
        return net, classes, output_layers

    def detect_objects(self, img, net, outputLayers):			
        blob = cv2.dnn.blobFromImage(img, scalefactor=0.00392, size=(320, 320), mean=(0, 0, 0), swapRB=True, crop=False)
        net.setInput(blob)
        outputs = net.forward(outputLayers)
        return blob, outputs

    def get_box_dimensions(self, outputs, height, width):
        boxes = []
        confs = []
        class_ids = []
        for output in outputs:
            for detect in output:
                scores = detect[5:]
                class_id = np.argmax(scores)
                conf = scores[class_id]
                if conf > 0.1:
                    center_x = int(detect[0] * width)
                    center_y = int(detect[1] * height)
                    w = int(detect[2] * width)
                    h = int(detect[3] * height)
                    x = int(center_x - w/2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confs.append(float(conf))
                    class_ids.append(class_id)
        return boxes, confs, class_ids

    def draw_labels(self, boxes, confs, class_ids, classes, img): 
        indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.1, .4)
        font = cv2.FONT_HERSHEY_PLAIN
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                # print(boxes[i])
                label = str(classes[class_ids[i]])
                color = (255, 255, 0)
                crop_img = img[y:y+h, x:x+w]
                filename = 'static/images/'+str(self.detected_frame)+'.jpg'
                self.detected_frame += 1
                # print(filename)
                if self.framecount - self.prev_framecount >= 48 or self.platecount == 0:
                    self.prev_framecount = self.framecount
                    process_image(self.platecount)
                    self.platecount += 1
                    # print(f"Plate number {self.platecount}")
                cv2.imwrite(filename, crop_img)
                cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
                cv2.putText(img, label, (x, y - 5), font, 1, color, 1)

    def get_frame(self):
        
        detector = cv2.CascadeClassifier('cascade_v1.xml')

        while True:
            _, frame = self.imcap.read()
            face = detector.detectMultiScale(image = frame, scaleFactor  = 1.1, minNeighbors = 10, minSize = (45, 45))
            self.framecount += 1
            for (x, y, w, h) in face:
                height, width, channels = frame.shape
                blob, outputs = self.detect_objects(frame, self.model, self.output_layers)
                boxes, confs, class_ids = self.get_box_dimensions(outputs, height, width)
                self.draw_labels(boxes, confs, class_ids, self.classes, frame)
                cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,0),2)
            
            if not frame is None:
                ret, jpeg = cv2.imencode('.jpg', frame)
                data = []
                data.append(jpeg.tobytes())
            else:
                process_image(self.platecount)
                break
            
            return data

        self.imcap.release()

    def __del__(self):
        self.imcap.release()
        cv2.destroyAllWindows()
