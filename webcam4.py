import cv2
import numpy as np
import time
from image_processing2 import process_image

class VideoCamera(object):
    def __init__(self, filepath):
        self.imcap = cv2.VideoCapture(filepath)
        self.imcap.set(3, 480) # set width as 480
        self.imcap.set(4, 480) # set height as 480
        self.detected_frame = 1 # used for naming the cropped plate image
        self.prev_framecount = 1
        self.framecount = 1
        self.platecount = 0
        self.net = cv2.dnn.readNetFromTensorflow('frozen_inference_graph.pb', 'output.pbtxt')
        self.classNames = { 1: 'Number_plate' }
        self.avg_fps = []
        self.max_confidence_frame = 0.0
        self.max_confidence = 0

    def __del__(self):
        self.imcap.release()
        cv2.destroyAllWindows()

    def get_frame(self):
        
        # detector = cv2.CascadeClassifier('cascade_v1.xml')

        while True:
            start_time = time.time()
            ret, frame = self.imcap.read()
            # face = detector.detectMultiScale(image = frame, scaleFactor  = 1.1, minNeighbors = 10, minSize = (45, 45))
            if not frame is None:
                self.framecount += 1
                # for (x, y, w, h) in face:
                frame_resized = cv2.resize(frame,(300,300)) # resize frame for prediction
            
                blob = cv2.dnn.blobFromImage(frame_resized, size=(300, 300), swapRB=True, crop=False)

                self.net.setInput(blob)
                detections = self.net.forward()

                cols = frame_resized.shape[1] 
                rows = frame_resized.shape[0]

                for i in range(detections.shape[2]):
                    confidence = detections[0, 0, i, 2] #Confidence of prediction 
                    if confidence > 0.5: # Filter prediction 
                        class_id = int(detections[0, 0, i, 1]) # Class label

                        # Object location 
                        xLeftBottom = int(detections[0, 0, i, 3] * cols) 
                        yLeftBottom = int(detections[0, 0, i, 4] * rows)
                        xRightTop   = int(detections[0, 0, i, 5] * cols)
                        yRightTop   = int(detections[0, 0, i, 6] * rows)
                        
                        # Factor for scale to original size of frame
                        heightFactor = frame.shape[0]/300.0  
                        widthFactor = frame.shape[1]/300.0 
                        # Scale object detection to frame
                        xLeftBottom = int(widthFactor * xLeftBottom) 
                        yLeftBottom = int(heightFactor * yLeftBottom)
                        xRightTop   = int(widthFactor * xRightTop)
                        yRightTop   = int(heightFactor * yRightTop)

                        crop_img = frame[yLeftBottom:yRightTop, xLeftBottom:xRightTop]
                        filename = 'static/images/'+str(self.detected_frame)+'.jpg'
                        self.detected_frame += 1
                        # print(filename)

                        if confidence > self.max_confidence:
                            self.max_confidence = confidence
                            self.max_confidence_frame = np.copy(crop_img)

                        if self.framecount - self.prev_framecount >= 72:
                            cv2.imwrite(filename, self.max_confidence_frame)
                            self.max_confidence = 0
                            self.max_confidence_frame = 0
                            self.prev_framecount = self.framecount
                            self.platecount += 1
                            
                        # Draw location of object  
                        cv2.rectangle(frame, (xLeftBottom, yLeftBottom), (xRightTop, yRightTop), (0, 255, 0))

                        # Draw label and confidence of prediction in frame resized
                        if class_id in self.classNames:
                            label = self.classNames[class_id] + ": " + str(confidence)
                            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

                            yLeftBottom = max(yLeftBottom, labelSize[1])
                            cv2.rectangle(frame, (xLeftBottom, yLeftBottom - labelSize[1]),(xLeftBottom + labelSize[0],
                            yLeftBottom + baseLine), (255, 255, 255), cv2.FILLED)

                            cv2.putText(frame, label, (xLeftBottom, yLeftBottom), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

                            print(label) #print class and confidence

                ret, jpeg = cv2.imencode('.jpg', frame)
                data = []
                data.append(jpeg.tobytes())
            else:
                filename = 'static/images/'+str(self.detected_frame)+'.jpg'
                cv2.imwrite(filename, self.max_confidence_frame)
                # process_image(self.platecount)
                # self.avg_fps.append(1.0 / (time.time() - start_time))
                # print("FPS: ", 1.0 / (time.time() - start_time)) # FPS = 1 / time to process loop
                # print(f'Average FPS: {np.mean(np.array(self.avg_fps))}')
                break

            self.avg_fps.append(1.0 / (time.time() - start_time))
            # print("FPS: ", 1.0 / (time.time() - start_time)) # FPS = 1 / time to process loop

            return data
