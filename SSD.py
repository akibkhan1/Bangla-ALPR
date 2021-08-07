import cv2
import numpy as np
import time
from datetime import datetime

class VideoCamera(object):
    def __init__(self, filepath):
        self.imcap = cv2.VideoCapture(filepath)
        self.imcap.set(3, 480) # set width as 480
        self.imcap.set(4, 480) # set height as 480
        self.framecount = 0
        self.prev_detected_frame = 0
        self.platecount = 0
        self.net = cv2.dnn.readNetFromTensorflow('weights/frozen_inference_graph.pb', 'configs/output.pbtxt')
        self.classNames = { 1: 'Number_plate' }
        self.avg_fps = []
        self.max_confidence = 0
        self.max_confidence_frames = {}

    def __del__(self):
        self.imcap.release()
        cv2.destroyAllWindows()

    def get_frame(self):
        
        # detector = cv2.CascadeClassifier('weights/cascade_v1.xml')

        while True:
            start_time = time.time()
            ret, frame = self.imcap.read()
            if not frame is None:
                self.framecount += 1

                frame_resized = cv2.resize(frame,(300, 300)) # resize frame for prediction
                blob = cv2.dnn.blobFromImage(frame_resized, size=(300, 300), swapRB=True, crop=False)
                self.net.setInput(blob)
                detections = self.net.forward()
                cols = frame_resized.shape[1] 
                rows = frame_resized.shape[0]

                for i in range(detections.shape[2]):
                    confidence = detections[0, 0, i, 2] #Confidence of prediction 
                    if confidence > 0.5: # Filter prediction 
                        self.prev_detected_frame = self.framecount

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

                        if confidence > self.max_confidence:
                            self.max_confidence = confidence
                        if int(confidence*100) not in self.max_confidence_frames.keys():
                            self.max_confidence_frames[int(confidence*100)] = [np.copy(crop_img)]
                        else:
                            self.max_confidence_frames[int(confidence*100)].append(np.copy(crop_img))
                            
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

                            # print(label) #print class and confidence

                if (self.framecount - self.prev_detected_frame >= 24) and (not len(self.max_confidence_frames) == 0):
                    self.platecount += 1
                    sorted_frames = dict(sorted(self.max_confidence_frames.items(), reverse=True))
                    i = 0
                    for row in sorted_frames.items():
                        for detection in row[1]:
                            if i == 3:
                                break
                            now = datetime.now()
                            current_time = now.strftime("%H-%M-%S")
                            filename = 'static/images/'+str(self.platecount)+'-'+str(i+1)+'-'+str(current_time)+'.jpg'
                            cv2.imwrite(filename, detection)
                            i += 1
                        if i == 3:
                            break
                    self.max_confidence_frames.clear()
                    self.max_confidence = 0

                ret, jpeg = cv2.imencode('.jpg', frame)
                data = []
                data.append(jpeg.tobytes())
            else:
                if not len(self.max_confidence_frames) == 0:
                    self.platecount += 1
                    sorted_frames = dict(sorted(self.max_confidence_frames.items(), reverse=True))
                    i = 0
                    for row in sorted_frames.items():
                        for detection in row[1]:
                            if i == 3:
                                break
                            now = datetime.now()
                            current_time = now.strftime("%H-%M-%S")
                            filename = 'static/images/'+str(self.platecount)+'-'+str(i+1)+'-'+str(current_time)+'.jpg'
                            cv2.imwrite(filename, detection)
                            i += 1
                        if i == 3:
                            break

                if start_time != time.time():
                    self.avg_fps.append(1.0 / (time.time() - start_time))
                # print("FPS: ", 1.0 / (time.time() - start_time)) # FPS = 1 / time to process loop
                print(f'Average FPS: {np.mean(np.array(self.avg_fps))}')
                break

            if start_time != time.time():
                self.avg_fps.append(1.0 / (time.time() - start_time))
            # print("FPS: ", 1.0 / (time.time() - start_time)) # FPS = 1 / time to process loop

            return data