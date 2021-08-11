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
        self.model, self.classes, self.output_layers = self.load_yolo()
        self.avg_fps = []
        self.max_confidence = 0
        self.max_confidence_frames = {}

    def load_yolo(self):
        net = cv2.dnn.readNet("weights/best-yolov3-tiny-2.weights", "configs/yolov3-tiny3-1cls.cfg")
        classes = []
        # with open("coco.names", "r") as f:
        with open("configs/lp.names", "r") as f:
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
                    self.prev_detected_frame = self.framecount
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
                label = str(classes[class_ids[i]])
                color = (255, 255, 0)
                crop_img = img[y:y+h, x:x+w]
                confidence = confs[i]
                if confidence > self.max_confidence:
                    self.max_confidence = confidence
                    if int(confidence*100) not in self.max_confidence_frames.keys():
                        self.max_confidence_frames[int(confidence*100)] = [np.copy(crop_img)]
                    else:
                        self.max_confidence_frames[int(confidence*100)].append(np.copy(crop_img))
                cv2.rectangle(img, (x,y), (x+w, y+h), color, 1)
                cv2.putText(img, label, (x, y - 5), font, 1, color, 1)

    def get_frame(self):
        
        detector = cv2.CascadeClassifier('weights/cascade_v1.xml')

        while True:
            start_time = time.time()
            _, frame = self.imcap.read()
            
            if not frame is None:
                self.framecount += 1
                face = detector.detectMultiScale(image = frame, scaleFactor  = 1.1, minNeighbors = 10, minSize = (45, 45))
                
                for (x, y, w, h) in face:
                    height, width, channels = frame.shape
                    blob, outputs = self.detect_objects(frame, self.model, self.output_layers)
                    boxes, confs, class_ids = self.get_box_dimensions(outputs, height, width)
                    self.draw_labels(boxes, confs, class_ids, self.classes, frame)
                    # cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,0),1)
                
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

        self.imcap.release()

    def __del__(self):
        self.imcap.release()
        cv2.destroyAllWindows()
