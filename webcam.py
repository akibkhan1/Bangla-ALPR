import cv2
import pickle

class VideoCamera(object):
    def __init__(self):
        self.imcap = cv2.VideoCapture(0)
        self.imcap.set(3, 640) # set width as 640
        self.imcap.set(4, 480) # set height as 480
        self.framecount = 0

    def __del__(self):
        self.imcap.stop()

    def get_frame(self):
        
        detector = cv2.CascadeClassifier('work_env\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml')
        
        while True:
            success, image = self.imcap.read()
            image = cv2.flip(image, 1)
            face = detector.detectMultiScale(image, 1.1, 7)
            for (x,y,w,h) in face:
                cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
                # cv2.imwrite('results/test'+str(self.framecount)+'.jpg', image)
                self.framecount+=1

            ret, jpeg = cv2.imencode('.jpg', image)
            data = []
            data.append(jpeg.tobytes())
            
            return data
