from detect import *
import cv2
import pickle
import os
import glob
import shutil

class VideoCamera(object):
    def __init__(self, filepath):
        # self.imcap = cv2.VideoCapture('cctv_footage_single.mp4')
        # print(filepath)
        self.imcap = cv2.VideoCapture(filepath)
        self.imcap.set(3, 480) # set width as 480
        self.imcap.set(4, 480) # set height as 480
        self.framecount = 0
        self.model = load_model()

    def __del__(self):
        cv2.destroyAllWindows()

    def get_frame(self):
        
        # detector = cv2.CascadeClassifier('..\work_env\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml')

        while True:
            success, image = self.imcap.read()
            # image = cv2.flip(image, 1)
            # face = detector.detectMultiScale(image, 1.1, 7)
            # for (x,y,w,h) in face:
                # cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
                # if self.framecount < 10:
            if not image is None:
                cv2.imwrite('results/test'+str(self.framecount)+'.jpg', image)
            else:
                break

                # if self.framecount == 10:
                    # result = detect()

            if not len(os.listdir('static/images/'))>9:
                detect(self.model,self.framecount)
            
            # shutil.rmtree('results')
            # os.makedirs('results')
            os.remove('results/test'+str(self.framecount)+'.jpg')


            self.framecount+=1
            ret, jpeg = cv2.imencode('.jpg', image)
            data = []
            data.append(jpeg.tobytes())
            
            return data
