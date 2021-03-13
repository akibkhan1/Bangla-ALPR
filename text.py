from ocr import *
import glob

files = glob.glob('F:\Contests, Codes and Assignments\Pioneer Alpha\Main Project\yolov5\crop_results\*')
plates = []
for f in files:
    text = detectText(f)
    plates.append(text)

print(plates)