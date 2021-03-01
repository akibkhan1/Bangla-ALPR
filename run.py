import torch

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

dir = 'data/images/'
imgs = [dir+'zidane.jpg']

results = model(imgs)
results.save()