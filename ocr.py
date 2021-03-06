import os, io
from google.cloud import vision_v1
from google.cloud.vision_v1 import types 

os.environ['GOOGLE_APPLICATION_CREDENTIALS']=r'visionAPI_key.json'

##Load functions from the vision library
client = vision_v1.ImageAnnotatorClient()

##Perform OCR
def detectText(img):
    with io.open(img,'rb') as image_file:
        content = image_file.read()

    image = vision_v1.types.Image(content=content)

    response = client.document_text_detection(image=image)
    texts = response.text_annotations
    output=texts[0].description

    if response.error.message:
        raise Exception(
             '{}\nFor more info on error messages, check: '
             'https://cloud.google.com/apis/design/errors'.format(
               response.error.message))

    return output
