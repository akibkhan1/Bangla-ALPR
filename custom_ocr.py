import cv2
import numpy as np
import glob
from keras.models import load_model

classes = {
    0: '০',
    1: '১',
    2: '২',
    3: '৩',
    4: '৪',
    5: '৫',
    6: '৬',
    7: '৭',
    8: '৮',
    9: '৯',
    10: '-',
    11: 'ঢাকা',
    12: 'খ',
    13: 'মেট্রো',
    14: 'শ',
    15: 'এ',
    16: 'ব',
    17: 'ভ',
    18: 'চ',
    19: 'ছ',
    20: 'দ',
    21: 'ড',
    22: 'ঢ',
    23: 'ই',
    24: 'গ',
    25: 'ঘ',
    26: 'হ',
    27: 'য',
    28: 'জ',
    29: 'ঝ',
    30: 'ক',
    31: 'ল',
    32: 'ম',
    33: 'ন',
    34: 'অ',
    35: 'প',
    36: 'ফ',
    37: 'র',
    38: 'স',
    39: 'ট',
    40: 'ঠ',
    41: 'থ',
    42: 'ঙ',
    43: 'উ'
}

def sort_function(filename):
    filename = filename.split('\\')[1]
    return int(filename[:-4])

if __name__ == "__main__":

    model = load_model("numbers_character_batch100.h5")
    cropped_imgs = sorted(glob.glob('static/cropped_imgs/*.jpg'), key=sort_function)
    for img in cropped_imgs:
        img = cv2.imread(img)
        img = cv2.resize(img, (64, 64), interpolation = cv2.INTER_AREA)
        img = np.array(img)
        data = img.reshape(-1, 64, 64, 3)
        pre = np.argmax(model.predict(data)[0])
        print(classes[pre])