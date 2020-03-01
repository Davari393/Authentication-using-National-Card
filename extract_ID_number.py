from keras.models import load_model
import numpy as np
import cv2
from keras.preprocessing import image
import tensorflow as tf
from keras.applications.inception_v3 import preprocess_input
import imutils as im
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
model = load_model('National_Card.h5')


def decode(predict):
    p = np.argmax(predict)
    p=str(p)
    return p

def Find_contors(filename):
    image = cv2.imread(filename,3)
    image = cv2.resize(image, (1000, 630))
    image = image[155:220, 535:810]
    image=cv2.copyMakeBorder(image, 5, 5, 5, 5, cv2.BORDER_REPLICATE)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.copyMakeBorder(gray, 5, 5, 5, 5, cv2.BORDER_REPLICATE)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if im.is_cv2() else cnts[1]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]
    return (cnts,image,gray)
def roi_select(cnt):
    cnts,image,gray=cnt
    L = [] #list of rois
    images=[] #list of each diget
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        if (w >= 40) & (w < 64):
            w = int(w / 2)
            L.append((x, y, w, h))
            L.append((x + w, y, w, h))
        elif (w > 64) & (w < 80):
            w = int(w / 3)
            L.append((x, y, w, h))
            L.append((x + w, y, w, h))
            L.append((x + w + w, y, w, h))
        elif (w > 80):
            w = int(w / 4)
            L.append((x, y, w, h))
            L.append((x + w, y, w, h))
            L.append((x + w + w, y, w, h))
            L.append((x + w + w + w, y, w, h))
        elif (w < 35):
            L.append((x, y, w, h))
    if (len(L) > 10):
        extra = len(L) - 10
        for i in range(0, extra):
            L = sorted(L, key=lambda x: x[2])
            L.remove(L[0])
    L = sorted(L, key=lambda x: x[0])
    l = 1
    for t in L:
        x, y, w, h = t
        roi = gray[y :y + h + 4, x - 4:x + w + 4]
        roi=cv2.resize(roi,(25,25))
        images.append(roi)

        cv2.waitKey(0)
        l += 1
    return images

def predict_format(img_path):
    national_number=""
    cnt=Find_contors(img_path)
    digits=roi_select(cnt)
    for i in range(0,10):
        x = image.img_to_array(digits[i])
        x/=255
        arr3D = np.repeat(x, 3, axis=2)
        arr3D = np.expand_dims(arr3D, axis=0)
        y = model.predict(arr3D)
        p= decode(y)
        national_number+=p

    string = 'the national number is: ' + national_number
    print(string)
