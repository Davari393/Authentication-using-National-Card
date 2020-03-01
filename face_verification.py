from keras import backend as K
K.set_image_data_format('channels_first')
import numpy as np
import tensorflow as tf
import FaceToolKit as ftk
from scipy import misc
import os
import cv2
verification_threshhold = 1.188
image_tensor_size = 160


# Class instantiations
v = ftk.Verification()

# Pre-load model for Verification
v.load_model("./models/20180204-160909/")
v.initial_input_output_tensors()
def webcam():
   face_classifier = cv2.CascadeClassifier('Haarcascades/haarcascade_frontalface_default.xml')
   cap = cv2.VideoCapture(0)
   while True:
        ret, frame = cap.read()
        flipped = cv2.flip(frame, 1)
        cv2.imshow('Webcam', flipped)
        if cv2.waitKey(1) == 13: #13 is the Enter Key
            cv2.imshow('frame' , flipped)
            cv2.imwrite('frame.png' , flipped)
            cv2.waitKey(300)
            break
   cap.release()
   cv2.destroyAllWindows()
   image = cv2.imread('frame.png')
   gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
   faces = face_classifier.detectMultiScale(gray, 1.3, 5)
   for (x,y,w,h) in faces:
      cv2.rectangle(image, (x,y), (x+w,y+h), (127,0,255), 2)
      cv2.imshow('Face Detection', image)
      croped = image[y:y+h,x:x+w]
      croped = cv2.resize(croped, (160,160))
      cv2.imwrite('face.png' , croped)
      cv2.waitKey(0)
   cv2.destroyAllWindows()
   return 'face.png'
def crop_cart(image_path):
    face_classifier = cv2.CascadeClassifier('Haarcascades/haarcascade_frontalface_default.xml')
    image=cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 3)
    if faces is ():
        print("No faces found")

    for (x,y,w,h) in faces:
        cv2.rectangle(image, (x,y), (x+w,y+h), (205, 239, 55), 2)
        images_croped = image[y:y+h ,x:x+w]
        images_croped = cv2.resize(images_croped, (160,160))
        cv2.imshow('Face Detection', image)
        cv2.imwrite('original_face.png' , images_croped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return 'original_face.png'


def verify(image_path):
    img1 = misc.imread(crop_cart(image_path))
    img2 = misc.imread(webcam())
    emb1 = v.img_to_encoding(img1, image_tensor_size)
    emb2 = v.img_to_encoding(img2, image_tensor_size)

    diff = np.subtract(emb1, emb2)
    dist = np.sum(np.square(diff))
    is_same = dist < 0.7
    print("distance img1 and img2 =", dist, " is+same =", is_same)
