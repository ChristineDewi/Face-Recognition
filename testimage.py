"""
Created on Fri Apr 19 17:27:50 2019
@author: Christine
"""
from imutils import face_utils
from imutils.face_utils import FaceAligner
import imutils
import time
import cv2
import dlib
import os
import numpy as np

# Saving and loading model and weights
from tensorflow.python.keras.models import model_from_json
from tensorflow.python.keras.models import load_model

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
fa = FaceAligner(predictor, desiredFaceWidth=64)

people = [people for people in os.listdir("D:/PHD CYUT CHRISTINE 2018 - 2021/SEMESTER 2/7394 Artificial Intelligence and Machine Learning/Face2/output_image/")]
# loop over the frames from the video stream
text_list = ['image1.jpg', 'image2.jpg', 'image3.jpg', 'image4.jpg', 'image5.jpg', 'image6.jpg','image7.jpg']
for i in text_list:
    while True:
        key = cv2.waitKey(1) & 0xFF
         # if the `e` key was pressed, break from the loop
        if key == ord("e"):
            break
        #frame = vs.read()
        #frame = imutils.resize(frame, width=800)
        #height, width = frame.shape[:2]
        img_data = cv2.imread(i)
        gray_frame = cv2.cvtColor(img_data, cv2.COLOR_BGR2GRAY)
         # detect faces in the grayayscale frame
        rects = detector(gray_frame, 0)
    
        # loopop over the face detections COLOR_BGR2GRAY
        for rect in rects:
            faceAligned = fa.align(img_data, gray_frame, rect)
            faceAligned = cv2.cvtColor(faceAligned, cv2.COLOR_BGR2GRAY)
            faceAligned = np.array(faceAligned)
            faceAligned = faceAligned.astype('float32')
            faceAligned /= 255.0
            faceAligned= np.expand_dims([faceAligned], axis=3)
    
            Y_pred =  loaded_model.predict(faceAligned)
            for index, value in enumerate(Y_pred[0]):
              result = people[index] + ': ' + str(int(value * 100)) + '%'
              cv2.putText(img_data, result, (14, 15 * (index + 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                          (255,255,0), 1)
            # draw rect around face
            (x,y,w,h) = face_utils.rect_to_bb(rect)
            cv2.rectangle(img_data, (x, y), (x+w, y+h), (255,0,255), 2)
            # draw person name
            result = np.argmax(Y_pred, axis=1)
            cv2.putText(img_data, people[result[0]], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0,255,255), 2)
        # show the frame
        cv2.imshow("Frame", img_data)

# do a bit of cleanup
cv2.destroyAllWindows()
#vs.stop()
