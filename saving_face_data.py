import numpy as np
import os
import cv2

def save_face_data(filename):
    counter = 0
    face_data = []
    face_detect = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
    webcam = cv2.VideoCapture(0)
    while True:
        ret,video = webcam.read()
        if ret == False:
            continue
        gray_video = cv2.cvtColor(video,cv2.COLOR_BGR2GRAY)
        faces = face_detect.detectMultiScale(gray_video,1.3,5)
        
        faces = sorted(faces, key = lambda f:f[2]*f[3] , reverse=True)
        
        for (x,y,w,h) in faces:
            cv2.rectangle(video,(x,y),(x+w,y+h),(0,255,255))
            padding = 10
            face_area = video[y-padding:y+h+padding,x-padding:x+w+padding]
            face_area = cv2.resize(face_area,(100,100))
            
            counter+=1
            if counter%10 == 0:
                face_data.append(face_area)
        ret, buffer = cv2.imencode('.jpg', video)
        video = buffer.tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + video + b'\r\n')
        if len(face_data) == 10:                        
            break
    webcam.release()
    face_data = np.asarray(face_data)
    face_data = face_data.reshape(face_data.shape[0],-1)
    #filename = input('Enter the name of the person :')
    dataset_path = './data/'
    np.save(dataset_path + filename + '.npy', face_data)