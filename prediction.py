from model import knn
import cv2
import os
import numpy as np

face_data = []
names = []

for fx in os.listdir('./data/'):
    if fx.endswith('.npy'):
        data_item = np.load('./data/'+fx)
        names.append(fx[:-4])
        face_data.append(data_item)

face_data_with_label = []
name_with_labels = {}
label = 0

for i in face_data:
    y_label = label*np.ones(i.shape[0],dtype=int)
    y_label = y_label.reshape(-1,1)
    face_data_with_label.append(np.concatenate((i,y_label),axis=1))
    
    # mapping between y_label and name so if it predict output = 0/1/2.. it will map it with the name
    name_with_labels[label] = names[label]
    label+=1

face_data_with_label = np.concatenate(face_data_with_label,axis=0)
train_set = face_data_with_label.reshape(-1,face_data[0].shape[1]+1)

face_detect = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
webcam = cv2.VideoCapture(0)

def prediction():
    while True:
        ret,video = webcam.read()
        if ret == False:
            continue
        else:    
            faces = face_detect.detectMultiScale(video,1.3,5)
            
            faces = sorted(faces, key = lambda f:f[2]*f[3] , reverse=True)
            
            for (x,y,w,h) in faces:
                
                padding = 10
                try:
                    face_area = video[y-padding:y+h+padding,x-padding:x+w+padding]
                    face_area = cv2.resize(face_area,(100,100))
                    out = knn(train_set,face_area.flatten())
                
                    predicted_name = name_with_labels[int(out)]
                    cv2.putText(video,predicted_name,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),2,cv2.LINE_AA)
                    cv2.rectangle(video,(x,y),(x+w,y+h),(0,255,255))
                except:
                    print('Please come in the center of the frame')
            ret, buffer = cv2.imencode('.jpg', video)
            video = buffer.tobytes()
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + video + b'\r\n')
            
            