from flask import Flask, render_template,Response,request
import cv2
from saving_face_data import save_face_data
face_detect = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

# video_record()
def video_record():
    webcam = cv2.VideoCapture(0)
    while True:
        success, video = webcam.read()
        if not success:
            break
        else:
            grey_video = cv2.cvtColor(video,cv2.COLOR_BGR2GRAY)
            faces = face_detect.detectMultiScale(grey_video,1.2,5)
            for (x,y,w,h) in faces:
                cv2.rectangle(video,(x,y),(x+w,y+h),(255,255,0),2) 
            ret, buffer = cv2.imencode('.jpg', video)
            video = buffer.tobytes()
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + video + b'\r\n')  
            # concat frame one by one and show result
    webcam.release()

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/video_feed')
def video_display():
    return Response(video_record(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/read_face_template',methods = ['POST', 'GET'])
def read_face_template():
    if request.method == 'POST':
        global filename
        filename = request.form['name']
        print(filename)
        return render_template('read_face.html')

@app.route('/read_face')
def read_face():
    return Response(save_face_data(filename), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/prediction_template')
def prediction_template():
    return render_template('prediction.html')

@app.route('/prediction')
def prediction():
    from prediction import prediction
    return Response(prediction(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)