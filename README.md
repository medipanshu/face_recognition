### Face Recognition

The face recognition ML project with Flask and Python is a web application that uses machine learning algorithms to detect and recognize faces.
In this project, we will be using various machine learning libraries such as OpenCV, OS, and NumPy to train and test your models. We will also be using Flask to create a user-friendly interface that allows users to upload images through webcam, and the application will use the trained models to detect and recognize faces in the uploaded images.

### Software and tools requirements 

1. [Github Account](https://github.com)
2. [VSCodeIDE](https://code.visualstudio.com/)
3. [GitCLI](https://git-scm.com/downloads)
4. [RenderAccount](https://render.com)

Create a new environment to run it on local machine

```
conda create -p diaenv python==3.7 -y
```
Activate the environment
```
conda activate diaenv/
```

### About files 

1. app.py => Application file.
2. model.py => a python script file for knn ml model.
3. prediction.py => a python script file to recognize the face.
4. saving_face_data.py => a python script file to store the images captured through webcam into .npy file.
5. haarcascade_frontalface_alt.xml =>  pre-trained classifier file that is used to detect human faces in images or video frames.

### About sub repositories

1. templates => contains index.html(html template for the project) 
2. data => directory where saving_face_data.py will save the images captured through webcam into .npy file.