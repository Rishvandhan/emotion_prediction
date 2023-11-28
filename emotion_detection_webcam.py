import pathlib
import cv2
from keras.models import load_model
import numpy as np

p=r'D:\python\workspace_machineLearning\facial emotion git\face_detect_model1.h5'
model=load_model(p)
cascade_path = pathlib.Path(cv2.__file__).parent.absolute()/ "data/haarcascade_frontalface_default.xml"
print(cascade_path)

clf = cv2.CascadeClassifier(str(cascade_path))

camera = cv2.VideoCapture(0)
b_box=[]
while True:
    _, frame= camera.read()
    #gray= cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = clf.detectMultiScale(
        frame,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30,30),
        flags = cv2.CASCADE_SCALE_IMAGE
    )
    #b_box.append(faces)
    print(faces)
    #print(b_box)
    for (x,y,width,height) in faces:
        cv2.rectangle(frame,(x, y),(x+width , y+height),(255,255,0),2)
        roi_gray_frame= frame[y:y +height, x:x + width]
        croped_img=np.expand_dims(np.expand_dims(cv2.resize(frame,(48,48)),-1),0)
        rgb_img= cv2.cvtColor(croped_img, cv2.COLOR_BGR2RGB)
        rgb_img=croped_img/255.0

        #predicting
        pred= model.predict(rgb_img.reshape(1,48,48,3))
        pred=pred.flatten()
        #print(pred)
        #pred=pred>0.5
        indices =pred.argmax()
        #print(indices)
        #emotion=(str(pred))
        if(indices ==0):
           cv2.putText(frame,'anger',(int(x),int(y)),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
        elif(indices ==1):
            cv2.putText(frame,'disgust',(int(x),int(y)),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
        elif(indices ==2):
            cv2.putText(frame,'fear',(int(x),int(y)),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
        elif(indices ==3):
            cv2.putText(frame,'happy',(int(x),int(y)),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
        elif(indices ==4):
            cv2.putText(frame,'neutral',(int(x),int(y)),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
        elif(indices ==5):
            cv2.putText(frame,'sad',(int(x),int(y)),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
        elif(indices ==6):
            cv2.putText(frame,'supprised',(int(x),int(y)),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)    

        #cv2.putText(frame,emotion,(int(x),int(y)),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)    

    
    cv2.imshow("faces",frame)
    #cv2.imshow("croped face",croped_img)
    if cv2.waitKey(1) == ord("q"):
        break


camera.release
cv2.destroyAllWindows