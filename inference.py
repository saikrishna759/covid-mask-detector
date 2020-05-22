import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img

print("loading.....")

model = load_model('masking.h5')
print("model loaded!")
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture(0)

print("started capturing")
label = ""
while True:
    ret , img = cap.read()
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, 1.1,20)
    for (x,y,w,h) in faces:
        img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),3)
        #labelSize=cv2.getTextSize(label,cv2.FONT_HERSHEY_COMPLEX,0.5,2)
        #roi_gray = gray[y:y+h, x:x+w]
        new_img = img[y:y+h, x:x+w]
        face = cv2.resize(new_img, (224, 224))
        face = img_to_array(face)
        face = preprocess_input(face)
        face = np.expand_dims(face, axis=0)

        #img =  np.array(img, dtype="float32")
        x1 = model.predict(face)
        if np.argmax(x1) == 0:
          label = "mask"
        else:
          label = "no mask"
        cv2.putText(img,label,(x-5,y-5),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),1)
    cv2.imshow('img',img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break



cap.release()
cv2.destroyAllWindows()
                      

