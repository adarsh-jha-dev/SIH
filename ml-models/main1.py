import cv2
import pickle
from deepface import DeepFace as Deepface
faceCascade=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
cap=cv2.VideoCapture(1)
if not cap.isOpened():
    cap=cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcam.")
i=0
d={}
while i<=30:
    ret,frame=cap.read()
    result=Deepface.analyze(frame,actions=['emotion'])
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=faceCascade.detectMultiScale(gray,1.1,4)
    for x,y,w,h in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h), (0,255,0),2)
    font=cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame,result['dominant_emotion'],(10,10),font,1,(0,255,255),2)
    cv2.imshow("Emotion Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if result["dominant_emotion"] not in d:
        d[result["dominant_emotion"]]=1
    else:
        d[result["dominant_emotion"]]+=1
    i+=1
key_with_max_value = max(d, key=d.get)
print(key_with_max_value,"is the major emotion")
    
cap.release()
cv2.destroyAllWindows()
