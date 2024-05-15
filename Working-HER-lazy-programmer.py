import cv2 
from fer import FER
# let's capture the webcam 
cap = cv2.VideoCapture(0)
# Emotion Detector 
Detector  = FER(mtcnn=True)
#face detector
#face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while cap.isOpened():
    success, frame = cap.read()
    # flip to make the mirror image 
    #frame = cv2.flip(frame , 1)
    #frame = cv2.resize(frame , (606,350))
    #frame = cv2.rotate(frame , 0)
    # detected = Detector.detect_emotions(frame)
    # print(detected)
    #faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    #cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    emo , score= Detector.top_emotion(frame)
    if emo:
        cv2.putText(frame ,str(emo),(50,50),cv2.FONT_HERSHEY_COMPLEX,2,(0,0,255),2)
        cv2.imwrite(f"{emo}.jpg",frame)
    cv2.imshow('Emotion Detector', frame)
    # cv2.imshow("Detector",frame)
    if cv2.waitKey(1) == ord('q'):
        break 