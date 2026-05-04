import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("emotion_model.h5")

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(0)

emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.2, 5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]

        # preprocess
        face = cv2.resize(face, (64, 64))
        face = face / 255.0
        face = face.reshape(1, 64, 64, 1)

        # prediction
        prediction = model.predict(face)
        emotion = emotion_labels[np.argmax(prediction)]

        print(emotion)  # debug

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(frame, emotion, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Emotion Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()