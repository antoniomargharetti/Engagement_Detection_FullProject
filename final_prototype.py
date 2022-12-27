import os
import matplotlib.pyplot as plt
from keras import models
import cv2
import numpy as np
from PIL import Image

# loading face detection models
base_dir = os.path.dirname(__file__)
prototxt_path = os.path.join(base_dir, 'model_data\deploy.prototxt.txt')
caffemodel_path = os.path.join(base_dir, 'model_data\weights.caffemodel')

#loading engagement detection model
model = models.load_model("my_model13.h5")
model.load_weights("my_model_weights13.h5")

level = []
frame = []
facemodel = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)

video_capture = cv2.VideoCapture(0)

# font
font = cv2.FONT_HERSHEY_SIMPLEX

# org
org = (0, 400)

# fontScale
fontScale = 0.5

# Blue color in BGR
color = (0, 255, 0)

# Line thickness of 2 px
thickness = 2

ans = 1

startX, startY, endX, endY, count, value = 0, 0, 0, 0, 0, 0

# face detector
while True:
    value2 = 0
    count += 1
    ret, frames = video_capture.read()

    (h, w) = frames.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frames, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    facemodel.setInput(blob)
    detections = facemodel.forward()

    for i in range(0, detections.shape[2]):
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        confidence = detections[0, 0, i, 2]

        if confidence >= 0.5:
            cv2.rectangle(frames, (startX, startY), (endX, endY), (0, 255, 0), 2)

            face = frames[startY:endY, startX:endX]
            cv2.imwrite("image_path.jpg", face)
            img = Image.open("image_path.jpg")
            img = img.resize((48, 48))
            img = img.convert('L')
            img.save("image_path.jpg")

            face = cv2.imread("image_path.jpg")
            predictions = model.predict(face.reshape(1, 48, 48, 3))
            p = predictions[0].argmax(axis=0)
            value += p
            level.append(p)
            frame.append(count)
            break

        else:
            value += 0
            level.append(0)
            frame.append(count)
            break

    if count % 15 == 0:
        ans = value / 15
        p = 0
        value = 0

    frames = cv2.putText(frames, "Engagement Percentage " + str(ans * 50), (0,475), font, fontScale, color, thickness,
                         cv2.LINE_AA)
    frames = cv2.putText(frames, "Current Engagement Level " + str(p), (0, 440), font, fontScale, color, thickness,
                         cv2.LINE_AA)
    cv2.imshow('Video', frames)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

avg_engagement_level = sum(level)/count

print("Average Engagement Level for the session " + str(float("{:.2f}".format(avg_engagement_level))))

video_capture.release()
cv2.destroyAllWindows()

plt.plot(frame, level)
plt.show()
