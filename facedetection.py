import cv2
import os
import numpy as np
from PIL import Image

base_dir = os.path.dirname(__file__)
prototxt_path = os.path.join(base_dir, 'model_data\deploy.prototxt.txt')
caffemodel_path = os.path.join(base_dir ,'model_data\weights.caffemodel')

# Read the model
model = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)

dataset = os.listdir('Dataset/')

for ttv in dataset:
    users = os.listdir('Dataset/'+ttv+'/')
    for user in users:
        clips = os.listdir('Dataset/'+ttv+'/'+user+'/')
        for clip in clips:
            images = os.listdir('Dataset/'+ttv+'/'+user+'/'+clip+'/')
            os.remove('Dataset/'+ttv+'/'+user+'/'+clip+'/' + images[0])
            for temp in images[1:]:

                image_path = os.path.join('Dataset', ttv, user, clip, temp)
                image = cv2.imread(image_path)
                (h, w) = image.shape[:2]
                blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
                model.setInput(blob)
                detections = model.forward()

                for i in range(0, detections.shape[2]):
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                    confidence = detections[0, 0, i, 2]

                    if confidence >= 0.5:
                        frame = image[startY:endY, startX:endX]
                        # print("done " + temp)
                        cv2.imwrite(image_path, frame)
                        img = Image.open(image_path)
                        img = img.resize((48, 48))
                        img = img.convert('L')
                        img.save(image_path)
                        break
            print(clip + " Done")
        print("User number " + user + " Done")
        print("============================================================")


