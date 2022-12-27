from keras import models
import os
import cv2


model = models.load_model("my_model8.h5")
model.load_weights("my_model_weights8.h5")
model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

count = 0
value = 0

images = os.listdir("temp_data")
for i in images:
    value = 0
    count+=1
    img_path = os.path.join("temp_data", i)
    image = cv2.imread(img_path)
    predictions = model.predict(image.reshape(1, 48, 48, 3))
    value = (predictions[0][0]*0 + predictions[0][1]*1 + predictions[0][2]*2)*50
    # if count%15 == 0:
    #     print(round(value/15))
    #     value = 0
print(value)
print("It loaded")