import matplotlib.pyplot as plt
import tensorflow as tf
import pickle
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import Dense, Input, AveragePooling2D, Flatten, Dropout
from tensorflow.keras.models import Model
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from imutils import paths
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import (
    img_to_array,
    ImageDataGenerator,
    load_img,
)
import numpy as np
import os

root_path = "dataset"
data = []
labels = []
# Loading dataset
for image_path in paths.list_images(root_path):
    label = image_path.split(os.path.sep)[-2]
    img = load_img(image_path)
    image = img_to_array(img)
    data.append(image)
    labels.append(label)
data = np.array(data)
labels = np.array(labels)
print(data.shape)
print(labels[3009])
# Generating Label index
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)
print(labels)
output = open("label_encoder_cars.pickle", "wb")
pickle.dump(lb, output)
output.close()
# [0 1] => raccoon
# [1 0] => no_raccoon

# Train test split for data
trainx, testx, trainy, testy = train_test_split(data, labels, test_size=0.33)

aug = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    zoom_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest",
)

baseModel = MobileNetV2(
    input_tensor=Input(shape=(244, 244, 3)), include_top=False, weights="imagenet"
)

headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten()(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.2)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

model = Model(baseModel.input, headModel)

for layer in baseModel.layers:
    layer.trainable = True

model.compile(loss="binary_crossentropy", optimizer=Adam(lr=1e-4), metrics=["accuracy"])
r = model.fit(
    aug.flow(trainx, trainy, batch_size=32),
    steps_per_epoch=len(trainx) // 32,
    validation_data=(testx, testy),
    validation_steps=len(testx) // 32,
    epochs=5,
)

plt.plot(r.history["loss"], label="loss")
plt.plot(r.history["val_loss"], label="val_loss")
plt.show()

plt.plot(r.history["accuracy"], label="accuracy")
plt.plot(r.history["val_accuracy"], label="val_accuracy")
plt.show()
print(model.summary())
model.save("model/licence-plate_v2.h5")

