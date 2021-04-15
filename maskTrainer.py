import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import AveragePooling2D, Flatten, Input, Dense, Dropout
from tensorflow.keras.models import Model
from imutils import paths
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import os
import numpy as np

#command: python maskTrainer.py --faces faces

# training parameters
lRate = 1e-4
epochNumber = 20
batches = 32

# obtain image path directories
iPaths = list(paths.list_images("faces"))
data, labels = [], []

# go over image paths
for iPath in iPaths:
	# extract the class label from the filename
	label = iPath.split(os.path.sep)[-2]

	# pre-process image
	img = load_img(iPath, target_size=(224, 224))
	img = img_to_array(img)
	img = preprocess_input(img)

	labels.append(label)
	data.append(img)

# numpy array conversion
data = np.array(data, dtype="float32")
labels = np.array(labels)


# binarize labels array
binarized = LabelBinarizer()
labels = binarized.fit_transform(labels)
labels = to_categorical(labels)

# initialize training and test variables and use 80% of data for training and 20% for testing
(trX, teX, trY, teY) = train_test_split(data, labels, test_size=0.20, stratify=labels, random_state=42)

# make the training image generator
datGen = ImageDataGenerator(rotation_range=20, zoom_range=0.15, width_shift_range=0.2, height_shift_range=0.2,
	shear_range=0.15, horizontal_flip=True, fill_mode="nearest")

# load network
base = MobileNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))

# head of model to be used
head = base.output
head = AveragePooling2D(pool_size=(7, 7))(head)
head = Flatten(name="flatten")(head)
head = Dense(128, activation="relu")(head)
head = Dropout(0.5)(head)
head = Dense(2, activation="softmax")(head)

# swap head and base models
model = Model(inputs=base.input, outputs=head)

# freeze base model layers
for l in base.layers:
	l.trainable = False


optim_izer = Adam(lr=lRate, decay= (lRate/epochNumber))
model.compile(loss="binary_crossentropy", optimizer=optim_izer, metrics=["accuracy"])

start_time = time.time()
# train the head of the network
networkHead = model.fit( datGen.flow(trX, trY, batch_size=batches), steps_per_epoch=(len(trX)//batches),
	validation_data=(teX, teY), validation_steps= (len(teX)//batches), epochs=epochNumber)

# make predictions on the testing set
index = model.predict(teX, batch_size=batches)

# image index labels
index = np.argmax(index, axis=1)

# display classification results
print(classification_report(teY.argmax(axis=1), index, target_names=binarized.classes_))
# save model
model.save("trained_mask.model", save_format="h5")
print("Total time: " + str((time.time() - start_time)/60.0) + " minutes.")

# obtain model performance plot
epochs = epochNumber
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, epochs), networkHead.history["loss"], label="train_loss")
plt.plot(np.arange(0, epochs), networkHead.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, epochs), networkHead.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, epochs), networkHead.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.legend(loc="center right")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.savefig("model_data.png")





