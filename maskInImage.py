from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import cv2
import os

#commad: python maskInImage.py -i test/test1.jpg

# arguments
arg = argparse.ArgumentParser()
arg.add_argument("-i", "--image", required=True)
args = vars(arg.parse_args())

# import models
depds = [os.path.sep.join(["models", "deploy.prototxt"]), os.path.sep.join(["models", "res10_300x300_ssd_iter_140000.caffemodel"])]
network = cv2.dnn.readNet(depds[0], depds[1])

# import our trained face amsk detector
model = load_model("trained_mask.model")

# obtain dimensions of passed in image
outputImg = cv2.imread(args["image"])
(H, W) = outputImg.shape[:2]

# construct a blob outline from the image
# outline = cv2.dnn.blobFromImage(outputImg, 1.0, (299, 299), (104.0, 177.0, 123.0))
outline = cv2.dnn.blobFromImage(outputImg, 1.0, (299, 299), 120)

# pass blob outline through network
network.setInput(outline)
detections = network.forward()


# loop over the detections
for i in range(detections.shape[2]):
	# obtain detected face confidence
	confidence = detections[0, 0, i, 2]

	# ensure reliability of faces with 50%
	if confidence > 0.5: 
		# get bounding box coordinates
		bBox = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
		(startX, startY, endX, endY) = bBox.astype("int")
		(startX, startY) = (max(0, startX), max(0, startY))
		(endX, endY) = (min(W - 1, endX), min(H - 1, endY))

		# pre-process image
		img = outputImg[startY:endY, startX:endX]
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		img = cv2.resize(img, (224, 224))
		img = img_to_array(img)
		img = preprocess_input(img)
		img = np.expand_dims(img, axis=0)

		# image passed and tested using our model
		(with_mask, without_mask) = model.predict(img)[0]

		# style output image
		lbl = "Mask" if with_mask > without_mask else "No Mask"
		colour = (0, 255, 0) if lbl == "Mask" else (40, 40, 255)

		# calculate the face mask confidence
		lbl = "{}: {}%".format(lbl, int(max(with_mask, without_mask) * 100))

		# draw a bounding box around the face and result text
		cv2.putText(outputImg, lbl, (startX, startY - 10), cv2.FONT_HERSHEY_COMPLEX, 0.5, colour, 2)
		cv2.rectangle(outputImg, (startX, startY), (endX, endY), colour, 2)

# display prediciton
cv2.imshow("Output", outputImg)
cv2.waitKey(0)




