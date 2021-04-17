import cv2
import time
import os
import imutils
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream


#command: python maskInVideo.py

def detection(fNetwork, mNetwork, cap):
	# compute frame dimension and rect
	(height, width) = cap.shape[:2]
	rect = cv2.dnn.blobFromImage(cap, 1.0, (300, 300), (104.0, 177.0, 123.0))
	# obtain detected faces
	fNetwork.setInput(rect)
	detections = fNetwork.forward()

	# get faces and their predicitons
	faces, coords, vals = [], [], []

	# loop detected faces
	for i in range(0, detections.shape[2]):
		# obtain cnfd
		cnfd = detections[0, 0, i, 2]

		if cnfd > 0.5:
			# obtain bounding box
			bound = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
			(sX, sY, eX, eY) = bound.astype("int")
			# keep bounding box within frame
			(sX, sY) = (max(0, sX), max(0, sY))
			(eX, eY) = (min(width - 1, eX), min(height - 1, eY))

			# extract region of interest and pre-process
			face = cap[sY:eY, sX:eX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)


			faces.append(face)
			coords.append((sX, sY, eX, eY))

# ensure at least one face was found in order to predict
	if len(faces) > 0:

		# make predictions on all faces at once for faster computation
		faces = np.array(faces, dtype="float32")
		vals = mNetwork.predict(faces, batch_size=32)

	# return each face and its coordinates inthe frame
	return (coords, vals)


# import models
depds = [os.path.sep.join(["models", "deploy.prototxt"]), os.path.sep.join(["models", "res10_300x300_ssd_iter_140000.caffemodel"])]
fNetwork = cv2.dnn.readNet(depds[0], depds[1])

# load our face mask model
mNetwork = load_model("trained_mask.model")

# open webcam
print("Starting webcam...")
capture = VideoStream(src=0).start()
# capture = VideoStream(src=1).start()
time.sleep(2.0)


# loop over each frame 
while True:
	# resize each frame for faster processing
	cap = capture.read()
	cap = imutils.resize(cap, width=400)
	# cap = cv2.flip(cap,1)

	# detect faces in frame
	(coords, vals) = detection(fNetwork, mNetwork, cap)


	for (bound, pred) in zip(coords, vals):
		# extract the predictions
		(sX, sY, eX, eY) = bound
		(mask, withoutMask) = pred

		# format the values displayed
		lbl = "Mask" if mask > withoutMask else "No Mask"
		colour = (0, 255, 0) if lbl == "Mask" else (0, 0, 255)
		# also show the confidence of the prediction
		lbl = "{}: {:1f}%".format(lbl, max(mask, withoutMask) * 100)
		# display prediction and label
		cv2.putText(cap, lbl, (sX, sY - 10), cv2.FONT_HERSHEY_COMPLEX, 0.45, colour, 2)
		cv2.rectangle(cap, (sX, sY), (eX, eY), colour, 2)


	# display frame
	cv2.imshow("Frame", cap)
	key = cv2.waitKey(1) & 0xFF

	# quit on q press
	if key == ord("q"):
		break

cv2.destroyAllWindows()
capture.stop()