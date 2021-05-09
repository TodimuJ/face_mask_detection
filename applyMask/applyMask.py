import dlib
import shutil
import json
import os
from imutils import face_utils, paths
import numpy as np
import time
import imutils
import sys
import cv2

# command: python applyMask.py 

# successfully applies face mask to human face on test image
def apply_mask(inputImage, outputImage):
	def channelConversion(front, back, channel):
		# convert types and normalize channel
		front = front.astype("float")
		back = back.astype("float")
		channel = channel.astype("float")/255

		# perform alpha blending
		front = cv2.multiply(channel, front)
		back = cv2.multiply(1 - channel, back)

		# overlay images
		output = cv2.add(front, back)
		
		# return the output image

		return output.astype("uint8")

	def loop(inputPath, outputPath, delay, finalDelay, lp): 
		# grab all image paths in the input directory
		imP = sorted(list(paths.list_images(inputPath)))
		# remove the last image path in the list
		lastPath = imP[-1]
		imP = imP[:-1]

		# make image
		file = "convert -delay {} {} -delay {} {} -loop {} {}".format(delay, " ".join(imP), finalDelay, lastPath, lp, outputPath)
		os.system(file)


	def applyImage(back, front, mask, points): 
		# print(front.shape)
		# print(mask.shape)
		# print(back.shape)

		# get coordinates to place the face mask
		(x, y) = points
		# obtain coordinates of face mask
		(H, W) = front.shape[:2]


		overlay = np.zeros(back.shape, dtype="uint8")
		overlay[y:y + H, x:x + W] = front

		# channel to control transparency
		channel = np.zeros(back.shape[:2], dtype="uint8")
		# print(back.shape)
		channel[y:y + H, x:x + W] = mask
		# print(alpha[y:y + H, x:x + W])
		channel = np.dstack([channel] * 3)
		# merge objects together
		output = channelConversion(overlay, back, channel)

		return output


	# declare face_mask and face_mask_mask
	maskF = cv2.imread("./face_mask_mask.png")
	maskB = cv2.imread("./face_mask.png")

	# create new temporary directory for storing images
	shutil.rmtree("temp", ignore_errors=True)
	os.makedirs("temp")


	# declare face detector data and  facial landmark predictor
	detector = cv2.dnn.readNetFromCaffe("../models/deploy.prototxt", "../models/res10_300x300_ssd_iter_140000.caffemodel")
	predictor = dlib.shape_predictor("../models/shape_predictor_68_face_landmarks.dat")

	# load the input image and construct an input blob from the image
	image = cv2.imread(inputImage)
	(H, W) = image.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
	
	# get detections from blob
	detector.setInput(blob)
	detections = detector.forward()

	# detect face in image
	i = np.argmax(detections[0, 0, :, 2])
	confidence = detections[0, 0, i, 2]
	# filter out weak detections
	if confidence < 0.5:
		print("Error: No faces within 50\% reliability detected")
		sys.exit(0)


	# compute x,y coordinates of face bounding box
	box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
	(startX, startY, endX, endY) = box.astype("int")

	# make rectangle over face bounding box
	rect = dlib.rectangle(int(startX), int(startY), int(endX), int(endY))
	# obtain facial landmarks for the face 
	shape = predictor(image, rect)
	shape = face_utils.shape_to_np(shape)

	# obtain landmark points for the nose cordinates
	(mStart, mEnd) = (32, 36)

	lMouthPts = shape[mStart:mEnd]
	rMouthPts = shape[mStart:mEnd]

	# obtainf the center of each side of the nose points
	lMouthCenter = lMouthPts.mean(axis=0).astype("int")
	rMouthCenter = rMouthPts.mean(axis=0).astype("int")


	# obtain angle between left and right side of the mouth
	dY = rMouthCenter[1] - lMouthCenter[1]
	dX = rMouthCenter[0] - lMouthCenter[0]
	theta = np.degrees(np.arctan2(dX, dY))

	# rotate face mask appropriately with obtained angle
	maskF = imutils.rotate_bound(maskF, theta)

	# resize face mask to cover the mouth and nose of face
	sW = int((endX - startX) * 1.1)
	maskF = imutils.resize(maskF, width=sW)

	# pre-process transparent background mask
	maskB = cv2.cvtColor(maskB, cv2.COLOR_BGR2GRAY)
	maskB = cv2.threshold(maskB, 0, 255, cv2.THRESH_BINARY)[1]
	maskB = imutils.rotate_bound(maskB, theta)
	maskB = imutils.resize(maskB, width=sW, inter=cv2.INTER_NEAREST)

	# apply face_mask to image over 2 steps starting from top
	steps = np.linspace(0, rMouthCenter[1], 2, dtype="int")

	for (i, y) in enumerate(steps):
		# periodically find the right position to place the mask
		if i!=0:
			shiftX = int(maskF.shape[1] * 0.5)
			shiftY = int(maskF.shape[0] * 0.35)

			y = max(0, y - shiftY)

			# add face mask to the image
			img1 = applyImage(image, maskF, maskB, (rMouthCenter[0] - shiftX, y))


			# output image to temp directory
			temp_img = os.path.sep.join(["temp", "{}.jpg".format(str(i).zfill(8))])
			cv2.imwrite(temp_img, img1)

	# create face-masked image
	loop("temp", outputImage, 5, 250, 0)
	# delete temporary directory
	shutil.rmtree("temp", ignore_errors=True)


i = 1
# inputPath = "../faces/with_mask/"
# outputPath = "../faces/with_mask/"
outputPath = "./"


images = ["test10.png", "test20.png", "test30.png", "test40.png", "test50.png"]

start_time = time.time()

for pic in images:
	# apply_mask(pic, "./config.json", outputPath + "test{}{}.png".format(str(i), str(1)))
	apply_mask(pic, outputPath + "test{}{}.png".format(str(i), str(1)))
	i += 1

# for pic in os.listdir(inputPath):
# 	if pic == ".DS_Store": #kept breaking the code so decided to skip it
# 		print(".DS_Store skipped")
# 		continue
# 	input_path = os.path.join(inputPath, pic)
# 	print(input_path)
# 	apply_mask(input_path, "./config.json", outputPath + "{}.jpg".format(str(i)))
# 	i += 1

print("Total time: " + str((time.time() - start_time)) + " seconds for " + str(i-1) + " pictures")
