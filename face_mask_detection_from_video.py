from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import cv2
import os
import  imutils
net = cv2.dnn.readNet('E:/Face Mask Detector/face_detector/deploy.prototxt', 'E:/Face Mask Detector/face_detector/res10_300x300_ssd_iter_140000.caffemodel')
face_mask_model = load_model('E:/Face Mask Detector/mask_detector.h5')
cap = cv2.VideoCapture(0)

while(True):
	ret, frame = cap.read()
	#gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	image = frame
	orig = image.copy()
	(h, w) = image.shape[:2]
	# construct a blob from the image
	blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))
	# pass the blob through the network and obtain the face detections
	print("[INFO] computing face detections...")
	net.setInput(blob)
	detections = net.forward()

	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the detection
		confidence = detections[0, 0, i, 2]
		# filter out weak detections by ensuring the confidence is
		# greater than the minimum confidence
		if confidence > 0.5:
			# compute the (x, y)-coordinates of the bounding box for
			# the object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
			# ensure the bounding boxes fall within the dimensions of
			# the frame
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# extract the face ROI, convert it from BGR to RGB channel
			# ordering, resize it to 224x224, and preprocess it
			face = image[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = np.expand_dims(face, axis=0)
			# pass the face through the model to determine if the face
			# has a mask or not
			result = face_mask_model.predict(face)
			print(result)
			if result[0] == 0:
				label = "Mask"
			else:
				label = "No Mask"
			color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
			# include the probability in the label
			#label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
			# display the label and bounding box rectangle on the output
			# frame
			cv2.putText(image, label, (startX, startY - 10),
						cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
			cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)
		# show the output image
	cv2.imshow('frame',frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()