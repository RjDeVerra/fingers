from imutils.video import VideoStream
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import imutils
import cv2

print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
handNet = load_model("hand_detector.model")

while True:
	frame = vs.read()
	frame = imutils.resize(frame, width=400)

	faces = []
	face = frame
	face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
	face = cv2.resize(face, (224, 224))
	face = img_to_array(face)
	face = preprocess_input(face)

	# add the face and bounding boxes to their respective
	# lists
	faces.append(face)
	faces = np.array(faces, dtype="float32")
	preds = handNet.predict(faces, batch_size=32)

	if preds[0][0] > 0.9 or preds[0][1] > 0.9:
		im = frame
		imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
		ret, thresh = cv2.threshold(imgray, 127, 255, 0)
		contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		#contour = cv2.drawContours(im, contours[0], -1, (0,255,0), 3)
		print(len(contours))
		a, b, w, h = cv2.boundingRect(contours[0])
		cv2.rectangle(im, (a, b), (a + w, b + h), (0, 255, 0), 2)
	print(preds)


	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	#cv2.imwrite(filename='saved_img.jpg', img=frame)