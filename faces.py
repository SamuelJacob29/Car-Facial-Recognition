import numpy as np
import cv2
import pickle

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
#eye_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_eye.xml')


recognizer = cv2.face.createLBPHFaceRecognizer()
recognizer.load("trainner.yml")

labels = {"person_name": 1}
with open("labels.pickle", "rb") as f:
	orig_labels = pickle.load(f)
	labels = {v:k for k,v in orig_labels.items()}

cap = cv2.VideoCapture(0)

while(True):
	# Capture frame-by-fram
	ret, frame = cap.read()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
	for(x, y, w, h) in faces:
 		#print(x,y,w,h)
 		#Region of interest Cuts off all the excess of the image and just shows a gray face
 		roi_gray = gray[y:y+h, x:x+w] #(ycord_start, ycord_end)
 		roi_color = frame[y:y+h, x:x+w]

 		#Recognize the Region of interest, Identify this person
 		id_, conf = recognizer.predict(roi_gray)

 		if conf >= 45 and conf <=85:
 			#print(id_)
 			#print(labels[id_] == 'samuel')
 			font = cv2.FONT_HERSHEY_SIMPLEX
 			name = labels[id_]
 			color = (255, 255, 255)
 			stroke = 2
 			cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)

		if labels[id_] == 'samuel':
			print("Start Car")
		elif labels[id_] != 'samuel':
			print("Dont Start Car")
 		img_item = "my_image.png"
 		cv2.imwrite(img_item, roi_color)

 		#Draws the rectangle
 		color = (0,0, 255) #BGR 0-255
 		stroke = 2
 		end_cord_x = x + w
 		end_cord_y = y + h
 		#Face Box
 		cv2.rectangle(frame, (x,y), (end_cord_x, end_cord_y), color, stroke)
 		
 		#Eye Box
 		#eyes = eye_cascade.detectMultiScale(roi_gray)
 		#for (ex, ey, ew, eh) in eyes:
 		#	cv2.rectangle(roi_color, (ex,ey),(ex + ew,ey + eh), color,  stroke)

	# Display the resulting frame
	cv2.imshow('frame',frame)
	if cv2.waitKey(20) & 0xFF == ord('q'):
		break
if labels[id_] == 'samuel':
	print("Start Car")
elif labels[id_] != 'samuel':
	print("Dont Start Car")
#When Everything is done, release the capture
cap.release()
cv2.destroyAllWindows()	