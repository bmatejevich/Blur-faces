# -*- coding: utf-8 -*-
"""
@author: brianmatejevich
"""

import cv2
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# For a video, give the path; for a webcam, this is usually 0 or 1
vid = cv2.VideoCapture(0)

# Initialize some variables
face_locations = []

while True:
    # Grab a single frame of video
    ret, frame = vid.read()

    # Resize frame of video to 1/4 size for faster face detection processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
    # Find all the faces and face encodings in the current frame of video
    face_locations = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Display the results
    for x,y,w,h in face_locations:
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        x *= 4
        y *= 4
        w *= 4
        h *= 4

        # Extract the region of the image that contains the face
        face_image = frame[y:y+h, x:x+w]
        cv2.imshow("Hidden Face", face_image)
        # Blur the face image
        face_image = cv2.GaussianBlur(face_image, (99, 99), 30)

        # Put the blurred face region back into the frame image
        frame[y:y+h, x:x+w] = face_image

    # Display the resulting image
    cv2.imshow('Press "q" to quit', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
vid.release()
cv2.destroyAllWindows()
