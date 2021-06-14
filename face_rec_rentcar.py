import face_recognition
import cv2
import numpy as np
import logging
import json
import os  # To get path from window 10 | ubuntu
import mysql.connector
import datetime
import time

images = []  # Array for collect images
classNames = []  # Array for collect image name
encodeListKnown = []

cap = cv2.VideoCapture(0)

"""
    To-do List:
    - 'mylist' to 'list.length'
    - 'nameImg' is image name. 
"""
"""
    Database: ofsmsdb
    Table: rent_car_form
    Att: id_user, name_user
"""
# Connect Database Server
mydb = mysql.connector.connect(
    host = "localhost",
    user = "root",
    database = "ofsmsdb"
)
my_cursor = mydb.cursor()

with open("face.json") as f:
    s = f.read()
    data = json.loads(s)

# data = json.loads(s)

print(len(data)) # output: 2

for x in range(len(data)):
    nameImg = data[x]['id']
    classNames.append(nameImg)

print(classNames)

for x in range(len(data)):
    encode = data[x]['encoding']
    encodeListKnown.append(encode)

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

# Get name and create form
def insert_form (id_user):
    insert_prefix = "INSERT INTO rent_car_form (id, id_user, id_car, timestamp, cf) VALUES (NULL," + id_user +", NULL, current_timestamp(), 0)"
    my_cursor.execute(insert_prefix)
    mydb.commit()
    print(my_cursor.rowcount, "record inserted")

# Read Frame
while True:
    success, img = cap.read()
    # Reduce size webcam image
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    # Convert webcam image from BGR to RGB
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    # Find face location
    faceCurFrame = face_recognition.face_locations(imgS)
    # Encodes imgs with faceCurFrame
    encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)

    for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace, tolerance=0.5)
        name = "Unknown"
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)

        # Get index of minimum value in faceDis[]
        matchIndex = np.argmin(faceDis)

        # IF match
        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            y1, x2, y2, x1 = faceLoc
            # Because we resize
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2-35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1+6, y2-6),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            generate_form(name)
            sys.exit()

    process_this_frame = not process_this_frame

    # Display the resulting image
    cv2.imshow('Video', img)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# Release handle to the webcam
cv2.destroyAllWindows()
