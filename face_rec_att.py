import face_recognition
import cv2
import numpy as np
import logging
import json
import os  # To get path from window 10 | ubuntu
import sys
import mysql.connector
from datetime import datetime
from datetime import date 

images = []  # Array for collect images
classNames = []  # Array for collect image name
encodeListKnown = []

cap = cv2.VideoCapture(0)

"""
    To-do List:
    - 'mylist' to 'list.length'
    - 'nameImg' is image name. 
"""

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

# Connect Database Server
mydb = mysql.connector.connect(
    host = "localhost",
    user = "root",
    database = " face_recognition_attendance"
)
my_cursor = mydb.cursor()

# About Time
today = date.today()
now = datetime.now()
now_str = now.strftime("%H:%M:%S")

name = "name" # Pre-define

def get_name(id_user_att):
    # Get Name
    global name
    get_name_prefix = "SELECT full_name_eng FROM users WHERE id_user=" + id_user_att
    my_cursor.execute(get_name_prefix)
    myresult = my_cursor.fetchone()    
    name = str(myresult[0])
    return

# Insert check_in time
def insert_check_in(id_user_att):
    insert_check_in_prefix = "INSERT INTO attendance (id, id_user, date, check_in, check_out) VALUES (%s, %s, %s, %s, %s);"
    # (NULL, '6130400354', '2021-05-22', '14:39:23', NULL);
    val = (None, id_user_att, str(today), now_str, None)
    # Execution
    my_cursor.execute(insert_check_in_prefix, val)
    # myresult = my_cursor.fetchall()
    mydb.commit()

# Insert check_out time
def update_check_out (id_user_att):
    insert_check_out_prefix = "UPDATE attendance SET check_out=%s WHERE id_user=%s AND date=%s"
    val = (now_str, id_user_att, str(today))
    # Execution
    my_cursor.execute(insert_check_out_prefix, val)
    # myresult = my_cursor.fetchall()
    mydb.commit()

def check_in_today(id_user_att,today):
    # About Time
    # today = date.today()
    # print(str(today))
    check_in_today_prefix = "SELECT * FROM attendance WHERE id_user=%s AND date=%s"
    val = (id_user_att, str(today))
    my_cursor.execute(check_in_today_prefix, val)

    myresult = my_cursor.fetchall()

    # for row in myresult:
    #     print(row)

    if(str(myresult) == "[]"):
        print("insert_check_in process")
        insert_check_in(id_user_att)
    else:
        print("You already have checked in.")

def check_in_or_out(now,id_user_att,today):
    # Define check out time for function switching 
    now = datetime.now()
    print("check_in_or_out process") 
    check_out_time_per = now.replace(hour=16, minute=30, second=0, microsecond=0)
    print("check in today process")
    # check_in_today(id_user_att,today)
     
    if (now < check_out_time_per):
        print("check in today in progress")    
        check_in_today(id_user_att,today)
        print("check in today completed")         
    else:
       update_check_out (id_user_att)
       print("Check out is completed.") 


# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

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
            id_user_att = classNames[matchIndex].upper()
            get_name(id_user_att)
            y1, x2, y2, x1 = faceLoc
            # Because we resize
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2-35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1+6, y2-6),                        
                   cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)                    
            
            print(name)            
            check_in_or_out(now,id_user_att,today)
            print("-----------------------------------------")
             
            
           

    process_this_frame = not process_this_frame

    # Display the resulting image
    cv2.imshow('Video', img)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    

# Release handle to the webcam
cv2.destroyAllWindows()
