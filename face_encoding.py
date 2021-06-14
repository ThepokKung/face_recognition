import face_recognition
import cv2
import numpy as np
import logging
import json
import os  # To get path from window 10 | ubuntu

path = 'TestImg_po'

images = []  # Array for collect images
classNames = []  # Array for collect image name
myList = os.listdir(path)  # The array collect list in path
print(myList)  # output: ['biden.jpg', 'chayut.jpg', 'obama.jpg']

# for cl in myList:
#     curImg = cv2.imread(f'{path}/{cl}')
#     images.append(curImg)
#     nameImg = os.path.splitext(cl)[0]
#     print(nameImg)
#     classNames.append(nameImg)


def write_json(data, filename="face.json"):
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)


with open("face.json") as json_file:
    data = json.load(json_file)

    for cl in myList:
        curImg = cv2.imread(f'{path}/{cl}')
        img = cv2.cvtColor(curImg, cv2.COLOR_BGR2RGB)
        nameImg = os.path.splitext(cl)[0]
        print(nameImg)        
        face_enc = face_recognition.face_encodings(img)

        if face_enc:
            encode = face_enc[0]
            encode_toList = encode.tolist()
            print(nameImg)
            print(type(encode))
            print ("-----------------------------------")
            y = {"id": nameImg, "encoding": encode_toList}
            data.append(y)
    write_json(data)
