import cv2 as cv
import numpy as np
import urllib.request


# change to your ESP32-CAM ip

url = 'http://192.168.10.187/cam-lo.jpg'       #hi or lo
winName = 'CAM'
cv.namedWindow("winName")

while 1:
 imgResponse = urllib.request.urlopen(url)
 imgNp=np.array(bytearray(imgResponse.read()),dtype=np.uint8)
 img=cv.imdecode(imgNp, -1)
 cv.imshow("winName",img)
 tecla = cv.waitKey(5) & 0xFF
 if tecla == 27:
   break


cv.destroyALLWINDOWS()