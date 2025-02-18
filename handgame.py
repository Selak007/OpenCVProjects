import cv2
import time
import handmodule as hm
import numpy as np
import math


pTime = 0
cap = cv2.VideoCapture(0)
detector = hm.handDetector()

from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)
#volume.GetMute()
#volume.GetMasterVolumeLevel()
volRange = volume.GetVolumeRange()

minVol = volRange[0]
maxVol = volRange[1]
while True:
    success, img = cap.read()  # Read frame from camera
    if not success:
        break

    # Process the image and find hands
    img = detector.findHands(img)
    lmList = detector.findPosition(img,draw=False)
    if len(lmList) > 4:
        x1,y1 = lmList[4][1],lmList[4][2]
        x2,y2 = lmList[8][1],lmList[8][2]
        cx,cy = (x1+x2)//2,(y1+y2)//2
        cv2.circle(img, (x1,y1), 15, (255,0,255), cv2.FILLED)
        cv2.circle(img, (x2,y2), 15, (255,0,255), cv2.FILLED)
        cv2.circle(img, (cx,cy), 15, (255,0,255), cv2.FILLED)
        cv2.line(img , (x1,y1),(x2,y2,),(255,0,255),3)

        length = math.hypot(x2 - x1,y2 - y1)

        vol = np.interp(length,[50,150],[minVol,maxVol])
        volBar = np.interp(length,[50,150],[400,150])
        volPer = np.interp(length,[50,300],[0,100])
        if length < 50:
            cv2.circle(img, (cx,cy), 15, (0,255,0), cv2.FILLED)
        volume.SetMasterVolumeLevel(vol, None)
        cv2.rectangle(img,(100,100),(185,400),(0,255,0),3)
        cv2.rectangle(img,(100,int(volBar)),(185,400),(0,255,0),cv2.FILLED)
        cv2.putText(img,f'{int(volPer)} %',(100,480),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,0),3)
    # Calculate and display FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (50, 70), cv2.FONT_HERSHEY_DUPLEX, 3, (255, 0, 255), 1)

    # Display the result
    cv2.imshow("Cam", img)

    # Break loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()