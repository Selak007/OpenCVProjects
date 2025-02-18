import cv2
import mediapipe as mp
import time

# Initialize VideoCapture
cap = cv2.VideoCapture(0)

# Initialize MediaPipe Hands
mpHands = mp.solutions.hands
hands = mpHands.Hands()  # You can add parameters like min_detection_confidence, min_tracking_confidence
mpDraw = mp.solutions.drawing_utils  # For drawing hand landmarks

pTime = 0
cTime = 0

while True:
    success, img = cap.read()  # Read frame from camera
    if not success:
        break

    # Convert the image to RGB for MediaPipe
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)  # Process the frame for hands

    # Check if any hands are detected
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            # Draw landmarks on the hand
            for id,lm in enumerate(handLms.landmark):
                print(id,lm)
                h,w,c = img.shape
                cx,cy = int(lm.x*w),int(lm.y*h)
                print(id,cx,cy)
                if id == 4:
                    cv2.circle(img,(cx,cy),25,(255,0,255),cv2.FILLED)
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img,str(int(fps)),(50,70),cv2.FONT_HERSHEY_DUPLEX,3,(255,0,255),1)
    # Display the result
    cv2.imshow("Cam", img)

    # Break loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
