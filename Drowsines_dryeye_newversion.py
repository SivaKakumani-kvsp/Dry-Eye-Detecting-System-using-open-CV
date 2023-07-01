# importing opencv package
import cv2
# for distance calculations
import numpy as np
#facial landmarks predicting package
import dlib
#for plotting the realtime graph
import cvzone
# to take control of image rotation, translation and adjustments
from imutils import face_utils
from cvzone.PlotModule import LivePlot
# to get the time calculation
import time

# for capturing the live video stream using opencv
video_capture = cv2.VideoCapture(0)

#for plotting the graph and its features and range on y-axis
plotY = LivePlot(560,360, [0,50], invert=True)

# to identify the both eyes landmarks from the pool of 68 facial landmarks
eyemarks = [36,37,38, 39, 40, 41, 42, 43, 44, 45, 46, 47]

# for detecting the facial landmarks using
face_detector = dlib.get_frontal_face_detector()
shape_identifier = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

#creating eye_blink function to calculate eye-aspect-ratio
def eye_blink(l,m,n,o,p,q):
    # finding distance between vertical landmarks of eye
    v1 = np.linalg.norm(m - o) 
    v2 = np.linalg.norm(n - p)
    v = v1+v2
    # finding the distance between horizontal landmarks of eye
    h = np.linalg.norm(l - q)
    # calculating the eye apsect ratio = ((p2-p4)+(p3-p5))/(p1-p6)*2
    #multiplying eye aspect ratio with 100 for better understanding
    Eye_ar = (v/(2.0*h))*100
     
    # creating the graphical plot for real time eye aspect ratio
    graph_plot = plotY.update(Eye_ar)
    cv2.imshow("Real Time Graph Viewer",graph_plot)

    # to find the blinking moments of ey
    if(Eye_ar > 24):
        return 2, Eye_ar
    elif(Eye_ar > 18 and Eye_ar <= 25):
        return 1, Eye_ar
    else:
        return 0, Eye_ar

#creating global variables
sleeping = 0
active = 0
drowsiness = 0
status = ""
color = (0, 0, 0)
blinkcounter = 0
counter = 0

# starting the time to identify the screen time
start_time = time.time()
while True:   
    
    #creating the window for display
    _, window = video_capture.read()
    gray = cv2.cvtColor(window, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray)
    window = window.copy()
    
    for face in faces:
        
        def zero(sleeping, drowsiness,active,status,color):
            sleeping += 1
            drowsiness = 0
            active = 0
            if(sleeping > 6):
                print("Alert....Sleepy")
                status = "SLEEPING DETECTED...."
                color = (255, 0, 0)
            return sleeping, drowsiness,active,status,color
                
        def one(sleeping, drowsiness,active,status,color):
            sleeping = 0
            active = 0
            drowsiness += 1
            if(drowsiness > 6):
                print("drowsiness....found")
                status = "DROWSY FACE...."
                color = (252, 5, 178)
            return sleeping, drowsiness,active,status,color
            
        def two(sleeping, drowsiness,active,status,color):
            drowsiness = 0
            sleeping = 0
            active += 1
            if(active > 6):
                print("Active...mood")
                status = "ACTIVE MOOD (: ... :)"
                color = (5, 252, 9)
            return sleeping, drowsiness,active,status,color
        
        landmarks = shape_identifier(gray, face)
        landmarks = face_utils.shape_to_np(landmarks)

        # giving the landmarks position of left eye
        left_blink, lfEye_ar = eye_blink(landmarks[36], landmarks[37],
                             landmarks[38], landmarks[41], landmarks[40], landmarks[39])

        if(left_blink == 0):
            sleeping, drowsiness,active,status,color=zero(sleeping, drowsiness,active,status,color)
            cv2.putText(window, status, (100, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        elif(left_blink == 1):
            sleeping, drowsiness,active,status,color=one(sleeping, drowsiness,active,status,color)
            cv2.putText(window, status, (100, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        else:
            sleeping, drowsiness,active,status,color=two(sleeping, drowsiness,active,status,color)
            cv2.putText(window, status, (100, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    
        #to display the eyelandmarks of left and right eye
        for n in eyemarks:
            (x, y) = landmarks[n]
            # we are assigning circle marks to the eye landmarks
            cv2.circle(window, (x, y), 1, (255, 255, 0), -1)
        
        #craeting another window for eye blink counter
        _, Blink_window = video_capture.read()
  
        if lfEye_ar < 20 and counter ==0 :
            blinkcounter +=1
            counter = 1
        if counter != 0:
            counter +=1
            if counter>10:
                counter = 0
        end_time = time.time()
        
        total_time = round((end_time - start_time),2)
        
        print(total_time)
        if int(total_time) < 10 and blinkcounter >= 5:
            print('Dry Eyes')
            cvzone.putTextRect(Blink_window, 'Dry Eyes Detected',[30,50])
        if int(total_time) == 10 and blinkcounter >= 5:
            print('Dry Eyes')
            
            blinkcounter = 0
        elif int(total_time) == 10:
            blinkcounter = 0
            start_time = time.time()
        
        cvzone.putTextRect(Blink_window, f'Blink Count: {blinkcounter}',[50,100])
        cv2.imshow("Blink Detecting Window", Blink_window)
        
    cv2.imshow("Face Detecting Window", window)
    key = cv2.waitKey(15)
    