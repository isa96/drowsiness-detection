#!/usr/bin/env python
import imutils
import dlib
from imutils.video import VideoStream
from imutils import face_utils
import time
import cv2
import numpy as np
from torch import classes
from EAR import eye_aspect_ratio
from MAR import mouth_aspect_ratio
import time
from imutils import face_utils
import dlib
import cv2
import tensorflow as tf
import playsound
import queue
from threading import Thread

#insialisasi dlib face predictor (HOG-based) lalu membuat facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(
    './dlib_shape_predictor/shape_predictor_68_face_landmarks.dat')

#insialisasi video stream dan sleep sebentar (2.0ms), memberi waktu untuk sensor camera untuk menyala/warm up
print("[INFO] initializing camera...")

vs = VideoStream(src=0).start()
# vs = VideoStream(usePiCamera=True).start() # Raspberry Pi
time.sleep(2.0)

# 400x225 to 1024x576
frame_width = 1024
frame_height = 576

sound_path = "alarm.wav"

# loop over the frames from the video stream 2D image points. If you change the image, you need to change vector
image_points = np.array([
    (359, 391),     # Nose tip 34
    (399, 561),     # Chin 9
    (337, 297),     # Left eye left corner 37
    (513, 301),     # Right eye right corne 46
    (345, 465),     # Left Mouth corner 49
    (453, 469)      # Right mouth corner 55
], dtype="double")

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

EYE_AR_CONSEC_FRAMES = 3  

COUNTER = 0
TOTAL = 0

COUNTER_DURATION = 0
TOTAL_DURATION= 0

COUNTER_MOUTH = 0
TOTAL_MOUTH=0
TOTAL_MOUTH_=0

TIME_COUNTER_EYE=0
COUNTER_MICROSLEEP = 0
TOTAL_MICROSLEEP = 0

TM_EAR_DURATION=0
TM_EAR=0
TM_MAR=0
TM_ALL=0
TM_MAR_=0
TM_STATE=0

# grab the indexes of the facial landmarks for the mouth
(mStart, mEnd) = (49, 68)

kalibrasi = True
sepuluh_pertama = []
eye_thres = 0

kalibrasi_mulut = True
seratus_pertama=[]
mouth_thres =0

#Moving Average
MA_EAR = []
MA_MAR = []

timeFrame = 0

TM_LOOP= time.time()
ALARM_ON = False
GAMMA = 1.5
threadStatusQ = queue.Queue()

invGamma = 1.0/GAMMA
table = np.array([((i / 255.0) ** invGamma) * 255 for i in range(0, 256)]).astype("uint8")

def gamma_correction(image):
    return cv2.LUT(image, table)

def histogram_equalization(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.equalizeHist(gray) 

def soundAlert(path, threadStatusQ):
    while True:
        if not threadStatusQ.empty():
            FINISHED = threadStatusQ.get()
            if FINISHED:
                break
        playsound.playsound(path)
while True:
    # grab the frame from the threaded video stream, resize it to have a maximum width of 400 pixels, and convert it to grayscale
    frame = vs.read()
    timeFrame = time.time()
    frame = imutils.resize(frame, width=1024, height=576)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    size = gray.shape

    # detect faces in the grayscale frame
    rects = detector(gray, 0)

    # check to see if a face was detected, and if so, draw the total number of faces on the frame
    if len(rects) > 0:
        text = "{} face(s) found".format(len(rects))
        cv2.putText(frame, text, (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # loop over the face detections
    for rect in rects:
        # compute the bounding box of the face and draw it on the  frame
        (bX, bY, bW, bH) = face_utils.rect_to_bb(rect)
        cv2.rectangle(frame, (bX, bY), (bX + bW, bY + bH), (0, 255, 0), 1)
        # determine the facial landmarks for the face region, then convert the facial landmark (x, y)-coordinates to a NumPy array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # extract the left and right eye coordinates, then use the coordinates to compute the eye aspect ratio for both eyes
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        # average the eye aspect ratio together for both eyes
        EAR = (leftEAR + rightEAR) / 2.0
        EAR_ = (leftEAR + rightEAR) / 2.0

        # compute the convex hull for the left and right eye, then visualize each of the eyes
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        mouth = shape[mStart:mEnd]

        mouthMAR = mouth_aspect_ratio(mouth)
        MAR = mouthMAR
        MAR_=mouthMAR

        # compute the convex hull for the mouth, then visualize the mouth
        mouthHull = cv2.convexHull(mouth)

        cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)
        cv2.putText(frame, "MAR: {:.2f}".format(MAR), (650, 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.putText(frame, "EAR: {:.2f}".format(EAR), (650, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        #set threshold EAR and MAR
        input = np.array ([[[EAR]]])
        if(len(sepuluh_pertama) < 10): sepuluh_pertama.append(EAR)
        # print("EAR", input)
        
        input_mouth= np.array ([[[MAR]]])
        if(len(seratus_pertama) < 100): seratus_pertama.append(MAR)
        # print("MAR", input_mouth)

        #### KALIBRASI MATA
        if kalibrasi and len(sepuluh_pertama) == 10:
            eye_thres=sum(sepuluh_pertama)/10
            kalibrasi=False

        #### KALIBRASI MULUT
        if kalibrasi_mulut and len(seratus_pertama) == 100:
            mouth_thres=max(seratus_pertama)
            kalibrasi_mulut=False

        # EAR classes, 1=drowssy, 0=alert
        MA_EAR.append(1 if EAR<eye_thres else 0)
        if len(MA_EAR) > 5:
            MA_EAR.pop(0)
        classes = 1 if sum(MA_EAR) / len(MA_EAR) > 0.5 else 0

        # MAR classes, 1=drowssy, 0=alert
        MA_MAR.append(1 if MAR>mouth_thres else 0)
        if len(MA_MAR) > 5:
            MA_MAR.pop(0)
        classes_mouth= 1 if sum(MA_MAR) / len(MA_MAR) > 0.5 else 0

          #YAWN STATE ML
        #predict with model
        input_yawnDet= np.array ([[[EAR_],[MAR_]]])

        interpreter = tf.lite.Interpreter(model_path="cnn_model.tflite")
        interpreter.allocate_tensors()

        # Get input and output tensors.
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Test the model on random input data.
        input_shape = input_details[0]['shape']
        input_data = np.array(input_yawnDet, dtype=np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_data)

        interpreter.invoke()

        # The function `get_ten\\\]]]]]]]]]]]]]\sor()` returns a copy of the tensor data.
        # Use `tensor()` in order to get a pointer to the tensor.
        prediction = interpreter.get_tensor(output_details[0]['index'])
        # print(prediction)

        classes_mouth_= 1 if prediction > 0.38 else 0

       # BLINK COUNT
        if EAR < eye_thres:
            COUNTER += 1
        else:
            if COUNTER > EYE_AR_CONSEC_FRAMES:
               TOTAL += 1
            COUNTER=0

        # BLINK_DURATION
        #  PERTAMA MENUTUP
        if classes == 1 and TM_EAR_DURATION == 0: 
            TM_EAR_DURATION = time.time()
        # BUKA MATA
        if classes == 0 and TM_EAR_DURATION> 0:
            if (time.time() - TM_EAR_DURATION) < 0.4:
                TOTAL_DURATION += 1
            else:
                pass
            TM_EAR_DURATION= 0

        # MICROSLEEP
        #  PERTAMA MENUTUP
        if classes == 1 and TM_EAR == 0: 
            TM_EAR = time.time()
        # BUKA MATA
        if classes == 0 and TM_EAR > 0:
            if (time.time() - TM_EAR) > 3:
                TOTAL_MICROSLEEP += 1
            else:
                pass
            TM_EAR = 0

        #YAWN COUNT MODEL
        if classes_mouth_ == 1 and TM_MAR_ == 0: 
            TM_MAR_ = time.time()
        # TUTUP MULUT
        if classes_mouth_ == 0 and TM_MAR_ > 0:
            if (time.time() - TM_MAR_) > 3:
                TOTAL_MOUTH_ += 1
            else:
                pass
            TM_MAR_ = 0
        
        #STATE
        if TOTAL_MOUTH_ > 0 or TOTAL_MICROSLEEP>0:
            state = 1
        else:
            state = 0


        cv2.putText(frame, "Class EAR: {:.2f}".format(classes), (650, 90), 
                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "Microsleep: {:.2f}".format(TOTAL_MICROSLEEP), (650, 110), 
                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "Pred yawn: {:.2f}".format(prediction[0][0]), (650, 140), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "Class yawn: {:.2f}".format(classes), (650, 170), 
                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "Driver State: {:.2f}".format(state), (650, 200), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        # cv2.putText(frame, "BlinkCount: {:.2f}".format(TOTAL), (650, 120), 
        #              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        # cv2.putText(frame, "Class MAR: {:.2f}".format(classes_mouth), (650, 150), 
        #              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        # cv2.putText(frame, "YawnCount: {:.2f}".format(TOTAL_MOUTH_), (650, 180), 
        #              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0
        #              , 255), 2)
        # cv2.putText(frame, "Microsleep: {:.2f}".format(TOTAL_MICROSLEEP), (650, 210), 
        #              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        # cv2.putText(frame, "Pred yawn: {:.2f}".format(prediction[0][0]), (650, 260), 
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        # cv2.putText(frame, "Class yawn: {:.2f}".format(classes), (650, 290), 
        #              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        # cv2.putText(frame, "Driver State: {:.2f}".format(state), (650, 400), 
        #               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
      
        k = cv2.waitKey(1) 
        if k == ord('r'):
                state_= 0
                drowsy_ = 0
                ALARM_ON = False
                threadStatusQ.put(not ALARM_ON)

        elif k == 27:
                break
        if state==1:
                cv2.putText(frame, "Mengantuk", (70, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                if not ALARM_ON:
                    ALARM_ON = True
                    threadStatusQ.put(not ALARM_ON)
                    thread = Thread(target=soundAlert, args=(sound_path, threadStatusQ,))
                    thread.setDaemon(True)
                    thread.start()

        else:
                ALARM_ON = False

        for (i, (x, kalibrasi)) in enumerate(shape):
            if i == 33:
                image_points[0] = np.array([x, kalibrasi], dtype='double')
                # write on frame in Green
                cv2.circle(frame, (x, kalibrasi), 1, (0, 255, 0), -1)
                cv2.putText(frame, str(i + 1), (x - 10, kalibrasi - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
            elif i == 8:
                # something to our key landmarks
                # save to our new key point lis
                image_points[1] = np.array([x, kalibrasi], dtype='double')
                # write on frame in Green
                cv2.circle(frame, (x, kalibrasi), 1, (0, 255, 0), -1)
                cv2.putText(frame, str(i + 1), (x - 10, kalibrasi - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
            elif i == 36:
                # something to our key landmarks
                # save to our new key point list
                # i.e. keypoints = [(i,(x,y))]
                image_points[2] = np.array([x, kalibrasi], dtype='double')
                # write on frame in Green
                cv2.circle(frame, (x, kalibrasi), 1, (0, 255, 0), -1)
                cv2.putText(frame, str(i + 1), (x - 10, kalibrasi - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
            elif i == 45:
                # something to our key landmarks
                # save to our new key point list
                # i.e. keypoints = [(i,(x,y))]
                image_points[3] = np.array([x, kalibrasi], dtype='double')
                # write on frame in Green
                cv2.circle(frame, (x, kalibrasi), 1, (0, 255, 0), -1)
                cv2.putText(frame, str(i + 1), (x - 10, kalibrasi - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
            elif i == 48:
                # something to our key landmarks
                # save to our new key point list
                # i.e. keypoints = [(i,(x,y))]
                image_points[4] = np.array([x, kalibrasi], dtype='double')
                # write on frame in Green
                cv2.circle(frame, (x, kalibrasi), 1, (0, 255, 0), -1)
                cv2.putText(frame, str(i + 1), (x - 10, kalibrasi - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
            elif i == 54:
                # something to our key landmarks
                # save to our new key point list
                # i.e. keypoints = [(i,(x,y))]
                image_points[5] = np.array([x, kalibrasi], dtype='double')
                # write on frame in Green
                cv2.circle(frame, (x, kalibrasi), 1, (0, 255, 0), -1)
                cv2.putText(frame, str(i + 1), (x - 10, kalibrasi - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
            else:
                # everything to all other landmarks
                # write on frame in Red
                cv2.circle(frame, (x, kalibrasi), 1, (0, 0, 255), -1)
                cv2.putText(frame, str(i + 1), (x - 10, kalibrasi - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
    # print(f'FPS:  {1.0 / (time.time() - timeFrame)} FPS')

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    #RESTART
    if time.time()-TM_LOOP > 60 :
            # COUNTER_MICROSLEEP += 1
            TM_LOOP = time.time()
            print("TIMER RESET")
            TOTAL=0
            TOTAL_MOUTH=0
            TOTAL_MICROSLEEP=0
            TOTAL_DURATION=0
            state = 0

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# print(image_points)

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()