from djitellopy import tello
import cv2
import cvzone
import face_recognition
import numpy as np

from cvzone.FaceDetectionModule import FaceDetector
from cvzone.PIDModule import PID
from cvzone.PlotModule import LivePlot

# detector = FaceDetector(minDetectionCon=0.5)
font = cv2.FONT_HERSHEY_DUPLEX
previous = "unknwon"

# cap = cv2.VideoCapture(0)
# _, img = cap.read()
hi, wi, = 480, 640
# print(hi, wi)
#                   P   I  D
xPID = PID([0.22, 0, 0.1], wi // 2)
yPID = PID([0.27, 0, 0.1], hi // 2, axis=1)
zPID = PID([0.005, 0, 0.003], 35000, limit=[-20,15])  # 12000

myPlotX = LivePlot(yLimit=[-100, 100], char='X')
myPlotY = LivePlot(yLimit=[-100, 100], char='Y')
myPlotZ = LivePlot(yLimit=[-100, 100], char='Z')

me = tello.Tello()
me.connect()
print(me.get_battery())
me.streamoff()
me.streamon()

# me.takeoff()
# me.move_up(40)

# Load a sample picture and learn how to recognize it.
Target_image = face_recognition.load_image_file("Image/Target.jpg")
Target_face_encoding = face_recognition.face_encodings(Target_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
    Target_face_encoding
]
known_face_names = [
    "Target"
]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
    # _, img = cap.read()
    img = me.get_frame_read().frame
    img = cv2.resize(img, (640, 480))

    # img, bboxs = detector.findFaces(img, draw=True)
    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(img)
        face_encodings = face_recognition.face_encodings(img, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
            name = "Unknown"

            # # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                # print(best_match_index)
                name = known_face_names[best_match_index]

            face_names.append(name)
            print(face_names)

    process_this_frame = not process_this_frame

    xVal = 0
    yVal = 0
    zVal = 0

    for (top, right, bottom, left), name in zip(face_locations, face_names):
        if name == "Unknown":
            continue
        results = face_recognition.face_landmarks(img)

        if len(results) != 0:
            # Face Center
            cx = left + ((right - left) // 2)
            cy = top + ((bottom - top) // 2)

            # Face Area Size
            area = (right - left) * (bottom - top)

            cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.rectangle(img, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
            # font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(img, name, (left + 6, bottom - 6), font, 1.0, (255, 0, 0), 1)

            xVal = int(xPID.update(cx))
            yVal = int(yPID.update(cy))
            zVal = int(zPID.update(area))

            print(xVal, yVal, zVal)
            print('area :', area)
            print('zVal', zVal)

            imgPlotX = myPlotX.update(xVal)
            imgPlotY = myPlotY.update(yVal)
            imgPlotZ = myPlotZ.update(zVal)

            img = xPID.draw(img, [cx, cy])
            img = yPID.draw(img, [cx, cy])
            img = zPID.draw(img, [cx, cy])

            imgStacked = cvzone.stackImages([img, imgPlotX, imgPlotY, imgPlotZ], 2, 0.75)
            # imgStacked = cvzone.stackImages([img], 1, 0.75)
            # Display Area
            # cv2.putText(imgStacked, str(area), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        else :
            print("Face Not Found")
            imgStacked = cvzone.stackImages([img], 1, 0.75)
            # imgStacked = cvzone.stackImages([img, imgPlotX, imgPlotY, imgPlotZ], 2, 0.75)
            cv2.putText(imgStacked, "Face Not Recognized", (-10, 50), font, 1.0, (255, 0, 255), 1)

        # me.send_rc_control(0, -zVal, -yVal, xVal)  # (lr, fb, up, yaw)

        # me.send_rc_control(0, -zVal, 0, xVal)
        # me.send_rc_control(0, -zVal, 0, 0)

    # cv2.imshow("Face Recognition & Tracking", imgStacked)
    cv2.imshow("Face Recognition & Tracking", img)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        # me.land()
        break

cv2.destroyAllWindows()
