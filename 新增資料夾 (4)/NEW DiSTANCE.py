import cv2
import serial  # Allow us to Communicate with Arduino
import time
import numpy as np

x, y, h, w = 0, 0, 0, 0
DISTANCE = 0
# Distance constants
KNOWN_DISTANCE = 20.4  # INCHES
RIGHT_WIDTH = 2.5  # INCHES
LEFT_WIDTH = 3.0  # INCHES

# Object detector constant
CONFIDENCE_THRESHOLD = 0.4
NMS_THRESHOLD = 0.3

# colors for object detected
COLORS = [(255, 0, 0), (255, 0, 255), (0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]
GREEN = (0, 255, 0)
RED = (0, 0, 255)
BLACK = (0, 0, 0)
YELLOW = (0, 255, 255)
PERPEL = (255, 0, 255)
WHITE = (255, 255, 255)
# defining fonts
fonts = cv2.FONT_HERSHEY_COMPLEX
fonts2 = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
fonts3 = cv2.FONT_HERSHEY_COMPLEX_SMALL
fonts4 = cv2.FONT_HERSHEY_TRIPLEX

# getting class names from classes.txt file
class_names = []
with open(r"C:\Users\User\Documents\GitHub\darknet\build\darknet\x64\Yolov4-Detector-and-Distance-Estimator-master\classes.txt", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]
#  setttng up opencv net
yoloNet = cv2.dnn.readNet(r"C:\Users\User\Documents\GitHub\darknet\build\darknet\x64\backup\yolov4-tiny-obj_final.weights",  r"C:\Users\User\Documents\GitHub\darknet\build\darknet\x64\cfg\yolov4-tiny-obj.cfg")

yoloNet.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
yoloNet.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)

model = cv2.dnn_DetectionModel(yoloNet)
model.setInputParams(size=(416, 416), scale=1 / 255, swapRB=True)


# object detector funciton /method
def object_detector(image):
    classes, scores, boxes = model.detect(image, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    # creating empty list to add objects data
    data_list = []
    for (classid, score, box) in zip(classes, scores, boxes):
        # define color of each, object based on its class id
        color = COLORS[int(classid) % len(COLORS)]

        label = "%s : %f" % (class_names[classid[0]], score)

        # draw rectangle on and label on object
        cv2.rectangle(image, box, color, 2)
        cv2.putText(image, label, (box[0], box[1] - 14), fonts, 0.5, color, 2)

        # getting the data
        # 1: class name  2: object width in pixels, 3: position where have to draw text(distance)
        if classid == 0:  # person class id
            data_list.append([class_names[classid[0]], box[2], (box[0], box[1] - 2)])
        elif classid == 1:
            data_list.append([class_names[classid[0]], box[2], (box[0], box[1] - 2)])
        # if you want inclulde more classes then you have to simply add more [elif] statements here
        # returning list containing the object data.
    return data_list


def focal_length_finder(measured_distance, real_width, width_in_rf):
    focal_length = (width_in_rf * measured_distance) / real_width

    return focal_length


# distance finder function
def distance_finder(focal_length, real_object_width, width_in_frmae):
    distance = (real_object_width * focal_length) / width_in_frmae
    return distance


# reading the reference image from dir
ref_right = cv2.imread(r"C:\Users\User\Documents\GitHub\darknet\build\darknet\x64\scripts\VOCdevkit\VOC2020\TESTImages\270.jpg")
ref_left = cv2.imread(r"C:\Users\User\Documents\GitHub\darknet\build\darknet\x64\scripts\VOCdevkit\VOC2020\TESTImages\270.jpg")

left_data = object_detector(ref_left)
left_width_in_rf = left_data[1][1]

right_data = object_detector(ref_right)
right_width_in_rf = right_data[0][1]

print(f"Right width in pixels : {right_width_in_rf} left width in pixel: {left_width_in_rf}")

# finding focal length
focal_right = focal_length_finder(KNOWN_DISTANCE, RIGHT_WIDTH, right_width_in_rf)

focal_left = focal_length_finder(KNOWN_DISTANCE, LEFT_WIDTH, left_width_in_rf)
cap = cv2.VideoCapture(0)
# cv2.imshow("ref_image", ref_image)
# Setting up Arduino For Communication
Arduino = serial.Serial(baudrate=9600, port='COM3')
# variable for Arduino Communication
Direction = 0
# Max 0 and Min 255 Speed of Motors
Motor1_Speed = 0  # Speed of motor Accurding to PMW values in Arduino
Motor2_Speed = 0
Truing_Speed = 110
net_Speed = 180

while True:
    ret, frame = cap.read()
    frame_height, frame_width, _ = frame.shape
    # print(frame_height, frame_width)
    # calling face_data function
    # Distance_leve =0
    RightBound = frame_width - 140
    Left_Bound = 140

    data = object_detector(frame)
    for d in data:
        if d[0] == 'R feet':
            distance = distance_finder(focal_right, RIGHT_WIDTH, d[1])
            x, y = d[2]
        elif d[0] == 'L feet':
            distance = distance_finder(focal_left, LEFT_WIDTH, d[1])
            x, y = d[2]
        cv.line(frame, (50, 33), (130, 33), (BLACK), 15)
        cv.putText(frame, f"Robot State", (50, 35), fonts, 0.4, (YELLOW), 1)

            # Direction Decider Condition
            if x < Left_Bound:
                # Writing The motor Speed
                Motor1_Speed = Truing_Speed
                Motor2_Speed = Truing_Speed
                print("Left Movement")
                # Direction of movement
                Direction = 3
                cv2.rectangle(frame, (50, 65), (170, 65), (BLACK), 15)
                cv2.putText(frame, f"Move Left {x}", (50, 70), fonts, 0.4, (YELLOW), 1)

            elif x > RightBound:
                # Writing The motor Speed
                Motor1_Speed = Truing_Speed
                Motor2_Speed = Truing_Speed
                print("Right Movement")
                # Direction of movement
                Direction = 4
                cv2.rectangle(frame, (50, 65), (170, 65), (BLACK), 15)
                cv2.putText(frame, f"Move Right {x}", (50, 70), fonts, 0.4, (GREEN), 1)

                # cv2.line(frame, (50,65), (170, 65), (BLACK), 15)
                # cv2.putText(frame, f"Truing = False", (50,70), fonts,0.4, (WHITE),1)

            elif distance > 70 and distance <= 200:
                # Writing The motor Speed
                Motor1_Speed = net_Speed
                Motor2_Speed = net_Speed
                # Direction of movement
                Direction = 2
                cv2.rectangle(frame, (50, 55), (200, 55), (BLACK), 15)
                cv2.putText(frame, f"Forward Movement", (50, 58), fonts, 0.4, (PERPEL), 1)
                print("Move Forward")

            elif distance > 20 and distance <= 70:
                # Writing The motor Speed
                Motor1_Speed = net_Speed
                Motor2_Speed = net_Speed
                # Direction of movement
                Direction = 1
                print("Move Backward")
                cv2.rectangle(frame, (50, 55), (200, 55), (BLACK), 15)
                cv2.putText(frame, f"Backward Movement", (50, 58), fonts, 0.4, (PERPEL), 1)

            else:
                # Writing The motor Speed
                Motor1_Speed = 0
                Motor2_Speed = 0
                # Direction of movement
                Direction = 0
                cv2.rectangle(frame, (50, 55), (200, 55), (BLACK), 15)
                cv2.putText(frame, f"No Movement", (50, 58), fonts, 0.4, (PERPEL), 1)

            cv.putText(frame, f'Dis: {round(distance, 2)} inch', (x + 5, y + 13), fonts, 0.48, GREEN, 2)
            data = f"A{Motor1_Speed}B{Motor2_Speed}D{Direction}"  # A233B233D2
            print(data)
            # Sending data to Arduino
            Arduino.write(data.encode())  # Encoding the data into Byte fromat and then sending it to the arduino
            time.sleep(0.002)  # Providing time to Arduino to Receive data.
            Arduino.flushInput()  # Flushing out the Input.

    cv2.rectangle(frame, (Left_Bound, 80), (Left_Bound, 480 - 80), (YELLOW), 2)
    cv2.rectangle(frame, (RightBound, 80), (RightBound, 480 - 80), (YELLOW), 2)
    cv2.imshow("frame", frame)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
