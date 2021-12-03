import cv2 as cv
import numpy as np

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
BLACK = (0, 0, 0)
# defining fonts
FONTS = cv.FONT_HERSHEY_COMPLEX

# getting class names from classes.txt file
class_names = []
with open(r"C:\Users\User\Documents\GitHub\darknet\build\darknet\x64\Yolov4-Detector-and-Distance-Estimator-master\classes.txt", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]
#  setttng up opencv net
yoloNet = cv.dnn.readNet(r"C:\Users\User\Documents\GitHub\darknet\build\darknet\x64\backup\yolov4-tiny-obj_final.weights",  r"C:\Users\User\Documents\GitHub\darknet\build\darknet\x64\cfg\yolov4-tiny-obj.cfg")

yoloNet.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
yoloNet.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)

model = cv.dnn_DetectionModel(yoloNet)
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
        cv.rectangle(image, box, color, 2)
        cv.putText(image, label, (box[0], box[1] - 14), FONTS, 0.5, color, 2)

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
ref_right = cv.imread(r"C:\Users\User\Documents\GitHub\darknet\build\darknet\x64\scripts\VOCdevkit\VOC2020\TESTImages\270.jpg")
ref_left = cv.imread(r"C:\Users\User\Documents\GitHub\darknet\build\darknet\x64\scripts\VOCdevkit\VOC2020\TESTImages\270.jpg")

left_data = object_detector(ref_left)
left_width_in_rf = left_data[1][1]

right_data = object_detector(ref_right)
right_width_in_rf = right_data[0][1]

print(f"Right width in pixels : {right_width_in_rf} left width in pixel: {left_width_in_rf}")

# finding focal length
focal_right = focal_length_finder(KNOWN_DISTANCE, RIGHT_WIDTH, right_width_in_rf)

focal_left = focal_length_finder(KNOWN_DISTANCE, LEFT_WIDTH, left_width_in_rf)
cap = cv.VideoCapture(0)
while True:
    ret, frame = cap.read()

    data = object_detector(frame)
    for d in data:
        if d[0] == 'R feet':
            distance = distance_finder(focal_right, RIGHT_WIDTH, d[1])
            x, y = d[2]
        elif d[0] == 'L feet':
            distance = distance_finder(focal_left, LEFT_WIDTH, d[1])
            x, y = d[2]
        cv.rectangle(frame, (x, y - 3), (x + 150, y + 23), BLACK, -1)
        cv.putText(frame, f'Dis: {round(distance, 2)} inch', (x + 5, y + 13), FONTS, 0.48, GREEN, 2)

    cv.imshow('frame', frame)

    key = cv.waitKey(1)
    if key == ord('q'):
        break
cv.destroyAllWindows()
cap.release()

