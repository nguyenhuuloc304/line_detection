import numpy as np
import cv2
import math

aLeftStored = 0
aRightStored = 0
bLeftStored = 0
bRightStored = 0
yMinStored = 0

def get_top_y_point(lineParams, y_min):
    global yMinStored
    param1 = 0.9
    param2 = 0.1

    a_left = lineParams[0]
    b_left = lineParams[1]
    a_right = lineParams[2]
    b_right = lineParams[3]

    yIntersection = int((a_right*b_left-a_left*b_right)/(a_right-a_left))
    margin = 10
    if (yIntersection + margin > y_min):
        y_min = yIntersection + margin

    if (yMinStored==0):
        yMinStored = y_min
    y_min = int(yMinStored * param1 + y_min * param2)
    yMinStored = y_min
    return y_min

def get_averaged_line_params(lineParams, leftHoughLinesExist, rightHoughLinesExist):
    global aLeftStored
    global bLeftStored
    global aRightStored
    global bRightStored

    param1 = 0.9
    param2 = 0.1

    a_left = lineParams[0]
    b_left = lineParams[1]
    a_right = lineParams[2]
    b_right = lineParams[3]

    if (aLeftStored == 0):
        aLeftStored = a_left
    if (bLeftStored == 0):
        bLeftStored = b_left
    if (aRightStored == 0):
        aRightStored = a_right
    if (bRightStored == 0):
        bRightStored = b_right

    if (not leftHoughLinesExist):
        a_left = aLeftStored
    else:
        a_left = aLeftStored * param1 + a_left * param2
        aLeftStored = a_left
        b_left = bLeftStored * param1 + b_left * param2
        bLeftStored = b_left

    if (not rightHoughLinesExist):
        a_right = aRightStored
    else:
        a_right = aRightStored * param1 + a_right * param2
        aRightStored = a_right
        b_right = bRightStored * param1 + b_right * param2
        bRightStored = b_right

    return [a_left, b_left, a_right, b_right]

def processImage (image):
    grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel_size = 5
    blur_gray = cv2.GaussianBlur(grey_image,(kernel_size, kernel_size),0)
    low_threshold = 50
    high_threshold = 150
    edges = cv2.Canny(blur_gray, low_threshold, high_threshold)
    mask = np.zeros_like(edges)
    ignore_mask_color = 255
    imshape = image.shape
    vertices = np.array([[(0,imshape[0]),(450, 320), (500, 320), (imshape[1],imshape[0])]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_edges = cv2.bitwise_and(edges, mask)
    rho = 2
    theta = np.pi/180
    threshold = 15
    min_line_length = 40
    max_line_gap = 30
    lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]),
                                min_line_length, max_line_gap)
    linesFiltered = []
    cumLengthLeft = 0
    cumLengthRight = 0
    leftHoughLinesExist = False;
    rightHoughLinesExist = False;
    a_left = 0
    b_left = 0
    a_right = 0
    b_right = 0
    y_min = 10000
    yTopMask = imshape[0] * 0.55
    for line in lines:
        for x1, y1, x2, y2 in line:
            if (y1 < y_min):
                y_min = y1
            if (y2 < y_min):
                y_min = y2
            a = float((y2 - y1) / (x2 - x1))
            b = (y1 - a * x1)
            length = math.sqrt(pow(y2 - y1, 2) + pow(x2 - x1, 2))

            if not np.isnan(a) or np.isinf(a) or (a == 0):
                if (a > -1.5) and (a < -0.3):
                    linesFiltered.append(line)
                    cumLengthLeft += pow(length, 2)
                    leftHoughLinesExist = True
                    a_left += a * pow(length, 2)
                    b_left += b * pow(length, 2)

                if (a > 0.3) and (a < 1.5):
                    linesFiltered.append(line)
                    cumLengthRight += pow(length, 2)
                    rightHoughLinesExist = True
                    a_right += a * pow(length, 2)
                    b_right += b * pow(length, 2)
    if (y_min == 10000):
        y_min = yTopMask

    y_max = imshape[0]

    if (cumLengthLeft != 0):
        a_left /= cumLengthLeft
        b_left /= cumLengthLeft

    if (cumLengthRight != 0):
        a_right /= cumLengthRight
        b_right /= cumLengthRight

    lineParams = [a_left, b_left, a_right, b_right]
    lineParams = get_averaged_line_params(lineParams, leftHoughLinesExist, rightHoughLinesExist)

    y_min = get_top_y_point(lineParams, y_min)

    x1_left = 0
    x2_left = 0
    x1_right = 0
    x2_right = 0

    a_left = lineParams[0]
    b_left = lineParams[1]
    a_right = lineParams[2]
    b_right = lineParams[3]

    if (a_left != 0):
        x1_left = int((y_max - b_left) / a_left)
        x2_left = int((y_min - b_left) / a_left)

    if (a_right != 0):
        x1_right = int((y_max - b_right) / a_right)
        x2_right = int((y_min - b_right) / a_right)

    foundLinesImage = np.zeros((imshape[0], imshape[1], 3), dtype=np.uint8)

    if (a_left != 0):
        cv2.line(foundLinesImage, (x1_left, y_max), (x2_left, y_min), [255, 0, 0], 7)
        pts = np.array([[x1_left, y_max - 10], [x2_left, y_min - 10], [x2_left, y_min - 50], [x1_left, y_max - 200]],
                       np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(foundLinesImage, [pts], True, (0 , 255, 255), 2)

    if (a_right != 0):
        cv2.line(foundLinesImage, (x1_right, y_max), (x2_right, y_min), [255, 0, 0], 7)
        pts = np.array([[x1_right, y_max - 10], [x2_right, y_min - 10], [x2_right, y_min - 50], [x1_right, y_max - 200]],
                       np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(foundLinesImage, [pts], True, (0, 255, 255), 2)
    if(a_left != 0) and (a_right != 0):
        pts = np.array(
            [[x1_left, y_max], [x2_left, y_min], [x2_right, y_min], [x1_right, y_max]], np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.fillConvexPoly(foundLinesImage, pts, (0, 255, 0))
        x1_center = int((x1_left + x1_right) / 2)
        x2_center = int((x2_left + x2_right) / 2)
        x_center = int(imshape[1] / 2)
        pts1 = (x2_center, y_min + 30)
        pts2 = (x1_center, y_max - 30)
        cv2.line(foundLinesImage, (x_center, y_min), (x_center, y_max), (255, 255, 0), 3)
        turn = "Giua"
        cv2.arrowedLine(foundLinesImage, pts2, pts1, (0, 0, 255), 3)
        if(x1_center < x_center - 10):
            turn = "Trai -" + str(x_center - x1_center)
        else:
            if(x1_center > x_center + 10):
                turn = "Phai +" + str(x1_center - x_center)
        cv2.putText(foundLinesImage, turn, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255))


    origWithFoundLanes = cv2.addWeighted(foundLinesImage, 0.3, image, 1., 0.)

    return origWithFoundLanes


video_capture = cv2.VideoCapture('test_videos/solidWhiteRight.mp4')

while (video_capture.isOpened()):
    ret, frame = video_capture.read()
    if ret:
        output = processImage(frame)
        cv2.imshow('frame',output)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release everything if job is finished
video_capture.release()
cv2.destroyAllWindows()

#cv2.imshow('Test image',lines_edges)
#cv2.waitKey(0)
#cv2.destroyAllWindows()