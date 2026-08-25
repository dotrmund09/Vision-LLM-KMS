import numpy as np
import cv2
import time


# 초록 LAB lower/upper range
green_lower = np.array([30, 60, 90], dtype=np.uint8)
green_upper = np.array([230, 115, 180], dtype=np.uint8)

# Morphological operation용 타원형 kernel
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

pipeline = (
    "nvarguscamerasrc sensor-id=0 ! "
    "video/x-raw(memory:NVMM), width=1280, height=720, framerate=30/1 ! "
    "nvvidconv ! "
    "video/x-raw, format=BGRx ! "
    "videoconvert ! "
    "video/x-raw, format=BGR ! "
    "queue leaky=downstream max-size-buffers=1 ! "
    "appsink drop=true max-buffers=1 sync=false"
)

cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

while True:
    start = time.time()
    ret, frame = cap.read()

    if not ret:
        break
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    frame = cv2.flip(frame, 0)

    # 가우시안 블러
    blr = cv2.GaussianBlur(frame, (11, 11), 0)

    # LAB 색공간 변환
    lab = cv2.cvtColor(blr, cv2.COLOR_BGR2LAB)

    # LAB color segmentation
    mask = cv2.inRange(lab, green_lower, green_upper)

    # Opening 2회, Dilation 2회
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=2)

    # Contour detection
    contour_lst, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contour_lst) > 0:
        # 가장 큰 contour 선택
        contour = max(contour_lst, key=cv2.contourArea)
        # 최소 외접원 반지름
        _, radius = cv2.minEnclosingCircle(contour)
        # 무게중심
        M = cv2.moments(contour)
        if M["m00"] != 0:
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        else:
            center = (0, 0)

        # 검출된 객체에 파란 원 overlay
        cv2.circle(frame, center, int(radius), (255, 0, 0), 2)
        cv2.circle(frame, center, 5, (255, 0, 0), -1)

    #cv2.imshow("Object Detection", frame)
    cv2.imshow("LAB Mask", mask)

    # while loop rate (FPS) 설정
    time.sleep(max(1. / 25 - (time.time() - start), 0))

cap.release()
cv2.destroyAllWindows()