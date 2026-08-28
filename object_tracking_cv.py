import numpy as np
import cv2
import time


# --------------- OpenCV Kalman Filter 설정 ---------------
# 상태 벡터: [x, y, vx, vy]
# 측정 벡터: [x, y]
kalman = cv2.KalmanFilter(4, 2)

# Transition Matrix (상태 전이 행렬)
kalman.transitionMatrix = np.array(
    [
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ],
    dtype=np.float32,
)

# Measurement Matrix (측정 행렬)
# 측정값으로부터 x, y만 관측
kalman.measurementMatrix = np.array(
    [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
    ],
    dtype=np.float32,
)

# Process Noise Covariance Matrix (프로세스 노이즈 공분산)
# 값이 클수록 모델 예측보다 측정값 변화에 더 유연하게 반응
kalman.processNoiseCov = np.array(
    [
        [1e-2, 0, 0, 0],
        [0, 1e-2, 0, 0],
        [0, 0, 5e-2, 0],
        [0, 0, 0, 5e-2],
    ],
    dtype=np.float32,
)

# Measurement Noise Covariance Matrix (측정 노이즈 공분산)
# 값이 클수록 측정값을 덜 신뢰하고 예측값을 더 신뢰
kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1.0

# Posteriori Error Covariance Matrix (초기 추정 오차 공분산)
kalman.errorCovPost = np.eye(4, dtype=np.float32)

# -------------------------------------------------------

# 초록 LAB lower/upper range
green_lower = np.array([30, 60, 90], dtype=np.uint8)
green_upper = np.array([230, 115, 180], dtype=np.uint8)

# Morphological operation용 타원형 kernel
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# 객체 최초 검출 여부 확인용 boolean
found = False

# 객체 좌표 및 반지름 변수 정의
x_bel, y_bel = 0, 0
radius = 0

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

        # Measurement (측정값)
        z = np.array([[center[0]], [center[1]]],dtype=np.float32)

        # 객체 최초 검출 (최초 a priori state 저장)
        if not found:
            kalman.statePre = np.array([[center[0]], [center[1]], [0.0], [0.0]], dtype=np.float32)
            found = True

    # 최초 검출 이후 Kalman Filter 적용
    # 측정값 사용 (visible) -> Prediction & Update
    if found and (len(contour_lst) > 0):
        predicted_state = kalman.predict()   # prediction
        corrected_state = kalman.correct(z)  # update
        x_bel = corrected_state[0, 0]
        y_bel = corrected_state[1, 0]

    # 측정값 미사용 (occluded) -> Prediction
    elif found and (len(contour_lst) <= 0):
        predicted_state = kalman.predict()   # prediction (update X)
        x_bel = predicted_state[0, 0]
        y_bel = predicted_state[1, 0]

    # 예측한 객체 위치에 노란 원 overlay
    cv2.circle(frame, (int(x_bel), int(y_bel)), int(radius), (0, 255, 255), 2)
    cv2.circle(frame, (int(x_bel), int(y_bel)), 5, (0, 255, 255), -1)

    cv2.imshow("Object Detection", frame)
    # cv2.imshow("LAB Mask", mask)

    # while loop rate (FPS) 설정
    time.sleep(max(1. / 25 - (time.time() - start), 0))

cap.release()
cv2.destroyAllWindows()
