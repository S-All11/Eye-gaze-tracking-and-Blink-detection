import cv2
import dlib
from scipy.spatial import distance as dist
import time
import math

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def midpoint(p1, p2):
    return int((p1.x + p2.x) / 2), int((p1.y + p2.y) / 2)

def fatigue_score_from_blink_rate(blinks_per_minute):
    if 15 <= blinks_per_minute <= 20:
        return 0
    elif blinks_per_minute < 10:
        return min(100, (10 - blinks_per_minute) * 5)
    elif blinks_per_minute > 25:
        return min(100, (blinks_per_minute - 25) * 5)
    else:
        return 10

EYE_AR_THRESH = 0.2
EYE_AR_CONSEC_FRAMES = 3
COUNTER = 0
TOTAL_BLINKS = 0
BLINK_MESSAGE_TIME = 0
MINUTE_BLINKS = 0
last_minute_check = time.time()
fatigue_score = 0

# 開眼継続時間
last_blink_time = time.time()
max_eye_open_time = 0
OPEN_EYE_THRESHOLD = 10
ADDITIONAL_FATIGUE_SCORE = 10

# 視線移動量記録
gaze_movement_total = 0
prev_gaze_center = None

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("D:\\python_vsfile\\models\\shape_predictor_68_face_landmarks.dat")

(lStart, lEnd) = (42, 48)
(rStart, rEnd) = (36, 42)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)

    for rect in rects:
        shape = predictor(gray, rect)
        shape = [(shape.part(i).x, shape.part(i).y) for i in range(68)]

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0

        if ear < EYE_AR_THRESH:
            COUNTER += 1
        else:
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                TOTAL_BLINKS += 1
                MINUTE_BLINKS += 1
                BLINK_MESSAGE_TIME = time.time()

                now = time.time()
                open_duration = now - last_blink_time
                if open_duration > max_eye_open_time:
                    max_eye_open_time = open_duration
                last_blink_time = now
            COUNTER = 0

        if time.time() - BLINK_MESSAGE_TIME < 0.5:
            cv2.putText(frame, "You blink now!!", (50, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        # 視線中心点
        left_center = midpoint(dlib.point(leftEye[0]), dlib.point(leftEye[3]))
        right_center = midpoint(dlib.point(rightEye[0]), dlib.point(rightEye[3]))
        gaze_center = ((left_center[0] + right_center[0]) // 2,
                       (left_center[1] + right_center[1]) // 2)

        if prev_gaze_center is not None:
            dx = gaze_center[0] - prev_gaze_center[0]
            dy = gaze_center[1] - prev_gaze_center[1]
            distance = math.hypot(dx, dy)
            gaze_movement_total += distance
        prev_gaze_center = gaze_center

        for point in leftEye + rightEye:
            cv2.circle(frame, (point[0], point[1]), 2, (255, 0, 0), -1)
        cv2.circle(frame, left_center, 3, (0, 255, 0), -1)
        cv2.circle(frame, right_center, 3, (0, 255, 0), -1)
        cv2.circle(frame, gaze_center, 4, (255, 255, 0), -1)

    # 1分ごとの評価処理
    if time.time() - last_minute_check >= 60:
        fatigue_score = fatigue_score_from_blink_rate(MINUTE_BLINKS)

        if max_eye_open_time > OPEN_EYE_THRESHOLD:
            fatigue_score += ADDITIONAL_FATIGUE_SCORE
            fatigue_score = min(fatigue_score, 100)

        MINUTE_BLINKS = 0
        max_eye_open_time = 0
        gaze_movement_total = 0
        last_minute_check = time.time()

    if fatigue_score < 20:
        color = (0, 255, 0)
    elif fatigue_score < 50:
        color = (0, 255, 255)
    else:
        color = (0, 0, 255)

    # 表示
    cv2.putText(frame, f"Total Blinks: {TOTAL_BLINKS}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(frame, f"Fatigue Score: {fatigue_score}", (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    cv2.putText(frame, f"Gaze Movement: {int(gaze_movement_total)}", (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 255), 2)

    cv2.imshow("Blink and Gaze Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
