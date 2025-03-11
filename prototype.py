import cv2
import dlib
from scipy.spatial import distance as dist
import time

# 瞬き検知のためのEAR計算関数
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# 視線検知のための中点計算関数
def midpoint(p1, p2):
    return int((p1.x + p2.x) / 2), int((p1.y + p2.y) / 2)

# EARのしきい値とまばたきのフレーム閾値
EYE_AR_THRESH = 0.2
EYE_AR_CONSEC_FRAMES = 3

# 初期設定
COUNTER = 0
TOTAL_BLINKS = 0
BLINK_MESSAGE_TIME = 0

# dlibの顔検出器とランドマーク検出器
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("D:\\python_vsfile\\models\\shape_predictor_68_face_landmarks.dat")

# 目のランドマークインデックス
(lStart, lEnd) = (42, 48)
(rStart, rEnd) = (36, 42)

# カメラ起動
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

        # 左右の目のランドマーク
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0

        # 瞬き検知
        if ear < EYE_AR_THRESH:
            COUNTER += 1
        else:
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                TOTAL_BLINKS += 1
                BLINK_MESSAGE_TIME = time.time()
            COUNTER = 0

        # 瞬きメッセージ表示
        if time.time() - BLINK_MESSAGE_TIME < 0.5:
            cv2.putText(frame, "You blink now!!", (50, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        # 視線検知用の目の中心点
        left_center = midpoint(dlib.point(leftEye[0]), dlib.point(leftEye[3]))
        right_center = midpoint(dlib.point(rightEye[0]), dlib.point(rightEye[3]))

        # 両目のランドマークを描画
        for point in leftEye + rightEye:
            cv2.circle(frame, (point[0], point[1]), 2, (255, 0, 0), -1)

        # 中心点を描画
        cv2.circle(frame, left_center, 3, (0, 255, 0), -1)
        cv2.circle(frame, right_center, 3, (0, 255, 0), -1)

    # 瞬き累計回数の表示
    cv2.putText(frame, f"Total Blinks: {TOTAL_BLINKS}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    cv2.imshow("Blink and Gaze Detection", frame)

    # 'q'キーで終了
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
