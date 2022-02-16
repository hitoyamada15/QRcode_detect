import cv2
import numpy as np


# -----------------------------------------------------------
# Init
# -----------------------------------------------------------
font = cv2.FONT_HERSHEY_SIMPLEX

# -----------------------------------------------------------
# 画像キャプチャ
# -----------------------------------------------------------
# VideoCaptureインスタンス生成
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# QRCodeDetectorインスタンス生成
qrd = cv2.QRCodeDetector()

while cap.isOpened():
    ret, frame = cap.read()

    if ret:
        # QRコードデコード
        retval, decoded_info, points, straight_qrcode = qrd.detectAndDecodeMulti(frame)

        if retval:
            points = points.astype(np.int32)

            for dec_inf, point in zip(decoded_info, points):
                if dec_inf == '':
                    continue

                # QRコード座標取得
                x = point[0][0]
                y = point[0][1]

                # QRコードデータ
                print('dec:', dec_inf)
                frame = cv2.putText(frame, dec_inf, (x, y - 6), font, .3, (0, 0, 255), 1, cv2.LINE_AA)

                # バウンディングボックス
                frame = cv2.polylines(frame, [point], True, (0, 255, 0), 1, cv2.LINE_AA)

        # 画像表示
        cv2.imshow('cv2', frame)

    # quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# キャプチャリソースリリース
cap.release()
