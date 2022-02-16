import cv2
from pyzbar.pyzbar import decode, ZBarSymbol


# -----------------------------------------------------------
# Init
# -----------------------------------------------------------
font = cv2.FONT_HERSHEY_SIMPLEX


# -----------------------------------------------------------
# 画像キャプチャ
# -----------------------------------------------------------
# VideoCaptureインスタンス生成
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while cap.isOpened():
    ret, frame = cap.read()

    if ret:
        # デコード
        value = decode(frame, symbols=[ZBarSymbol.QRCODE])

        if value:
            for qrcode in value:

                # QRコード座標
                x, y, w, h = qrcode.rect

                # QRコードデータ
                dec_inf = qrcode.data.decode('utf-8')
                print('dec:', dec_inf)
                frame = cv2.putText(frame, dec_inf, (x, y-6), font, .3, (255, 0, 0), 1, cv2.LINE_AA)

                # バウンディングボックス
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 1)

        # 画像表示
        cv2.imshow('pyzbar', frame)

    # quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# キャプチャリソースリリース
cap.release()
