import cv2
from pyzbar.pyzbar import decode, ZBarSymbol


# -----------------------------------------------------------
# initial
# -----------------------------------------------------------
font = cv2.FONT_HERSHEY_SIMPLEX
FILE_PNG_AB = '4K_Jet_1642499491.jpg'


# -----------------------------------------------------------
# function_qr_dec
# -----------------------------------------------------------
def function_qrdec_pyzbar(img_bgr):

    # QRコードデコード
    value = decode(img_bgr, symbols=[ZBarSymbol.QRCODE])

    if value:
        for qrcode in value:

            # QRコード座標取得
            x, y, w, h = qrcode.rect

            # QRコードデータ
            dec_inf = qrcode.data.decode('utf-8')
            print('dec:', dec_inf)
            img_bgr = cv2.putText(img_bgr, dec_inf, (x, y - 6), font, .3, (255, 0, 0), 1, cv2.LINE_AA)

            # バウンディングボックス
            cv2.rectangle(img_bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow('image', img_bgr)
        cv2.waitKey(0)
        

# -----------------------------------------------------------
# sample program
# -----------------------------------------------------------
img_BGR = cv2.imread(FILE_PNG_AB, cv2.IMREAD_COLOR)
function_qrdec_pyzbar(img_BGR)
