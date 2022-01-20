# QRコード
# https://qiita.com/PoodleMaster/items/0afbce4be7e442e75be6

# 矩形検出
# https://symfoware.blog.fc2.com/blog-entry-2163.html

import cv2
import numpy as np

from pyzbar.pyzbar import decode, ZBarSymbol

# ファイルを読み込み
image_file = 'Trapezoid_correction.jpg'
# -----------------------------------------------------------
# initial
# -----------------------------------------------------------
font = cv2.FONT_HERSHEY_SIMPLEX

# -----------------------------------------------------------
# function_qr_dec
# -----------------------------------------------------------

def rectangle_detect():

    src = cv2.imread(image_file, cv2.IMREAD_COLOR)
    
    # 画像の大きさ取得
    height, width, channels = src.shape
    image_size = height * width
    
    # src = cv2.resize(src, (int(width * 0.5), int(height * 0.5)))
    # cv2.imwrite('resize_1.png', src)
    # src = cv2.resize(src, (int(width), int(height)))
    # cv2.imwrite('resize_2.png', src)
    
    # 画像の大きさ取得
    # height, width, channels = src.shape
    # image_size = height * width
    
    # --------------------------------------------------------------------------
    # グレースケール化
    img_gray = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
    # しきい値指定によるフィルタリング
    retval, dst = cv2.threshold(img_gray, 127, 255, cv2.THRESH_TOZERO_INV )
    dst = cv2.bitwise_not(dst)
    # 再度フィルタリング
    retval, dst = cv2.threshold(dst, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    # 輪郭を抽出
    contours, hierarchy = cv2.findContours(dst, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # --------------------------------------------------------------------------
    # この時点での状態をデバッグ出力
    dst = cv2.imread(image_file, cv2.IMREAD_COLOR)
    dst = cv2.drawContours(dst, contours, -1, (0, 0, 255, 255), 10, cv2.LINE_AA)
    cv2.imwrite('debug_1.png', dst)
    # --------------------------------------------------------------------------
    # グレースケール化
    img_gray = cv2.cvtColor(dst, cv2.COLOR_RGB2GRAY)
    # しきい値指定によるフィルタリング
    retval, dst = cv2.threshold(img_gray, 127, 255, cv2.THRESH_TOZERO_INV )
    dst = cv2.bitwise_not(dst)
    # 再度フィルタリング
    retval, dst = cv2.threshold(dst, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    # 輪郭を抽出
    contours, hierarchy = cv2.findContours(dst, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # --------------------------------------------------------------------------
    
    
    cv2.imwrite('debug_2.png', dst)
    detect_count = 1
    dst = cv2.imread(image_file, cv2.IMREAD_COLOR)
    for i, contour in enumerate(contours):
        # 小さな領域の場合は間引く
        area = cv2.contourArea(contour)
        if area < 4000:
            continue
        # 画像全体を占める領域は除外する
        if image_size * 0.8 < area:
            continue
        
        # 外接矩形を取得
        x,y,w,h = cv2.boundingRect(contour)
        # dst = cv2.rectangle(dst,(x,y),(x+w,y+h),(0,255,0),2)
        
        cut_img = dst[y : y+h , x : x+w]
        value = decode(cut_img, symbols=[ZBarSymbol.QRCODE])

        if value:
            for qrcode in value:
                # QRコード座標取得
                x, y, w, h = qrcode.rect
        
                # QRコードデータ
                dec_inf = qrcode.data.decode('utf-8')
                print('dec:', dec_inf)
                cut_img = cv2.putText(cut_img, dec_inf, (x, y-5), font, 0.6, (255, 0, 0), 1, cv2.LINE_AA)
                # バウンディングボックス
                cv2.rectangle(cut_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.imwrite("qr_detect/qr_detect_"+str(i)+".jpg", cut_img)
                print(detect_count)
                detect_count += 1
        # cv2.imshow('image', cut_img)
        # cv2.waitKey(0)
        
        
    # 結果を保存
    cv2.imwrite('result.png', dst)
    
    
if __name__ == '__main__':
    rectangle_detect()

