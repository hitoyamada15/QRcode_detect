import numpy as np
import cv2
from cv2 import aruco
import matplotlib.pyplot as plt
import math
from pyzbar.pyzbar import decode, ZBarSymbol

# 画像読み込み 
FRAME = cv2.imread("4K_1642565186_700mm.jpg")

# ARマーカーの番号
AR_top_left = 0
AR_top_right =  1
AR_bottom_right =  3
AR_bottom_left =  2

# 検知箇所を画像にマーキング
AR_detect_img = "aruco_detect.jpg"

# 射影変換・透視変換した画像の保存
Trapezoid_img = "Trapezoid_correction.jpg"
# 台形補正 比率調整
W_ratio = 0.8

# 四角を抽出した画像
Rectangle_img = "Rectangle.jpg"

# 結果画像
Result_img = "result.jpg"

font = cv2.FONT_HERSHEY_SIMPLEX


# ARマーカーから4点の座標を取得
# Web:【python できること】簡単！ARマーカーの作り方と検知する方法
# URL:https://hituji-ws.com/code/python/python-armarker/
# Web:【簡単】QRコードの作成と読み取り in Python
# URL:https://qiita.com/PoodleMaster/items/0afbce4be7e442e75be6
def Aruco_detect(frame):
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
	
	top_left_xy_list = []
	top_right_xy_list = []
	bottom_left_xy_list = []
	bottom_right_xy_list = []
	# ARマーカー検知
	aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
	parameters =  aruco.DetectorParameters_create()
	corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
	
	# 座標とidの確認
	for i in range(len(ids)):
		# 検知したidの4点取得
		c = corners[i][0]
		x1, x2, x3, x4 = c[:, 0]
		y1, y2, y3, y4 = c[:, 1]

		print(f"id={ids[i]}")
		print("X座標", x1, x2, x3, x4)
		print("Y座標", y1, y2, y3, y4)
		print("中心座標", c[:, 0].mean(), c[:, 1].mean())

		if ids[i] == AR_top_left: # 左上
			top_left_x = int(c[:, 0].mean())
			top_left_y = int(c[:, 1].mean())
			top_left_xy_list = np.append(top_left_xy_list, top_left_x)
			top_left_xy_list = np.append(top_left_xy_list, top_left_y)
		elif ids[i] == AR_top_right: # 右上
			top_right_x = int(c[:, 0].mean())
			top_right_y = int(c[:, 1].mean())
			top_right_xy_list = np.append(top_right_xy_list, top_right_x)
			top_right_xy_list = np.append(top_right_xy_list, top_right_y)
		elif ids[i] == AR_bottom_right: # 右下
			bottom_left_x = int(c[:, 0].mean())
			bottom_left_y = int(c[:, 1].mean())
			bottom_left_xy_list = np.append(bottom_left_xy_list, bottom_left_x)
			bottom_left_xy_list = np.append(bottom_left_xy_list, bottom_left_y)
		elif ids[i] == AR_bottom_left: # 左下
			bottom_right_x = int(c[:, 0].mean())
			bottom_right_y = int(c[:, 1].mean())
			bottom_right_xy_list = np.append(bottom_right_xy_list, bottom_right_x)
			bottom_right_xy_list = np.append(bottom_right_xy_list, bottom_right_y)
	
	# 検知箇所を画像にマーキング
	frame_markers = aruco.drawDetectedMarkers(frame.copy(), corners, ids)
	frame_markers = cv2.cvtColor(frame_markers, cv2.COLOR_BGR2RGB)
	cv2.imwrite(AR_detect_img, frame_markers)
	"""
	top_left_xy_list = np.array([781, 76])
	top_right_xy_list = np.array([2472, 766])
	bottom_left_xy_list = np.array([641, 2415])
	bottom_right_xy_list = np.array([2469, 1860])
	"""
	
	return top_left_xy_list, top_right_xy_list, bottom_left_xy_list, bottom_right_xy_list


#台形補正
#Web:OpenCVによる台形補正・射影変換を解説【Python】
#URL:https://self-development.info/opencvによる台形補正・射影変換を解説【python】/
def Trapezoid_correction(img, top_left_xy, top_right_xy, bottom_left_xy, bottom_right_xy):
	# 変換前4点の座標 p1:左上 p2:右上 p3:左下 p4:右下
	p1 = np.array(top_left_xy)
	p2 = np.array(top_right_xy)
	p3 = np.array(bottom_left_xy)
	p4 = np.array(bottom_right_xy)
	
	# 幅取得
	o_width = np.linalg.norm(p2 - p1)
	o_width = math.floor(o_width * W_ratio)
	 
	# 高さ取得
	o_height = np.linalg.norm(p3 - p1)
	o_height = math.floor(o_height)
	 
	# 変換前の4点
	src = np.float32([p1, p2, p3, p4])
	 
	# 変換後の4点
	dst = np.float32([[0, 0],[o_width, 0],[0, o_height],[o_width, o_height]])
	 
	# 変換行列
	M = cv2.getPerspectiveTransform(src, dst)
	 
	# 射影変換・透視変換する
	output = cv2.warpPerspective(img, M,(o_width, o_height))
	 
	# 射影変換・透視変換した画像の保存
	cv2.imwrite(Trapezoid_img, output)

	return output


# QRコード
# https://qiita.com/PoodleMaster/items/0afbce4be7e442e75be6

# 矩形検出
# https://symfoware.blog.fc2.com/blog-entry-2163.html
def rectangle_detect(src):
	# 画像の大きさ取得
	height, width, channels = src.shape
	image_size = height * width
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
	dst = cv2.imread(Trapezoid_img, cv2.IMREAD_COLOR)
	dst = cv2.drawContours(dst, contours, -1, (0, 0, 255, 255), 15, cv2.LINE_AA)
	cv2.imwrite(Rectangle_img, dst)
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

	detect_count = 1
	for i, contour in enumerate(contours):
		# 小さな領域の場合は間引く
		area = cv2.contourArea(contour)
		if area < 4000:
		 	continue
		# 画像全体を占める領域は除外する
		if image_size * 0.99 < area:
		 	continue

		# 外接矩形を取得
		x,y,w,h = cv2.boundingRect(contour)
		dst = cv2.rectangle(dst,(x,y),(x+w,y+h),(0,255,0),2)
		cut_img = src[y : y+h , x : x+w]
		cv2.imwrite("rectangle_detect.jpg", dst)
		value = decode(cut_img, symbols=[ZBarSymbol.QRCODE])

		if value:
			for qrcode in value:
				print(detect_count)
				# QRコード座標取得
				x, y, w, h = qrcode.rect

				# QRコードデータ
				dec_inf = qrcode.data.decode('utf-8')
				print('dec:', dec_inf)
				cut_img = cv2.putText(cut_img, dec_inf, (x, y-6), font, 0.4, (255, 0, 0), 1, cv2.LINE_AA)
				# バウンディングボックス
				cv2.rectangle(cut_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
				# cv2.imwrite("qr_detect/qr_detect_%03d.jpg"%detect_count, cut_img)

				detect_count += 1

	# 結果を保存
	cv2.imwrite(Result_img, src)


if __name__ == '__main__':
	top_left_xy, top_right_xy, bottom_left_xy, bottom_right_xy = Aruco_detect(FRAME)

	output_img = Trapezoid_correction(FRAME, top_left_xy, top_right_xy, bottom_left_xy, bottom_right_xy)
	
	rectangle_detect(output_img)

