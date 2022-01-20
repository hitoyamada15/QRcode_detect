import numpy as np
import cv2
from cv2 import aruco
import matplotlib.pyplot as plt
import math

# 画像読み込み 
FRAME = cv2.imread("4K_V2EQ_60.jpg")
# 台形補正 比率調整
W_ratio = 0.8


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
		print("i",ids[i])
		x1, x2, x3, x4 = c[:, 0]
		y1, y2, y3, y4 = c[:, 1]

		print(f"id={ids[i]}")
		print("X座標", x1, x2, x3, x4)
		print("Y座標", y1, y2, y3, y4)
		print("中心座標", c[:, 0].mean(), c[:, 1].mean())

		if ids[i] == 0:
			top_left_x = int(c[:, 0].mean())
			top_left_y = int(c[:, 1].mean())
			top_left_xy_list = np.append(top_left_xy_list, top_left_x)
			top_left_xy_list = np.append(top_left_xy_list, top_left_y)
		elif ids[i] == 5:
			top_right_x = int(c[:, 0].mean())
			top_right_y = int(c[:, 1].mean())
			top_right_xy_list = np.append(top_right_xy_list, top_right_x)
			top_right_xy_list = np.append(top_right_xy_list, top_right_y)
		elif ids[i] == 11:
			bottom_left_x = int(c[:, 0].mean())
			bottom_left_y = int(c[:, 1].mean())
			bottom_left_xy_list = np.append(bottom_left_xy_list, bottom_left_x)
			bottom_left_xy_list = np.append(bottom_left_xy_list, bottom_left_y)
		elif ids[i] == 14:
			bottom_right_x = int(c[:, 0].mean())
			bottom_right_y = int(c[:, 1].mean())
			bottom_right_xy_list = np.append(bottom_right_xy_list, bottom_right_x)
			bottom_right_xy_list = np.append(bottom_right_xy_list, bottom_right_y)
	
	# 検知箇所を画像にマーキング
	frame_markers = aruco.drawDetectedMarkers(frame.copy(), corners, ids)
	frame_markers = cv2.cvtColor(frame_markers, cv2.COLOR_BGR2RGB)
	cv2.imwrite("aruco_detect.jpg",frame_markers)
	
	return top_left_xy_list, top_right_xy_list, bottom_left_xy_list, bottom_right_xy_list

#台形補正
#Web:OpenCVによる台形補正・射影変換を解説【Python】
#URL:https://self-development.info/opencvによる台形補正・射影変換を解説【python】/
def Trapezoid_correction(img, top_left_xy, top_right_xy, bottom_left_xy, bottom_right_xy):
	# 変換前4点の座標　p1:左上　p2:右上 p3:左下 p4:右下
	p1 = np.array(top_left_xy)
	p2 = np.array(top_right_xy)
	p3 = np.array(bottom_left_xy)
	p4 = np.array(bottom_right_xy)
	 
	#　幅取得
	o_width = np.linalg.norm(p2 - p1)
	o_width = math.floor(o_width * W_ratio)
	 
	#　高さ取得
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
	cv2.imwrite("Trapezoid_correction.jpg", output)

	return output

top_left_xy, top_right_xy, bottom_left_xy, bottom_right_xy = Aruco_detect(FRAME)
print(top_left_xy, top_right_xy, bottom_left_xy, bottom_right_xy)
Trapezoid_correction(FRAME, top_left_xy, top_right_xy, bottom_left_xy, bottom_right_xy)




