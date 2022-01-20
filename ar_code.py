import numpy as np
import cv2
from cv2 import aruco
import matplotlib.pyplot as plt

# 画像読み込み 
FRAME = cv2.imread("4K_V2EQ_60.jpg")

def aruco_detect(frame):
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


top_left_xy, top_right_xy, bottom_left_xy, bottom_right_xy = aruco_detect(FRAME)
print(top_left_xy, top_right_xy, bottom_left_xy, bottom_right_xy)
