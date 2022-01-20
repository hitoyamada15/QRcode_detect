import numpy as np
import cv2
from cv2 import aruco
import math
import time
import datetime
from pyzbar.pyzbar import decode, ZBarSymbol


cap = cv2.VideoCapture(0)
# 画質設定
CAP_FRAME = [1920, 1080] # FullHD
# CAP_FRAME = [3840, 2160] # 4K
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_FRAME[0]) 
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_FRAME[1]) 

FONT = cv2.FONT_HERSHEY_SIMPLEX

# ARマーカーの番号
AR_top_left     = 0
AR_top_right    = 1
AR_bottom_left  = 2
AR_bottom_right = 3

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
	cv2.imwrite("./aruco_detect/"+ save_time +"_aruco_detect.jpg", frame_markers)
	"""
	# もしARマーカーを読み取れないときに使う
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
	cv2.imwrite("./trapezoid_correction/"+ save_time + "_trapezoid_correction.jpg", output)

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
	dst = cv2.imread("./trapezoid_correction/"+ save_time + "_trapezoid_correction.jpg", cv2.IMREAD_COLOR)
	dst = cv2.drawContours(dst, contours, -1, (0, 0, 255, 255), 10, cv2.LINE_AA)
	cv2.imwrite("./rectangle/" + save_time + "_rectangle.jpg", dst)
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

	detect_count = 0
	for i, contour in enumerate(contours):
		# 小さな領域の場合は間引く
		area = cv2.contourArea(contour)
		if area < 4000:
		 	continue
		# 画像全体を占める領域は除外する
		if image_size * 0.9 < area:
		 	continue

		# 外接矩形を取得
		x,y,w,h = cv2.boundingRect(contour)
		dst = cv2.rectangle(dst,(x,y),(x+w,y+h),(0,255,0),2)
		cut_img = src[y : y+h , x : x+w]
		cv2.imwrite("./roundingrect_detect/" + save_time + "_roundingrect_detect.jpg", dst)
		value = decode(cut_img, symbols=[ZBarSymbol.QRCODE])

		if value:
			for qrcode in value:
				print(detect_count)
				# QRコード座標取得
				x, y, w, h = qrcode.rect

				# QRコードデータ
				dec_inf = qrcode.data.decode('utf-8')
				print('dec:', dec_inf)
				cut_img = cv2.putText(cut_img, dec_inf, (x, y-6), FONT, 0.4, (255, 0, 0), 1, cv2.LINE_AA)
				# バウンディングボックス
				cv2.rectangle(cut_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

				detect_count += 1
	print(detect_count,"個検出")
	# 結果を保存
	cv2.imwrite("./result/" + save_time + "result.jpg", src)
	time.sleep(3)
	

if __name__ == '__main__':
	while(True):
		ret, cap_frame = cap.read()

		now_date = datetime.datetime.now()
		save_time = now_date.strftime('%Y%m%d_%H%M%S')
		print(save_time)

		cv2.namedWindow('qr_detection', cv2.WINDOW_NORMAL)
		cv2.imshow('qr_detection',cap_frame)
		# Key 's'を押したら検出
		if cv2.waitKey(10) & 0xFF == ord('s'):
			print("detect start")
			cv2.imwrite("./orignal/" + save_time + "_orignal.jpg", cap_frame)
			# 関数(Aruco_detect)
			top_left_xy, top_right_xy, bottom_left_xy, bottom_right_xy = Aruco_detect(cap_frame)
			# 関数(Trapezoid_correction)				
			output_img = Trapezoid_correction(cap_frame, top_left_xy, top_right_xy, bottom_left_xy, bottom_right_xy)
			# 関数(rectangle_detect)
			rectangle_detect(output_img)

		# Key 'q'を押したら終了
		elif cv2.waitKey(10) & 0xFF == ord('q'):
		 	break

	cap.release()
	cv2.destroyAllWindows()

