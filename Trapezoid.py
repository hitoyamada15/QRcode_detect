import cv2
import numpy as np
import math
 
# 比率調整
w_ratio = 1.1
 
# 入力画像のパス
input_file_path = "4K_V2EQ_60.jpg" 
# 出力画像のパス
output_file_path = "4K_V2EQ_60_output.jpg"  

def Trapezoid_correction(top_left_xy, top_right_xy, bottom_left_xy, bottom_right_xy)
	# 変換前4点の座標　p1:左上　p2:右上 p3:左下 p4:左下
	p1 = np.array([267, 960])
	p2 = np.array([2544, 378])
	p3 = np.array([216, 1494])
	p4 = np.array([2592, 1053])
	 
	# 入力画像の読み込み
	img = cv2.imread(input_file_path)
	 
	#　幅取得
	o_width = np.linalg.norm(p2 - p1)
	o_width = math.floor(o_width * w_ratio)
	 
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
	cv2.imwrite(output_file_path, output)

Trapezoid_correction(top_left_xy, top_right_xy, bottom_left_xy, bottom_right_xy)
