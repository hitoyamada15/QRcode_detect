from pyzbar.pyzbar import decode
import cv2
import time
from datetime import datetime, timedelta, timezone

# タイムゾーンの生成
JST = timezone(timedelta(hours=+9), 'JST')

# GOOD, タイムゾーンを指定している．早い
start_time = datetime.now(JST)
print(start_time)
# datetime.fromtimestamp(UNIX時間, JST)

count = 0
# rtspのURL指定でキャプチャするだけ
IP_ADDRESS = 'rtsp://192.168.50.122:8554/unicast'
cap = cv2.VideoCapture(IP_ADDRESS)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
# cap.set(cv2.CAP_PROP_FPS, 25)

# cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX
while cap.isOpened():
	ret,frame = cap.read()
    
	if not(ret):
		st = time.time()
		cap = cv2.VideoCapture(IP_ADDRESS)
		print("tot time lost due to reinitialization : ",time.time()-st)
		count += 1
		print("restart =",count)
		now_time = datetime.now(JST)
		print(now_time)
		continue

	#if ret == True:
	if ret == True:
		d = decode(frame)
		if d:
			for barcode in d:
				x,y,w,h = barcode.rect
				cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
				barcodeData = barcode.data.decode('utf-8')
				frame = cv2.putText(frame,barcodeData,(x,y-10),font,.5,(0,0,255),2,cv2.LINE_AA)

		for i in range (20):
			img = cap.read()
			img = None
		# frame = cv2.resize(frame,dsize=(1280, 720))
		cv2.namedWindow('ATOM_Cam', cv2.WINDOW_NORMAL)
		cv2.imshow('ATOM_Cam',frame)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()

