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
for i in range (100):
	ret,frame = cap.read()
	frame = cv2.resize(frame,dsize=(1280, 720))
	cv2.imshow('ATOM_Cam',frame)
	
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

height, width, channels = frame.shape

frame1 = frame[0:height//2, 0:width//2]     
# cv2.imwrite("./test-tl.jpg", clp)   
d = decode(frame1)
if d:
	for barcode in d:
		x,y,w,h = barcode.rect
		cv2.rectangle(frame1,(x,y),(x+w,y+h),(0,0,255),2)
		barcodeData = barcode.data.decode('utf-8')
		frame1 = cv2.putText(frame1,barcodeData,(x,y-10),font,.5,(0,0,255),2,cv2.LINE_AA)


frame2 = frame[0:height//2, width//2:width]     
# cv2.imwrite("./test-tr.jpg", clp)   
d = decode(frame2)
if d:
	for barcode in d:
		x,y,w,h = barcode.rect
		cv2.rectangle(frame2,(x,y),(x+w,y+h),(0,0,255),2)
		barcodeData = barcode.data.decode('utf-8')
		frame2 = cv2.putText(frame2,barcodeData,(x,y-10),font,.5,(0,0,255),2,cv2.LINE_AA)
		
frame3 = frame[height//2:height, 0:width//2]     
# cv2.imwrite("./test-ul.jpg", clp)   
d = decode(frame3)
if d:
	for barcode in d:
		x,y,w,h = barcode.rect
		cv2.rectangle(frame3,(x,y),(x+w,y+h),(0,0,255),2)
		barcodeData = barcode.data.decode('utf-8')
		frame3 = cv2.putText(frame3,barcodeData,(x,y-10),font,.5,(0,0,255),2,cv2.LINE_AA)
		
frame4 = frame[height//2:height, width//2:width]     
# cv2.imwrite("./test-ur.jpg", clp)
d = decode(frame4)
if d:
	for barcode in d:
		x,y,w,h = barcode.rect
		cv2.rectangle(frame4,(x,y),(x+w,y+h),(0,0,255),2)
		barcodeData = barcode.data.decode('utf-8')
		frame4 = cv2.putText(frame4,barcodeData,(x,y-10),font,.5,(0,0,255),2,cv2.LINE_AA)

im_v_1 = cv2.vconcat([frame1, frame3])
im_v_2 = cv2.vconcat([frame2, frame4])
img = cv2.hconcat([im_v_1, im_v_2])

cv2.imwrite("img.jpg", img)

cap.release()

