import cv2
import time

# VideoCapture オブジェクトを取得します
cap = cv2.VideoCapture(0)

width = 1920*2
height = 1080*2

cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

while(True):
	ret, frame = cap.read()
	
	resized_img = cv2.resize(frame,(int(width/4), int(height/4)))
	re_width, re_height, _ = resized_img.shape
	cv2.drawMarker(resized_img, (int(re_height/2), int(re_width/2)), (255, 0, 0))

	cv2.imshow('frame',resized_img)
	
	unix_time = int(time.time())
	print(unix_time)
	if cv2.waitKey(1) & 0xFF == ord('s'):
		cv2.imwrite("4K_"+str(unix_time)+".jpg",frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()
