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

while(True):
    ret, frame = cap.read()
    print(frame.shape)
    
    if not(ret):
        st = time.time()
        cap = cv2.VideoCapture(IP_ADDRESS)
        print("tot time lost due to reinitialization : ",time.time()-st)
        count += 1
        print("restart =",count)
        now_time = datetime.now(JST)
        print(now_time)
        continue
    
    # cv2.setWindowProperty('frame', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    # frame = cv2.resize(frame,(1280, 720))
    cv2.namedWindow('ATOM_1', cv2.WINDOW_NORMAL)
    cv2.imshow('ATOM_1',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
