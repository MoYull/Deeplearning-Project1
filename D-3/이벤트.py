import cv2
import numpy as np 

img = np.zeros((500,500, 3), np.uint8 )

def draw_text(event, x, y, flags, params):
    if event==cv2.EVENT_LBUTTONDOWN:
        cv2.putText(img, "hi", (x, y), cv2.FONT_HERSHEY_TRIPLEX, 1.2, (255,255,0), 2)
        cv2.imshow("MyWindow", img)

#함수를 이벤트 핸들러로 등록하기 
cv2.namedWindow("MyWindow")
cv2.setMouseCallback("MyWindow", draw_text) #함수주소를 등록한다 

cv2.imshow('MyWindow', img)
cv2.waitKey(0) # 키 입력 대기 (창이 닫히지 않도록 함)
cv2.destroyAllWindows()