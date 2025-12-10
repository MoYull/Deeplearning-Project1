import cv2
import numpy as np

# 1. 전역 변수 설정
# True면 마우스가 눌린 상태(드래그 중), False면 놓인 상태
drawing = False 
# 사각형의 시작점 좌표
ix, iy = -1, -1 

# 500x500 크기의 검은색 배경 이미지 생성
img = np.zeros((500, 500, 3), np.uint8)

# 2. 마우스 이벤트 콜백 함수 정의
def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, img #전역변수 설정, 파이썬특징 
    
    # 마우스 왼쪽 버튼이 눌렸을 때 (시작점 저장)
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y # 시작 좌표 저장
        
    # 마우스가 이동 중일 때 (실시간으로 사각형 그리기)
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            # 원본 이미지의 복사본을 만들어 실시간으로 그리기
            img_temp = img.copy() 
            # 현재 위치 (x, y)까지 직사각형 그리기 (빨간색)
            cv2.rectangle(img_temp, (ix, iy), (x, y), (0, 0, 255), 2)
            cv2.imshow('Image', img_temp) # 임시 이미지를 화면에 표시
            #새로 만들어진 임시 이미지(img_temp) 위에 현재 드래그하는 빨간색 사각형을 그립니다.
            #이 임시 이미지를 화면에 표시합니다.
            #다음 마우스 이동 이벤트가 발생하면, 다시 img를 복사하여 새로운 img_temp를 만들고 
            #그 위에 사각형을 그립니다. 이전에 그렸던 img_temp는 버려지고 사라집니다.

    # 마우스 왼쪽 버튼이 놓였을 때 (사각형 완성)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        # 최종 사각형을 원본 이미지에 영구적으로 그리기 (초록색)
        cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 0), 2)
        cv2.imshow('Image', img)


# 3. 메인 루프 설정
cv2.namedWindow('Image') # 창 이름 설정, 창에 접근하기 위해서 
# 설정된 창에 콜백 함수를 연결
# 이벤트 처리 
cv2.setMouseCallback('Image', draw_rectangle) #Image창에 이벤트 핸들러 연결하기 

# 4. 이미지 표시 및 대기
cv2.imshow('Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()