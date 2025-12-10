import cv2
import numpy as np

# 1. 500x500 크기의 검은색 배경 이미지 생성
# 3채널 (컬러), 픽셀 값 타입은 8비트 정수 (0~255)
img = np.zeros((500, 500, 3), dtype=np.uint8) 

# 2. 선 그리기: 대각선 (파란색)
cv2.line(img, (0, 0), (500, 500), (255, 0, 0), 2) # B=255, G=0, R=0, 두께 2

# 3. 사각형 그리기: 중심부 (녹색, 채우기)
# (100, 100)부터 (400, 400)까지
#cv2.rectangle(img, (100, 100), (400, 400), (0, 255, 0), 9) 
cv2.rectangle(img, (100, 100), (400, 400), (0, 255, 0), -1)  #사각형 안을 채워라

# 4. 원 그리기: 중심부 (빨간색)
# 중심 (250, 250), 반지름 50
cv2.circle(img, (250, 250), 50, (0, 0, 255), 5) 

# 5. 텍스트 출력: 제목 (흰색)
text = "OpenCV Drawing Test"
# 시작 좌표 (50, 50), 폰트 크기 1, 두께 2
cv2.putText(img, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

# 6. 이미지 표시
cv2.imshow('Drawing Canvas', img)

cv2.waitKey(0)
cv2.destroyAllWindows()