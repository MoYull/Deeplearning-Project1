import cv2
import numpy as np


file_path = './images/sample2.jpg'
img_color = cv2.imread(file_path)

if img_color is None:
    print("이미지 로드 실패")
    exit()

# 1. 전처리: 흑백 변환
img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
img_copy = img_color.copy() # 윤곽선을 그릴 원본 복사본

# 2. Canny 엣지 검출
# 임계값: (50, 150)
img_edge = cv2.Canny(img_gray, 50, 150) 
cv2.imshow('Canny Edge', img_edge)

# 3. 윤곽선 검출
# Canny 엣지 맵(이진 이미지)을 입력으로 사용합니다.
# cv2.RETR_EXTERNAL: 가장 바깥쪽 윤곽선만 찾음
# cv2.CHAIN_APPROX_SIMPLE: 수직, 수평, 대각선 끝점만 저장하여 메모리 절약
contours, hierarchy = cv2.findContours(img_edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 4. 윤곽선 그리기
# 모든 윤곽선(-1)을 빨간색으로, 두께 2로 원본 복사본에 그립니다.
cv2.drawContours(img_copy, contours, -1, (0, 0, 255), 2)

# 5. 결과 표시
print(f"검출된 윤곽선 개수: {len(contours)}")
cv2.imshow('Contours Drawn', img_copy)

cv2.waitKey(0)
cv2.destroyAllWindows()