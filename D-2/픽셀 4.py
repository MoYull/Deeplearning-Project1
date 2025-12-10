#이미지 채널 분리 및 단일 채널 시각화
#컬러 이미지를 BGR 세 개의 개별 흑백 이미지(채널)로 분리하고, 각 채널의 정보를 시각적으로 확인합니다. 
#이는 이미지의 어떤 부분이 각 기본 색상(파랑, 초록, 빨강)에 얼마나 기여하는지 이해하는 도움.
#cv2.split() & cv2.merge(): 채널을 분리했다가 다시 합치는 기본 구조를 배웁니다.

#단일 채널 시각화: cv2.merge((b, zeros, zeros))처럼 특정 채널 외에 모두 0으로 채우면, 
#해당 채널의 픽셀 값(밝기)이 그 색상으로 얼마나 표현되는지 시각적으로 확인할 수 있습니다.]

import cv2
import numpy as np

file_path = './images/sample2.jpg'
img_color = cv2.imread(file_path)

if img_color is None:
    print(f"오류: '{file_path}' 파일을 로드할 수 없습니다.")
    exit()

# 1. 채널 분리
# cv2.split()을 사용해 B, G, R 채널을 3개의 2차원 배열로 분리합니다.
b, g, r = cv2.split(img_color)

# 2. 개별 채널을 3채널 이미지로 복원하여 시각화 (선택적)
# 특정 채널만 켜고 나머지 채널을 0(검은색)으로 채워 해당 채널의 색상 기여도를 보여줍니다.
#img_color - 3차원임 height, width, channel
zeros = np.zeros(img_color.shape[:2], dtype=img_color.dtype) # 0으로 채워진 2차원 배열 생성

img_blue = cv2.merge((b, zeros, zeros))  # 파란색 채널만 남김
img_green = cv2.merge((zeros, g, zeros)) # 초록색 채널만 남김
img_red = cv2.merge((zeros, zeros, r))   # 빨간색 채널만 남김

# 3. 결과 이미지 표시
cv2.imshow('Original', img_color)
cv2.imshow('Blue Channel Visual', img_blue)
cv2.imshow('Green Channel Visual', img_green)
cv2.imshow('Red Channel Visual', img_red)
cv2.imshow('B Channel (Gray Scale)', b) # 순수 흑백 B 채널

cv2.waitKey(0)
cv2.destroyAllWindows()