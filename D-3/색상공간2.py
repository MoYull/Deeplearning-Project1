#원본 이미지 준비 및 BGR → HSV 변환
#HSV 색상 원에서 빨간색(Red)은 0~360(OpenCV H에서는 0과 179)의 양 끝에 걸쳐 있습니다.
#범위 설정: 따라서 빨간색을 온전히 포착하기 위해 두 개의 하한(Lower) 및 상한(Upper) 배열을 정의합니다. 

import cv2
import numpy as np

# 1. 빨간색 객체를 포함하는 가상 이미지 생성
img = np.zeros((400, 400, 3), dtype=np.uint8)
# 중앙에 밝고 선명한 빨간색 원을 그립니다. (BGR 순서: B=0, G=0, R=255)
cv2.circle(img, (200, 200), 80, (0, 0, 255), -1)

## 2. BGR -> HSV 색상 공간 변환
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

## 3. 빨간색 범위 정의 (HSV 값)
# OpenCV에서 H(Hue)는 0-179 범위를 가집니다. 빨간색은 색상 원의 양 끝에 걸쳐 있습니다 (0 근처와 179 근처).
# 따라서 두 개의 범위를 설정하여 두 영역을 모두 포착해야 합니다.
# 

# 첫 번째 빨간색 범위 (주변 0)                  색상  명도  채도
lower_red1 = np.array([0, 100, 100])
upper_red1 = np.array([10, 255, 255])

# 두 번째 빨간색 범위 (주변 179)
lower_red2 = np.array([160, 100, 100])
upper_red2 = np.array([179, 255, 255])

## 4. 임계 처리 (Thresholding)를 통한 마스크 생성
# cv2.inRange(src, lowerb, upperb) 함수는 특정 범위에 속하는 픽셀만 흰색(255)으로, 
# 나머지는 검은색(0)으로 만듭니다.

# 첫 번째 범위 마스크
mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
# 두 번째 범위 마스크
mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

# 두 마스크를 OR 연산하여 최종 빨간색 마스크 생성 (두 범위 중 하나라도 포함되면 빨간색)
final_mask = cv2.bitwise_or(mask1, mask2)

## 5. 마스크를 원본 이미지에 적용하여 ROI 추출
# cv2.bitwise_and(src1, src2, mask)는 마스크 영역(흰색 부분)에 해당하는 원본 이미지 부분만 보여줍니다.
red_roi = cv2.bitwise_and(img, img, mask=final_mask)

## 6. 결과 출력
cv2.imshow('1. Original Image', img)
cv2.imshow('2. Final Red Mask', final_mask)
cv2.imshow('3. Red ROI Extracted', red_roi)

cv2.waitKey(0)
cv2.destroyAllWindows()
