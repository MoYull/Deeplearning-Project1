#pip install --upgrade matplotlib numpy

import cv2
import numpy as np
from matplotlib import pyplot as plt

file_path = './images/sample2.jpg' 
img_color = cv2.imread(file_path)

if img_color is None:
    print("이미지 로드 실패")
    exit()

# 1. 흑백 이미지로 변환
img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

# 2. 히스토그램 계산 (흑백 이미지)  #0-채널번호, None-특정영역만 계산하고 싶을때
# 256: 8비트 이미지(0~255)의 경우, 픽셀 값 1마다 1개의 막대(Bin)를 할당하여 가장 상세한 히스토그램을 만듭니다 
#[0, 256]: 0부터 256 미만, 즉 0부터 255까지의 모든 픽셀 값을 히스토그램에 포함합니다.
#img_gray 이미지 전체를 대상으로, 0번 채널(유일한 채널)의 0부터 255까지의 픽셀 값을 256개의 구간(Bin)으로 나누어 
# 그 분포(픽셀 개수)를 계산하라는 명령입니다.
hist = cv2.calcHist([img_gray], [0], None, [256], [0, 256])

# 3. 히스토그램 평활화 적용
img_equalized = cv2.equalizeHist(img_gray)

# 4. 평활화된 이미지의 히스토그램 계산
hist_eq = cv2.calcHist([img_equalized], [0], None, [256], [0, 256])

# 5. 결과 이미지 표시
cv2.imshow('Original Gray', img_gray)
cv2.imshow('Equalized Image', img_equalized)

# 6. 히스토그램 시각화 (matplotlib 사용)
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(hist, color='black')
plt.title('Original Histogram')
plt.xlabel('Pixel Value (0-255)')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
plt.plot(hist_eq, color='black')
plt.title('Equalized Histogram')
plt.xlabel('Pixel Value (0-255)')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show() # matplotlib 창 표시

cv2.waitKey(0)
cv2.destroyAllWindows()