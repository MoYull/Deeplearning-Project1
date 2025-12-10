#이미지의 모든 픽셀 값에 특정 상수를 더하거나 곱하여 이미지 전체의 밝기를 조절하는 방법을 보여줍니다. 
#이는 픽셀 단위의 연산을 NumPy의 벡터화된 연산으로 처리하여 매우 빠르게 수행합니다.
#벡터화 연산: cv2.add()나 cv2.subtract()를 사용하면 for 반복문 없이 이미지 전체의 픽셀 값을 한 번에 연산할 수 있어 속도가 빠릅니다.
#포화 연산 (Saturation): 픽셀 값의 범위인 0~255를 벗어나는 값은 자동으로 0이나 255로 조정됩니다.

import cv2
import numpy as np

file_path = './images/sample2.jpg'
img = cv2.imread(file_path)

if img is None:
    print(f"오류: '{file_path}' 파일을 로드할 수 없습니다.")
    exit()

# 1. 원본 이미지의 복사본 준비
img_bright = img.copy()
img_dark = img.copy()

# 2. 밝기 증가 (모든 픽셀에 +50 더하기)
# cv2.add() 함수는 픽셀 값이 255를 초과할 경우 자동으로 255로 잘라줍니다 (포화 연산).
increase_value = 50
img_bright = cv2.add(img, increase_value) 

# 3. 밝기 감소 (모든 픽셀에 -50 빼기)
# cv2.subtract() 함수는 픽셀 값이 0 미만일 경우 자동으로 0으로 잘라줍니다.
decrease_value = 50
img_dark = cv2.subtract(img, decrease_value)

# 4. 결과 이미지 표시
cv2.imshow('Original', img)
cv2.imshow('Brightness Increased (+50)', img_bright)
cv2.imshow('Brightness Decreased (-50)', img_dark)

cv2.waitKey(0)
cv2.destroyAllWindows()