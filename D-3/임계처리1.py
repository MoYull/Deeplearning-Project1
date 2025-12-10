import cv2
import numpy as np

file_path = './images/sample2.jpg'
img_color = cv2.imread(file_path)

if img_color is None:
    print(f"오류: '{file_path}' 파일을 로드할 수 없습니다.")
    exit()

# 1. BGR -> Gray Scale 변환
img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

# 2. 단순 임계 처리 (Threshold = 127)
# 픽셀 값이 127보다 크면 255(흰색), 아니면 0(검은색)
ret, img_binary = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)

# 3. 오츠 임계 처리 (자동 기준값)
# ret_otsu 변수에 최적의 기준값이 저장됨
ret_otsu, img_otsu = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# 4. 결과 이미지 표시
cv2.imshow('Original Color', img_color)
cv2.imshow('Gray Scale', img_gray)
cv2.imshow('Simple Binary (T=127)', img_binary)
cv2.imshow(f'Otsu Binary (T={ret_otsu:.0f})', img_otsu)

cv2.waitKey(0)
cv2.destroyAllWindows()