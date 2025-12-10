#색상 채널 분리 및 재결합
#컬러 이미지를 B, G, R 세 개의 개별 흑백 이미지(채널)로 분리하고, 특정 채널만 조작하거나 재결합하는 방법을 보여줍니다.

#cv2.split(): 3차원 배열을 3개의 2차원 배열(채널)로 나눕니다.
#b[:] = 255: NumPy 슬라이싱을 사용하여 파란색 채널 전체의 모든 픽셀 값을 최댓값(255)으로 설정합니다.
#cv2.merge(): 분리된 2차원 채널들을 다시 3차원 컬러 이미지로 합쳐서 결과를 시각화합니다.

import cv2

file_path = './images/sample2.jpg'
img_color = cv2.imread(file_path)

if img_color is None:
    print(f"오류: '{file_path}' 파일을 로드할 수 없습니다.")
    exit()

# 1. 채널 분리
# cv2.split() 함수를 사용하여 B, G, R 3개의 2차원 배열로 분리합니다.
b, g, r = cv2.split(img_color)

# 2. 특정 채널 조작 (파란색 채널 픽셀 값 255로 모두 설정)
# 파란색 채널이 완전히 켜지면 이미지 전체에 푸른색이 강해집니다.
b[:] = 255 

# 3. 채널 재결합
# cv2.merge() 함수를 사용하여 분리된 채널을 다시 컬러 이미지로 합칩니다.
img_merged = cv2.merge( (b, g, r) )

# 4. 결과 이미지 표시
cv2.imshow('Original Image', img_color)
cv2.imshow('B Channel Maxed Out', img_merged)
# 개별 채널 확인 (흑백으로 보임)
cv2.imshow('Blue Channel Only', b) 

cv2.waitKey(0)
cv2.destroyAllWindows()


