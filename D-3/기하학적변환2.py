import cv2
import numpy as np
import os

file_path = './images/sample2.jpg'
img = cv2.imread(file_path)

if img is None:
    print(f"오류: '{file_path}' 파일을 로드할 수 없습니다.")
    exit()

# 이미지의 높이(h)와 너비(w) 가져오기
h, w = img.shape[:2]

# -----------------------------------------------------------
# 1. 이미지 이동 (Translation)

# 이동 행렬 M 생성 (오른쪽으로 100픽셀, 아래로 50픽셀 이동)
# M = [[1, 0, Tx], [0, 1, Ty]]
Tx, Ty = 100, 50
M_translation = np.float32([
    [1, 0, Tx],
    [0, 1, Ty]
])

# cv2.warpAffine() 함수를 사용하여 변환 적용
# (원본 이미지, 변환 행렬, 출력 크기(너비, 높이))
img_translated = cv2.warpAffine(img, M_translation, (w, h))

# -----------------------------------------------------------
# 2. 이미지 회전 (Rotation)

# 회전 중심점 설정 (이미지의 중앙)
center = (w // 2, h // 2)
angle = 45  # 시계 반대 방향으로 45도 회전
scale = 1.0 # 크기 유지

# 회전 행렬 M_rotation 계산
M_rotation = cv2.getRotationMatrix2D(center, angle, scale)

# 변환 적용
img_rotated = cv2.warpAffine(img, M_rotation, (w, h))

# -----------------------------------------------------------
# 3. 결과 이미지 표시

cv2.imshow('Original', img)
cv2.imshow('Translated (R+100, D+50)', img_translated)
cv2.imshow('Rotated (45 deg)', img_rotated)

cv2.waitKey(0)
cv2.destroyAllWindows()