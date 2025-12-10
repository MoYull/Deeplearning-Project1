import cv2
import numpy as np

# 1. 이미지 파일 로드
image_path = './images/sample2.jpg' 
img = cv2.imread(image_path)

if img is None:
    print(f"오류: '{image_path}' 이미지를 로드할 수 없습니다.")
    print("이미지 파일이 존재하는지, 경로가 올바른지 확인해주세요.")
    exit()

# 이미지의 높이(h), 너비(w)
(h, w) = img.shape[:2]

# 2. 변환 행렬 (Transformation Matrix) 생성
# 회전 변환 행렬 M을 생성합니다.
# cv2.getRotationMatrix2D(center, angle, scale)
# - center: 회전의 중심점 (이미지의 중앙)
# - angle: 회전 각도 (시계 반대 방향이 양수, 여기서는 -45도로 시계 방향 회전)
# - scale: 크기 비율 (0.6배로 축소)
center = (w // 2, h // 2)
angle = -45  # 시계 방향으로 45도 회전
scale = 0.6  # 60% 크기로 축소

M = cv2.getRotationMatrix2D(center, angle, scale)
# M은 2x3 크기의 변환 행렬입니다. 

# 3. 변환 적용 (Warping)
# cv2.warpAffine(src, M, dsize)
# - src: 원본 이미지
# - M: 2x3 변환 행렬
# - dsize: 출력 이미지의 크기 (원본과 동일한 크기로 지정)
img_transformed = cv2.warpAffine(img, M, (w, h))

# 4. 결과 출력
cv2.imshow('1. Original Image', img)
cv2.imshow('2. Affine Transformed (Rotated and Scaled)', img_transformed)

cv2.waitKey(0)
cv2.destroyAllWindows()