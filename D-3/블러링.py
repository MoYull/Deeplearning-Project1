"""
1. 노이즈 제거 및 이미지 개선 (Noise Reduction)
가장 일반적인 목적은 이미지나 비디오에 포함된 잡음(노이즈)을 줄이는 것입니다.

원리: 블러링 필터(예: 가우시안 블러, 평균 필터)는 주변 픽셀들의 평균값을 사용하여 중심 픽셀 값을 대체합니다.
 노이즈는 보통 주변 픽셀과 값이 크게 다른 '튀는' 픽셀들이므로, 평균화 과정을 통해 이러한 노이즈가 주변의 부드러운 값에 섞여 
 희석되고 제거됩니다.

효과: 이미지의 전반적인 품질을 향상시키고, 이후의 복잡한 이미지 처리(예: 에지 검출, 객체 인식) 과정에서 
오류를 줄여 처리의 안정성을 높입니다.

2. 세부 정보 평활화 및 특징 추출 전처리 (Feature Preprocessing)
이미지의 불필요하거나 과도하게 미세한 세부 정보를 제거하고 중요한 특징을 부각하기 위한 전처리 과정입니다.

세부 정보 제거: 작은 텍스처나 미세한 주름 등의 고주파수(High-Frequency) 성분을 제거하여 이미지를 평활화(Smoothing)
시킵니다.

에지 검출: 에지(가장자리)를 검출하기 전에 블러링을 적용하면 노이즈로 인한 잘못된 에지 검출을 방지하고 주요하고 
명확한 에지만을 남길 수 있습니다. (예: 캐니 에지 검출 알고리즘은 전처리 단계로 가우시안 블러를 사용합니다.)

3. 프라이버시 보호 및 민감 정보 가리기 (Privacy and Security)
특정 영역의 정보를 의도적으로 식별할 수 없게 만듦으로써 개인의 프라이버시를 보호합니다.

민감 정보 마스킹: 사람의 얼굴, 차량 번호판, 문서에 포함된 개인 정보 등 민감하거나 식별 가능한 부분을 블러 처리하여 
정보 유출을 방지합니다.

예시: CCTV나 거리 뷰 서비스에서 얼굴이나 번호판을 모자이크 또는 블러 처리하는 것이 대표적인 예입니다.
"""

import cv2
import numpy as np

file_path = './images/sample2.jpg' 
img = cv2.imread(file_path)

if img is None:
    print("이미지 로드 실패")
    exit()

# 1. 가우시안 블러 (노이즈 제거)
# 5x5 커널, sigmaX=0 (자동 계산)
img_gaussian = cv2.GaussianBlur(img, (5, 5), 0)

#본래의 정보와는 전혀 상관없는 밝거나 어두운 점들이 무작위로 흩뿌려진 것처럼 보입니다.
# 2. 미디언 블러 (소금-후추 노이즈에 효과적)
# 5x5 커널 (홀수)
img_median = cv2.medianBlur(img, 5)

# 3. 샤프닝 (엣지 강조)
# 샤프닝 커널 정의 (NumPy float32 타입 필수)
kernel_sharpen = np.array([
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]
], dtype=np.float32)

# cv2.filter2D를 사용하여 커널 적용
# -1은 입력과 동일한 깊이(채널 및 타입)를 사용하라는 의미
img_sharpened = cv2.filter2D(img, -1, kernel_sharpen)


# 4. 결과 이미지 표시
cv2.imshow('Original', img)
cv2.imshow('Gaussian Blur', img_gaussian)
cv2.imshow('Median Blur', img_median)
cv2.imshow('Sharpened', img_sharpened)

cv2.waitKey(0)
cv2.destroyAllWindows()