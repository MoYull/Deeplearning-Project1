import cv2
import numpy as np

# 이미지 경로
file_path = './images/Ham.jpg'
img = cv2.imread(file_path)

if img is None:
    print("이미지 로드 실패")
    exit()

# 1. 그레이스케일 변환
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 2. 세 가지 블러 적용 (커널 5x5)
blur_normal = cv2.blur(gray, (5, 5))                # 평균 블러
blur_gaussian = cv2.GaussianBlur(gray, (5, 5), 0)   # 가우시안 블러
blur_median = cv2.medianBlur(gray, 5)               # 미디언 블러

# 3. 가로로 이미지 합치기 (모두 크기가 동일해야 함)
# gray는 단일 채널이므로 merge로 3채널로 변환해주면 보기 좋음
gray_3c = cv2.merge([gray, gray, gray])
blur_normal_3c = cv2.merge([blur_normal, blur_normal, blur_normal])
blur_gaussian_3c = cv2.merge([blur_gaussian, blur_gaussian, blur_gaussian])
blur_median_3c = cv2.merge([blur_median, blur_median, blur_median])

# 4개 이미지를 가로로 연결
result = cv2.hconcat([gray_3c, blur_normal_3c, blur_gaussian_3c, blur_median_3c])

# 4. 출력
cv2.imshow("Gray | Mean Blur | Gaussian | Median", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
