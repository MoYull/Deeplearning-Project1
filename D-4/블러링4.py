import cv2
import numpy as np

# ----------------- 1. 설정 및 이미지 생성 -----------------

# 500x500 크기의 더미 이미지 생성 (예시를 위해 초기화된 이미지)
img = np.zeros((500, 500, 3), dtype=np.uint8)
img[:] = (180, 250, 180) # 연한 녹색 배경

# 민감 정보가 있다고 가정한 'ROI (Region of Interest)' 정의
# 중앙 사각형 영역 (예: 얼굴 위치)
x_start, y_start = 150, 150
x_end, y_end = 350, 350
roi_w = x_end - x_start
roi_h = y_end - y_start

# '민감 정보'가 있음을 표시 (흰색 사각형)
cv2.rectangle(img, (x_start, y_start), (x_end, y_end), (255, 255, 255), -1) 
cv2.putText(img, "Sensitive Info Here", (x_start + 10, y_start + 50), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)


# --- 2. 블러링을 이용한 프라이버시 보호 ---
img_blur = img.copy()

# 2-1. ROI 영역 추출
roi_blur = img_blur[y_start:y_end, x_start:x_end]

# 2-2. 가우시안 블러 적용 (커널 크기가 클수록 더 강하게 흐려짐)
#블러링을 적용할 주변 픽셀 영역의 크기
# 커널 크기는 홀수여야 합니다. 여기서는 (51, 51)을 사용
blurred_roi = cv2.GaussianBlur(roi_blur, (51, 51), 0)

# 2-3. 처리된 ROI를 원본 이미지에 다시 삽입
img_blur[y_start:y_end, x_start:x_end] = blurred_roi


# --- 3. 모자이크 (픽셀화)를 이용한 프라이버시 보호 ---
img_mosaic = img.copy()

# 3-1. ROI 영역 추출
roi_mosaic = img_mosaic[y_start:y_end, x_start:x_end]

# 3-2. 픽셀화를 위한 축소 및 확대 과정
# 축소 비율 (값이 작을수록 모자이크가 굵어짐)
scale = 0.1 
w_small = int(roi_w * scale)
h_small = int(roi_h * scale)

# 축소: 이미지를 작은 크기로 줄여 픽셀 정보를 뭉개버립니다.
img_small = cv2.resize(roi_mosaic, (w_small, h_small), 
                       interpolation=cv2.INTER_LINEAR)

# 확대: 다시 원래 크기로 확대하여 픽셀을 블록화(모자이크)합니다.
img_mosaic_processed = cv2.resize(img_small, (roi_w, roi_h), 
                                  interpolation=cv2.INTER_NEAREST) 
# INTER_NEAREST는 인접 보간법으로, 픽셀을 뭉개진 사각형 블록 형태로 유지합니다.

# 3-3. 처리된 ROI를 원본 이미지에 다시 삽입
img_mosaic[y_start:y_end, x_start:x_end] = img_mosaic_processed


# --- 4. 결과 출력 및 비교 ---
cv2.putText(img, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
cv2.putText(img_blur, "Blurred (Privacy)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
cv2.putText(img_mosaic, "Mosaic (Pixelated)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

# 원본, 블러, 모자이크 이미지를 수평으로 합쳐서 비교
img_final = np.hstack([img, img_blur, img_mosaic])

cv2.imshow('Privacy Protection Techniques', img_final)
cv2.waitKey(0)
cv2.destroyAllWindows()