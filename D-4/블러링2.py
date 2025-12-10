import cv2
import numpy as np

# --- 1. 이미지 로드 및 전처리 ---

# 가상의 이미지 (예: 흰색 배경에 검은색 사각형)를 생성합니다.
# 실제 파일 사용 시: img_original = cv2.imread('your_image_path.jpg')
img_original = np.zeros((400, 600, 3), dtype=np.uint8)
img_original[:] = (255, 255, 255) # 흰색 배경
cv2.rectangle(img_original, (100, 100), (500, 300), (50, 50, 50), -1) # 중앙에 어두운 사각형

# 엣지 검출을 위해 BGR 이미지를 회색조(Grayscale)로 변환
img_gray = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)


# --- 2. 블러링 적용 (가우시안 블러 사용) ---

# 3x3 가우시안 커널을 사용하여 이미지를 부드럽게 만듭니다.
# 노이즈를 감소시켜 Canny가 더 깨끗한 엣지를 잡도록 준비합니다.
# (커널 크기가 클수록 더 많이 흐려집니다. 여기서는 (3, 3)을 사용)
img_blurred = cv2.GaussianBlur(img_gray, (3, 3), 0)


# --- 3. 엣지 검출 및 결과 비교 ---

# 3-1. 블러링 적용 X: 원본 회색조 이미지에 Canny 적용
# 노이즈가 있다면 그대로 엣지로 검출될 가능성이 높습니다.
canny_original = cv2.Canny(img_gray, 50, 150)

# 3-2. 블러링 적용 O: 블러 처리된 이미지에 Canny 적용 (권장되는 방법)
# 노이즈가 제거되어 더 깨끗하고 명확한 엣지만 남습니다.
canny_blurred = cv2.Canny(img_blurred, 50, 150)


# --- 4. 결과 출력 ---

# 비교를 위해 이미지를 수직으로 합칩니다.
# 두 결과는 모두 단일 채널(흑백) 이미지이므로, np.hstack을 사용하기 위해 차원을 맞춰줄 필요는 없습니다.

# 원본, 블러, Canny-원본, Canny-블러 순으로 한 창에 표시
img_combined_top = np.hstack([img_gray, img_blurred])
img_combined_bottom = np.hstack([canny_original, canny_blurred])
img_combined = np.vstack([img_combined_top, img_combined_bottom])

# 텍스트를 추가하여 각 결과가 무엇인지 표시합니다 (OpenCV 기본 텍스트 사용).
cv2.putText(img_combined, "1. Original Grayscale", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
cv2.putText(img_combined, "2. Gaussian Blurred", (620, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
cv2.putText(img_combined, "3. Canny on Original (Noisy)", (20, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
cv2.putText(img_combined, "4. Canny on Blurred (Clean)", (620, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)


cv2.imshow("Blurring Before Edge Detection Comparison", img_combined)
cv2.waitKey(0)
cv2.destroyAllWindows()

