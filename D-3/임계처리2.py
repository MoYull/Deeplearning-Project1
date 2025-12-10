import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# --- 1. 파일 경로 설정 및 이미지 로드 (사용자 수정 필요) ---
# 예제를 실행하려면 'your_fun_photo.jpg' 경로를 실제 이미지 경로로 변경하세요.
IMAGE_PATH = './images/sample2.jpg'

# 안전을 위해 실제 파일 로드 대신 더미 이미지 생성 코드를 사용하며,
# 사용자는 아래의 'img_original' 로드 부분을 실제 이미지 로드로 교체해야 합니다.

try:
    # 사용자 이미지 로드 (실제 사용할 때 이 줄의 주석을 해제하고 경로를 수정하세요)
    img_original = cv2.imread(IMAGE_PATH)
    
    # [데모용 더미 이미지 생성: 실제 코드를 실행할 때는 위 1줄만 남기고 제거하세요]
    # h, w = 400, 600
    # np.random.seed(42)
    # # 무작위 픽셀로 구성된 3채널 이미지 생성 (명도 레벨 구분이 잘 보이도록)
    # dummy_data = np.uint8(np.linspace(0, 255, h * w).reshape(h, w)) 
    # dummy_data = cv2.cvtColor(dummy_data, cv2.COLOR_GRAY2BGR)
    # img_original = dummy_data 
    
    if img_original is None:
        print(f"오류: 이미지 파일 '{IMAGE_PATH}'를 로드할 수 없습니다.")
        exit()

except Exception as e:
    print(f"이미지 로드 중 오류 발생: {e}")
    exit()


# --- 2. 팝아트 설정: 임계값과 색상 정의 ---
# 명도 레벨을 나눌 4개의 임계값 (0~255)
THRESHOLDS = [50, 100, 150, 200]

# 명도 레벨에 매칭될 5가지 팝아트 색상 (BGR 순서: OpenCV 표준)
# [B, G, R]
POP_COLORS = [
    [0, 0, 0],         # Level 1: 검은색 (가장 어두운 영역)
    [255, 0, 255],     # Level 2: 마젠타 (중간 어두운 영역)
    [255, 255, 0],     # Level 3: 시안 (중간 밝은 영역)
    [0, 255, 255],     # Level 4: 노란색 (밝은 영역)
    [255, 255, 255]    # Level 5: 흰색 (가장 밝은 영역)
]

# --- 3. 이미지 전처리 및 팝아트 생성 ---

# 팝아트 효과를 위해 흑백(Grayscale) 명도 정보만 사용합니다.
img_gray = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)

# 최종 팝아트 결과 이미지를 담을 빈 BGR 배열 생성
img_pop_art = np.zeros_like(img_original, dtype=np.uint8)

# --------------------------------------------------------
# 임계 처리 응용: 다단계 명도 영역 추출 및 색상 칠하기
# --------------------------------------------------------

# 1. 0 ~ T1 (50) 영역 (가장 어두운 Level 1)
# 50을 기준으로 반전 임계 처리(THRESH_BINARY_INV): 50 이하 영역이 마스크(흰색)가 됨
_, mask_level1 = cv2.threshold(img_gray, THRESHOLDS[0], 255, cv2.THRESH_BINARY_INV) 

# 해당 마스크 영역에 Level 1 색상(POP_COLORS[0])을 칠함
color_plane = np.full(img_original.shape, POP_COLORS[0], dtype=np.uint8)
img_pop_art = cv2.bitwise_or(img_pop_art, cv2.bitwise_and(color_plane, color_plane, mask=mask_level1))

# 2. T1 ~ T4 사이의 영역 (Level 2, 3, 4)
for i in range(len(THRESHOLDS) - 1):
    T_low = THRESHOLDS[i]      # 명도 하한 (예: 50)
    T_high = THRESHOLDS[i+1]   # 명도 상한 (예: 100)
    
    # 1) T_low를 초과하는 모든 영역 마스크 (예: 50 초과)
    _, mask_T_low = cv2.threshold(img_gray, T_low, 255, cv2.THRESH_BINARY)
    
    # 2) T_high를 초과하는 모든 영역 마스크 (예: 100 초과)
    _, mask_T_high = cv2.threshold(img_gray, T_high, 255, cv2.THRESH_BINARY)

    # 3) 영역 추출: (mask_T_low) - (mask_T_high) = [T_low 초과 ~ T_high 이하] 영역 마스크
    region_mask = cv2.subtract(mask_T_low, mask_T_high)
    
    # 해당 영역에 Level (i+1) 색상(POP_COLORS[i+1])을 칠함
    color_plane = np.full(img_original.shape, POP_COLORS[i+1], dtype=np.uint8)
    masked_region = cv2.bitwise_and(color_plane, color_plane, mask=region_mask)
    img_pop_art = cv2.bitwise_or(img_pop_art, masked_region)

# 3. T4 (200) 초과 영역 (가장 밝은 Level 5)
# 200 초과 영역이 마스크(흰색)가 됨
_, mask_final = cv2.threshold(img_gray, THRESHOLDS[-1], 255, cv2.THRESH_BINARY)
color_plane = np.full(img_original.shape, POP_COLORS[-1], dtype=np.uint8)

# Level 5 색상(흰색)을 칠하고 최종 결과에 합침
img_pop_art = cv2.bitwise_or(img_pop_art, cv2.bitwise_and(color_plane, color_plane, mask=mask_final))


# --- 4. 결과 시각화 ---
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.title('Original Image (Grayscale Used)')
# BGR -> RGB 변환 후 시각화
plt.imshow(cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)) 

plt.subplot(1, 2, 2)
plt.title('Pop Art Thresholding Effect (5 Levels)')
# BGR -> RGB 변환 후 시각화
plt.imshow(cv2.cvtColor(img_pop_art, cv2.COLOR_BGR2RGB)) 

plt.show()