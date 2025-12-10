import cv2
import numpy as np
import os

# 사용할 이미지 파일 경로 설정
file_path = './images/sample2.jpg'

if not os.path.exists(file_path):
    print(f"오류: '{file_path}' 파일을 찾을 수 없습니다. 테스트 이미지를 다운로드하여 저장해주세요.")
    exit() #프로그램 종료하기 

# 1. 이미지 읽기 (컬러: 기본값)
img = cv2.imread(file_path)

if img is None:
    print("오류: 이미지를 로드할 수 없습니다.")
    exit()

# 이미지 복사 (원본 보호를 위해)
img_copy = img.copy()

# -----------------------------------------------------------
# 2. 픽셀 값 접근 및 조작 (특정 지점의 BGR 값 변경)
# 100행, 200열 픽셀의 B, G, R 값을 변경합니다.
# OpenCV는 BGR 순서입니다.
# B - 0, G -1, R-2 
r_val = img_copy[100, 200, 2] # 원본 R 값 확인
print(r_val)
img_copy[100, 200] = [0, 0, 255] # 픽셀을 순수한 빨간색으로 변경 (B=0, G=0, R=255)
print(f"픽셀 (100, 200)의 원본 R 값: {r_val}")
print(f"픽셀 (100, 200)의 변경된 BGR 값: {img_copy[100, 200]}")


# #특정 영역의 값을 빨간색으로 바꿔보자
# img_copy[100:200, 200:300] = [0, 0, 255] # 픽셀을 순수한 빨간색으로 변경 (B=0, G=0, R=255)
# cv2.imshow('Processed Image with ROI', img_copy)

# # 5. 결과 이미지 표시
cv2.imshow('Processed Image with ROI', img_copy)

# 3. ROI (Region of Interest) 추출, roi 란 분석하고 처리해야 할 중요한 정보를 포함하는 영역의 경계
# NumPy 슬라이싱을 사용하여 이미지의 일부 영역을 추출합니다.
roi = img[100:300, 50:250]  #일부영역을 추출한다 
print(f"ROI 영역 shape: {roi.shape}")

# 4. ROI를 다른 영역에 복사 (마스킹 또는 대체)
# 추출한 ROI를 원본 이미지의 다른 위치에 덮어씁니다.
# ROI 크기와 같아야 함
img_copy[400:600, 300:500] = roi

# # 5. 결과 이미지 표시
cv2.imshow('Original Image', img)
cv2.imshow('Processed Image with ROI', img_copy)

# 키 입력 대기 및 창 닫기
cv2.waitKey(0)
cv2.destroyAllWindows()
