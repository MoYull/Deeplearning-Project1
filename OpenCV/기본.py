#cv2를 이용해 이미지를 읽고 출력하는 예제 
import cv2
import numpy as np
import os # 파일 존재 확인용

# 1. 사용할 이미지 파일 경로 설정
file_path = './images/sample.jpg'  

# 2. 이미지 읽기 (컬러: 기본값)
# 이미지 파일이 NumPy 배열로 로드됩니다.
img_color = cv2.imread(file_path)  #이미지 파일 경로를 준다
print(type (img_color)) #타입도 확인 - ndarray 임 
print(f"원본 이미지 shape: {img_color.shape}")
    
# 3. 이미지 표시
# 'Color Image'라는 이름의 창에 원본 이미지를 표시합니다.
cv2.imshow('Color Image', img_color) 

# 4. 키 입력 대기
# '0'은 키 입력이 있을 때까지 무한정 대기하라는 의미입니다.
cv2.waitKey(0)

# 5. 모든 창 닫기
cv2.destroyAllWindows()