#텍스트를 출력하고 흑백으로 변동하는 예제                        
import cv2
import numpy as np
import os # 파일 존재 확인용

# 1. 사용할 이미지 파일 경로 설정
file_path = './images/sample233.jpg' 

# 파일이 존재하는지 확인 (없으면 에러 방지)
if not os.path.exists(file_path):
    print(f"오류: '{file_path}' 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
    # 임의의 검은색 이미지 생성 (파일이 없을 경우)
    img_color = np.zeros((300, 1000, 3), dtype=np.uint8)
    cv2.putText(img_color, "No Image Found", 
                (50, 150), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                3,  #폰트크기
                (255, 255, 255), 
                8)  #폰트굵기
    file_name_out = 'output_no_file.jpg'
else:
    # 2. 이미지 읽기 (컬러: 기본값)
    # 이미지 파일이 NumPy 배열로 로드됩니다.
    img_color = cv2.imread(file_path)
    print(f"원본 이미지 shape: {img_color.shape}")
    file_name_out = 'output_gray.jpg'
    
# 3. 이미지 표시
# 'Color Image'라는 이름의 창에 원본 이미지를 표시합니다.
cv2.imshow('Color Image', img_color) 


####################################################
#BGR-blue green red 임 
# 4. 색상 공간 변환 (BGR -> Gray Scale)
# 컬러 이미지를 흑백 이미지로 변환합니다. cvtColor
img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
print(f"흑백 이미지 shape: {img_gray.shape}") # 3차원에서 2차원으로 변경됨

# 5. 흑백 이미지 표시
cv2.imshow('Gray Scale Image', img_gray)

# 6. 키 입력 대기
# '0'은 키 입력이 있을 때까지 무한정 대기하라는 의미입니다.
cv2.waitKey(0)

# 7. 흑백 이미지 저장
cv2.imwrite(file_name_out, img_gray)
print(f"흑백 이미지 저장 완료: {file_name_out}")

# 8. 모든 창 닫기
cv2.destroyAllWindows()