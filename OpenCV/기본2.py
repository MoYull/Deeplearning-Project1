#이미지를 찾지 못했을때 예외처리를 하는 예제
import cv2
import numpy as np
import os # 파일 존재 확인용

# 1. 사용할 이미지 파일 경로 설정
file_path = './images/sample2.jpg' 

# 파일이 존재하는지 확인 (없으면 에러 방지)

if not os.path.exists(file_path):
    print(f"오류: '{file_path}' 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
    # 임의의 검은색 이미지 생성 (파일이 없을 경우)
    #np.zeros - 매개변수가 tuple여야 한다. 300 by 400 by 3 크기의 이미지를 만들고 값을 0으로 초기화 한다 
    #색상은 검정이 된다 값이 0으로 구성되어 있으므로 
    img_color = np.zeros((300, 400, 3), dtype=np.uint8)
    cv2.putText(img_color, "No Image Found", (50, 150), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, 
                (255, 255, 255), 2)
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

# 4. 키 입력 대기
# '0'은 키 입력이 있을 때까지 무한정 대기하라는 의미입니다.
cv2.waitKey(0)

# 5. 모든 창 닫기
cv2.destroyAllWindows()