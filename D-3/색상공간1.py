import cv2
import numpy as np

# 1. 이미지 파일 경로 설정 (사용자의 이미지 경로로 변경하세요)
# 예시로 OpenCV에서 제공하는 가짜 이미지 데이터를 생성합니다.
# 실제 사용 시에는 cv2.imread('your_image_path.jpg')를 사용하세요.
img = np.zeros((300, 300, 3), dtype=np.uint8)
# 이미지 중앙에 파란색 사각형을 그려 BGR 이미지를 만듭니다.
cv2.rectangle(img, (100, 100), (200, 200), (255, 0, 0), -1) # B=255, G=0, R=0 (파란색)

# 2. BGR 이미지를 HSV 이미지로 변환
# cvtColor 함수를 사용하여 색상 공간을 변환합니다.
hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# 3. 결과 이미지 출력
cv2.imshow('Original BGR Image', img)
cv2.imshow('Converted HSV Image', hsv_img)


# 4. 종료 대기
# 아무 키를 누를 때까지 창을 유지
cv2.waitKey(0)
cv2.destroyAllWindows()
