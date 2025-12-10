import cv2
import numpy as np

# 전역 변수
drawing = False
mode = None     # 'circle' or 'line'
ix, iy = -1, -1

# 500x500 검은색 배경
img = np.zeros((500, 500, 3), np.uint8)

# 마우스 콜백 함수
def draw_shape(event, x, y, flags, param):
    global ix, iy, drawing, mode, img

    # ---- 왼쪽 버튼 눌렀을 때 → 원 그리기 모드 시작 ----
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        mode = 'circle'
        ix, iy = x, y

    # ---- 오른쪽 버튼 눌렀을 때 → 선 그리기 모드 시작 ----
    elif event == cv2.EVENT_RBUTTONDOWN:
        drawing = True
        mode = 'line'
        ix, iy = x, y

    # ---- 마우스 이동 중 (드래그 중) → 실시간 그리기 ----
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            img_temp = img.copy()  # 원본 보존
            if mode == 'circle':
                # 반지름 계산
                radius = int(np.sqrt((x - ix)**2 + (y - iy)**2))
                cv2.circle(img_temp, (ix, iy), radius, (255, 0, 0), 2)  # 파란 원
            elif mode == 'line':
                cv2.line(img_temp, (ix, iy), (x, y), (0, 0, 255), 2)  # 빨간 선
            cv2.imshow('Image', img_temp)

    # ---- 마우스 버튼 떼었을 때 → 도형 완성 (원본에 저장) ----
    elif event == cv2.EVENT_LBUTTONUP or event == cv2.EVENT_RBUTTONUP:
        drawing = False
        if mode == 'circle':
            radius = int(np.sqrt((x - ix)**2 + (y - iy)**2))
            cv2.circle(img, (ix, iy), radius, (255, 0, 0), 2)
        elif mode == 'line':
            cv2.line(img, (ix, iy), (x, y), (0, 0, 255), 2)
        cv2.imshow('Image', img)


# --- 메인 실행부 ---
cv2.namedWindow('Image')
cv2.setMouseCallback('Image', draw_shape)

cv2.imshow('Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
