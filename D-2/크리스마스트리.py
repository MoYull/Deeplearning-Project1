import cv2
import numpy as np
import random

# 1. 배경(하늘) 이미지 만들기 (파란색 하늘 느낌)
img = np.zeros((600, 500, 3), dtype=np.uint8)
img[:] = (180, 120, 50)  # 약간 어두운 겨울 하늘 색 (B, G, R)

# 2. 바닥 눈
cv2.rectangle(img, (0, 450), (500, 600), (255, 255, 255), -1)  # 흰색 바닥

# 3. 크리스마스트리 
# 트리 색 (진한 초록)
tree_color1 = (28, 53, 23)
tree_color2 = (48, 101, 69)
tree_color3 = (71, 166, 132)

# 트릭 삼각형 3단
pts3 = np.array([[250, 240], [110, 410], [390, 410]], np.int32)
cv2.fillPoly(img, [pts3], tree_color3)

# 2단
pts2 = np.array([[250, 170], [130, 330], [370, 330]], np.int32)
cv2.fillPoly(img, [pts2], tree_color2)# 2단
pts2 = np.array([[250, 170], [130, 330], [370, 330]], np.int32)
cv2.fillPoly(img, [pts2], tree_color2)

# 1단
pts1 = np.array([[250, 100], [150, 250], [350, 250]], np.int32)
cv2.fillPoly(img, [pts1], tree_color1)


# 4. 트리 기둥
cv2.rectangle(img, (230, 410), (270, 460), (35, 91, 118), -1)

# 5. 트리에 장식 (랜덤 색 원)
for _ in range(15):
    x = random.randint(160, 340)
    y = random.randint(150, 380)
    color = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
    cv2.circle(img, (x, y), 7, color, -1)

# 6. 트리 꼭대기에 별
cv2.circle(img, (250, 90), 12, (0, 255, 255), -1)  # 노란 별

# 7. Merry Christmas 현수막
cv2.rectangle(img, (70, 20), (430, 80), (255, 255, 255), -1)  # 흰 배경
cv2.putText(img, "Merry Christmas", (85, 65),
            cv2.FONT_HERSHEY_TRIPLEX, 1.2, (0, 0, 255), 2)

# 8. 눈 내리는 효과
for _ in range(150):
    x = random.randint(0, 500)
    y = random.randint(0, 600)
    cv2.circle(img, (x, y), random.randint(2, 4), (255, 255, 255), -1)

# 9. 이미지 출력
cv2.imshow("Christmas Scene", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

#파일 저장
cv2.imwrite("D-2/christmas_tree.png", img)

