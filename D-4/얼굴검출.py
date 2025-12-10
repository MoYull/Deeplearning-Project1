import cv2
import os

# 1. 찾은 절대 경로를 r'...' 안에 붙여넣습니다.
# 예시: 'C:\ProgramData\anaconda3\envs\cv_env\Library\etc\haarcascades\haarcascade_frontalface_default.xml'
# 이 경로는 당신이 직접 찾은 경로여야 합니다!
cascade_file_path = r'/opt/anaconda3/envs/deeplearning/lib/python3.11/site-packages/cv2/data/haarcascade_frontalface_default.xml'
# 예: r'C:\ProgramData\anaconda3\envs\cv_env\Library\etc\haarcascades\haarcascade_frontalface_default.xml'

# 2. 분류기 로드
face_cascade = cv2.CascadeClassifier(cascade_file_path)

# 3. 로드 성공 여부 검사 (오류 방지 핵심 코드)
if face_cascade.empty():
    print("--- 오류 발생! ---")
    print(f"하르 캐스케이드 파일 로드 실패. 다음 경로를 확인하세요:")
    print(f"지정한 경로: {cascade_file_path}")
    
    # 파일을 찾을 수 없는 경우, 이 시점에서 프로그램 종료
    # 'detectMultiScale'을 호출하기 전에 멈춥니다.
    exit() 

print("하르 캐스케이드 파일 로드 성공.")

file_path = './images/sample.jpg'
img = cv2.imread(file_path)

if img is None:
    print("이미지 로드 실패")
    exit()

# 2. 전처리: 흑백 변환 (Haar 캐스케이드는 흑백 이미지에서 더 빠름)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 3. 얼굴 검출
# detectMultiScale(이미지, scaleFactor, minNeighbors)
# - scaleFactor: 이미지 크기 축소 비율 (1.1은 10%씩 축소)
# - minNeighbors: 최종적으로 객체로 인정하기 위한 최소 검출 횟수
faces = face_cascade.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# 4. 검출된 얼굴 영역에 사각형 그리기
for (x, y, w, h) in faces:
    # 원본 컬러 이미지에 초록색 사각형 그리기
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

# 5. 결과 표시
print(f"검출된 얼굴 개수: {len(faces)}")
cv2.imshow('Face Detection (Haar Cascade)', img)

cv2.waitKey(0)
cv2.destroyAllWindows()