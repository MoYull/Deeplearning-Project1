import cv2

# 1. 비디오 캡처 객체 생성 (0은 기본 웹캠을 의미)
cap = cv2.VideoCapture(0)

# 캡처가 성공적으로 열렸는지 확인
if not cap.isOpened():
    print("웹캠을 열 수 없습니다.")
    exit()

while True:
    # 2. 프레임 읽기
    # ret: 성공적으로 프레임을 읽었으면 True
    # frame: 읽어온 프레임 (컬러 이미지)
    ret, frame = cap.read()

    if not ret:
        print("프레임을 받을 수 없습니다. 종료합니다.")
        break

    # 3. 이미지 처리 적용: BGR -> Gray Scale 변환
    # 비디오의 모든 프레임에 이미지 처리 기술을 적용할 수 있습니다.
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 4. 화면에 결과 표시
    cv2.imshow('Original Video', frame)
    cv2.imshow('Gray Scale Video', gray_frame)

    # 5. 종료 조건 설정
    # 'q' 키를 누르면 루프를 종료합니다.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 6. 자원 해제
cap.release()
cv2.destroyAllWindows()