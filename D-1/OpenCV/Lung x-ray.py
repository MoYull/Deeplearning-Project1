import os
import shutil
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pickle  #파일저장에
from PIL import Image as pilimg
#pip install tqdm
from tqdm import tqdm # 학습 진행 상황 시각화를 위해 추가

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# 이후 PyTorch, NumPy, TensorFlow 등을 import 합니다.
# 경고 무시 설정
warnings.filterwarnings('ignore')

import random 
def set_seed(seed_value=42):
    random.seed(seed_value) # Python 기본 난수
    np.random.seed(seed_value) # NumPy 난수
    torch.manual_seed(seed_value) # PyTorch CPU 난수
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value) # PyTorch GPU 난수
        torch.cuda.manual_seed_all(seed_value) # PyTorch 모든 GPU 난수
        
        # CUDNN이 결정적 연산을 사용하도록 강제
        torch.backends.cudnn.deterministic = True 
        torch.backends.cudnn.benchmark = False
        
set_seed(42)

# 1. 경로 설정 및 파라미터
original_dataset_dir = './data/cats_and_dogs/train'
"""
cats_and_dogs
   ㄴ train 

cats_and_dogs_small
   ㄴ train 
       ㄴcats
       ㄴdogs 
   ㄴ test 
       ㄴcats
       ㄴdogs 
   ㄴ validation 
        ㄴcats
       ㄴdogs 

"""
base_dir = './data/cats_and_dogs_small'

train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

model_save_path_pth = 'cats_and_dogs_model.pth'
history_filepath = 'cats_and_dogs_history.pkl'

batch_size = 16
img_height = 180
img_width = 180
num_epochs = 30 # 예시로 epoch 수를 30으로 설정했습니다.
learning_rate = 0.001 # learning_rate 추가

# GPU 사용 가능 여부 확인
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"사용 중인 디바이스: {device}")

# 이미지 복사 함수 
def ImageCopyMove():
    #경로가 있는지 확인해본다  경로가 있으면 True 없으면 False 
    if os.path.exists(base_dir):
        #shutil :shell util - 명령어 해석기 
        shutil.rmtree(base_dir, ignore_errors=True)
    os.makedirs(train_dir)  #디렉토리 생성 
    os.makedirs(validation_dir)
    os.makedirs(test_dir)
    
    train_cats_dir = os.path.join(train_dir, 'cats')
    train_dogs_dir = os.path.join(train_dir, 'dogs')
    validation_cats_dir = os.path.join(validation_dir, 'cats')
    validation_dogs_dir = os.path.join(validation_dir, 'dogs')
    test_cats_dir = os.path.join(test_dir, 'cats')
    test_dogs_dir = os.path.join(test_dir, 'dogs')

    os.makedirs(train_cats_dir)
    os.makedirs(train_dogs_dir)
    os.makedirs(validation_cats_dir)
    os.makedirs(validation_dogs_dir)
    os.makedirs(test_cats_dir)
    os.makedirs(test_dogs_dir)
    
    #["cat.0.jpg", "cat.1.jpg",..... ]
    fnames = ['cat.{}.jpg'.format(i) for i in range(1000)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(train_cats_dir, fname)
        shutil.copyfile(src, dst)
        
    fnames = ['cat.{}.jpg'.format(i) for i in range(1000, 1500)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(validation_cats_dir, fname)
        shutil.copyfile(src, dst)
    
    fnames = ['cat.{}.jpg'.format(i) for i in range(1500, 2000)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(test_cats_dir, fname)
        shutil.copyfile(src, dst)
    
    fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(train_dogs_dir, fname)
        shutil.copyfile(src, dst)
    
    fnames = ['dog.{}.jpg'.format(i) for i in range(1000, 1500)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(validation_dogs_dir, fname)
        shutil.copyfile(src, dst)
    
    fnames = ['dog.{}.jpg'.format(i) for i in range(1500, 2000)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(test_dogs_dir, fname)
        shutil.copyfile(src, dst)

    print("이미지 복사 및 폴더 생성 완료!")


# 2. PyTorch 모델 정의 (CNN)
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        #180 by 180
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 90 by 90
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 45 by 45

        # Flatten 후 크기: 64 * 45 * 45 = 129600
        self.fc1 = nn.Linear(64 * 45 * 45, 512)
        self.dropout = nn.Dropout(0.5)  #데이터를 0.5를 날린다. , 계산되었던 가중치를 절반을 없앤다. 
        #의도적으로 노이즈를 발생시킨다. 과대적합을 차단한ㄷ다. 
        self.fc2 = nn.Linear(512, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid() # 라벨이 2개면 

    def forward(self, x):
        x = self.pool1(self.relu(self.conv1(x)))
        x = self.pool2(self.relu(self.conv2(x)))
        
        x = x.view(-1, 64 * 45 * 45) #완전연결신경망 과 차원을 일치시키다. 
        
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.sigmoid(self.fc2(x))
        return x

# 3. 데이터 로딩 및 학습 함수 (검증 단계 및 출력 추가)
def DataIncrease():
    #파이프라인 -> 다른 하나의 프로세스의 출력이 다른 프로세스이 입력이 될때
    #데이터 증강 및 전처리 파이프라인
    #폴더로 부터 파일을 읽어와야 한다. 그러기 전에 파일을 

    train_transforms = transforms.Compose([
        transforms.Resize((img_height, img_width)), #모든 이미지의 크기를 맞춘다, 입력데이터 크기 통일
        transforms.RandomHorizontalFlip(p=0.5),  
        #이미지를 50%의 확률(p=0.5)로 무작위로 수평 반전 이는 모델이 왼쪽/오른쪽 방향에 덜 민감하게 만듬
        transforms.RandomRotation(10), #이미지를 무작위로 -10도에서 +10도 사이로 회전시킵니다.
        transforms.ToTensor(), #PIL Image 형태의 데이터를 PyTorch가 처리할 수 있는 텐서(Tensor)로 변환합니다, 자동으로 0~1사이로 정규화
    ])

    #검증 데이터 전처리 (val_transforms)검증 단계에서는 데이터의 정확한 성능을 측정해야 하므로 무작위 변환을 피합니다.
    val_transforms = transforms.Compose([
        transforms.Resize((img_height, img_width)),
        transforms.ToTensor(),
    ])

    # 데이터셋 불러오기
    #datasets.ImageFolder: torchvision에서 제공하는 매우 유용한 클래스입니다.
    #train_dir과 같이 클래스별로 하위 폴더가 구성된 디렉토리 경로(예: train/cats, train/dogs)를 받습니다.
    #자동으로 하위 폴더 이름을 클래스 레이블로 인식하고, 폴더 내의 이미지 파일을 찾아서 지정된 transform 파이프라인을 적용합니다.

    #이미지 증강처리
    #Dataset과 DataLoader의 작동 방식
    
    #캐쉬 -> 속도 cpu > ram > ssd > 입출력 
    #본래의미의 캐쉬 : cpu와 ram메모리 사이에 속도 차가 발생한다. 캐쉬메모리(sram), ram(dram)
    #캐쉬->컴퓨터 입장에서 캐쉬를 뭐를 해야할지 모른다. 

    #1. Dataset (ImageFolder)
    # 파일 시스템에 있는 원본 이미지의 경로만 가지고 있음. 이미지를 메모리에 미리 캐시하지 않는다.
    #2. DataLoader: 학습 루프(for inputs, labels in train_loader:)가 돌 때마다 배치 사이즈만큼의 데이터를 요구
    #DataLoader가 이미지를 요청하면, Dataset은 해당 이미지 파일을 디스크에서 로드한 다음, 
    #정의된 transform 파이프라인을 처음부터 끝까지 실행.
    #매 에포크(Epoch), 매 배치(Batch)마다 
    # transforms.RandomHorizontalFlip(p=0.5)는 새로운 난수를 생성하여, 이번 배치의 이미지를 뒤집을지 말지 다시 결정합니다.
    # transforms.RandomRotation(10)도 새로운 난수를 생성하여, 이번 배치의 이미지를 얼마나 회전시킬지(-10° ~ +10°) 다시 결정합니다.
    # 동일한 결과를 얻고 싶으면 반드시 시드를 지정해야 한다
                                                                            
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    val_dataset = datasets.ImageFolder(validation_dir, transform=val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    model = ConvNet().to(device)
    criterion = nn.BCELoss() # Binary Cross-Entropy Loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    history = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': []}

    print("PyTorch 모델 학습 시작...")
    for epoch in range(num_epochs):
        # 훈련 단계 (Training Step)
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        #훈련중인 부분을 보이게 하려고 한다. 
        for images, labels in train_loop:
            images, labels = images.to(device), labels.to(device)
            labels = labels.float().unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            predicted = (outputs > 0.5).float()
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
            train_loop.set_postfix(batch_loss=loss.item())
            
        avg_train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct_train / total_train
        history['loss'].append(avg_train_loss)
        history['accuracy'].append(train_acc)
        
        # 검증 단계 (Validation Step)
        model.eval()
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        val_loop = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Validation]")
        with torch.no_grad():
            for images, labels in val_loop:
                images, labels = images.to(device), labels.to(device)
                labels = labels.float().unsqueeze(1)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item()
                
                predicted = (outputs > 0.5).float()
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
                
                val_loop.set_postfix(batch_loss=loss.item())

        avg_val_loss = running_val_loss / len(val_loader)
        val_acc = 100 * correct_val / total_val
        history['val_loss'].append(avg_val_loss)
        history['val_accuracy'].append(val_acc)
        
        # 에포크별 결과 출력
        print(f"\n[Epoch {epoch+1}/{num_epochs}]")
        print(f"  Training Loss: {avg_train_loss:.4f}, Training Accuracy: {train_acc:.2f}%")
        print(f"  Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_acc:.2f}%")
    
    print("학습 완료!")
    
    # 모델 저장  - model.state_dict() :가중치
    torch.save(model.state_dict(), model_save_path_pth)
    print(f"모델 가중치가 '{model_save_path_pth}'에 저장되었습니다.")
    
    # 히스토리 저장
    with open(history_filepath, 'wb') as file:
        pickle.dump(history, file)
    print(f"학습 히스토리가 '{history_filepath}'에 저장되었습니다.")

# 4. 모델 로드 및 평가 함수 (LoadModels)
def LoadModels():
    try:
        #모델의 상태 딕셔너리(state_dict)**를 불러옵니다. 이 딕셔너리에는 모델의 학습된 모든 가중치와 편향이 키-값 쌍으로 포함되어 있습니다.
        #참고: torch.load는 저장 장치(CPU/GPU)에 관계없이 딕셔너리 형태의 데이터를 메모리로 불러옵니다.
        #model.load_state_dict(...): 생성된 빈 모델 인스턴스(model)에 불러온 가중치와 편향 값들을 적용하여, 모델을 학습이 완료된 상태로 만듭니다.

        model = ConvNet()
        model.load_state_dict(torch.load(model_save_path_pth))
        model.to(device)

        print(f"모델이 '{model_save_path_pth}'에서 성공적으로 로드되었습니다.")
    except Exception as e:
        print(f"모델 로드 중 오류 발생: {e}")
        return False, None
    
    try:
        with open(history_filepath, 'rb') as file:
            history = pickle.load(file)
            print(f"학습 히스토리가 '{history_filepath}'에 성공적으로 불러왔습니다.")
    except Exception as e:
        print(f"학습 히스토리 읽는 중 오류 발생: {e}")
        # 히스토리 파일이 없거나 오류가 나면 그래프를 그릴 수 없으므로 False 반환
        return False, None

    # 그래프 그리기
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    if 'accuracy' in history and 'val_accuracy' in history:
        plt.plot(history['accuracy'], label='Training Acc')
        plt.plot(history['val_accuracy'], label='Validation Acc')
        plt.title('Training and Validation Accuracy')
        plt.legend()
    else:
        plt.title('Accuracy data missing')

    plt.subplot(1, 2, 2)
    if 'loss' in history and 'val_loss' in history:
        plt.plot(history['loss'], label='Training Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
    else:
        plt.title('Loss data missing')
        
    plt.show()

    return True, model

# 5. 예측 함수
def Predict():
    success, loaded_model = LoadModels()
    if not success:
        return

    test_transforms = transforms.Compose([
        transforms.Resize((img_height, img_width)),
        transforms.ToTensor(),
    ])

    test_dataset = datasets.ImageFolder(test_dir, transform=test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    loaded_model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            labels = labels.float().unsqueeze(1)
            
            outputs = loaded_model(images)
            predicted = (outputs > 0.5).float()
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"테스트 데이터셋의 정확도: {accuracy:.2f}%")

# 메인 실행 부분
if __name__ == "__main__":
    while True:
        print("\n--- 메뉴 ---")
        print("1. 이미지 복사")
        print("2. 데이터 학습")
        print("3. 모델 로드 및 시각화")
        print("4. 예측하기")
        print("5. 종료")
        sel = input("선택: ")
        
        if sel == "1":
            ImageCopyMove()
        elif sel == "2":
            DataIncrease()
        elif sel == "3":
            LoadModels()
        elif sel == "4":
            Predict()
        elif sel == "5":
            break
        else:
            print("잘못된 입력입니다.")

    #휴리스틱 분석 - 어림짐작, 경험치에 의해서 분석을 해봤는데  안맞으면 처음부터 다시 