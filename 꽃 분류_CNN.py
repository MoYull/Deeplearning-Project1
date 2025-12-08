import numpy as np
import os
import random
from PIL import Image as pilimg
import imghdr
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# ======================================================================
# 1. 데이터 로딩 및 전처리 (Keras 코드와 동일)
# ======================================================================

# 데이터 만드는 수 
def makeData(folder, label, isTrain):
    if isTrain == 'train':
        path = "./data/flowers_/train/" + folder
    else:
        path = "./data/flowers_/test/" + folder
    data = [] 
    labels = [] 
    i=1 
    for filename in os.listdir(path):
        if i%1000==0:
            print(i)
        i += 1
        try: 
            kind = imghdr.what(path + "/" + filename)
            if kind in ["gif", "png", "jpeg", "jpg"]:
                img = pilimg.open(path + "/" + filename).convert('RGB') # PyTorch는 3채널 RGB를 선호
                resize_img = img.resize((150, 150)) #크기줄이기:시간단축
                pixel = np.array(resize_img)
                if pixel.shape==(150, 150, 3): #150x150x3일 경우 제외/이상이미지 제외
                    data.append(pixel)
                    labels.append(label)
        except Exception as e:
            print(f"{filename} error: {e}")

    np.savez("imagedata{}.npz".format(str(label) + '_' + isTrain), data=data, targets=labels)
    print("파일저장완료")


# 이미 생성된 npz 파일이 있다고 가정하고 로딩하는 코드
# makeData 함수는 위 Keras 코드의 실행 결과와 동일한 데이터를 생성

def dataCreate():
    flowers = ["daisy", "dandelion", "rose", "sunflower", "tulip"]
    label=0 
    for flower in flowers:
        makeData(flower, label, 'train')
        makeData(flower, label, 'test')
        label += 1


def load_and_concat_data(mode):
    datas = []
    targets = []
    for label in range(5):
        npz_file = np.load(f"imagedata{label}_{mode}.npz")
        datas.append(npz_file["data"])
        targets.append(npz_file["targets"])
    
    return np.concatenate(datas, axis=0), np.concatenate(targets, axis=0)


trainData_np, trainTarget_np = load_and_concat_data('train')
testData_np, testTarget_np = load_and_concat_data('test')

# 데이터 형태 변환 (PyTorch에 맞게)
# PyTorch는 `channels_first` (C, H, W) 형태를 선호합니다.
# 데이터 정규화: 0-255 -> 0.0-1.0
trainData_tensor = torch.from_numpy(trainData_np).float().permute(0, 3, 1, 2) / 255.0
testData_tensor = torch.from_numpy(testData_np).float().permute(0, 3, 1, 2) / 255.0

# PyTorch에서는 One-Hot Encoding을 사용하지 않고, CrossEntropyLoss에 정수 라벨을 바로 전달
trainTarget_tensor = torch.from_numpy(trainTarget_np).long()
testTarget_tensor = torch.from_numpy(testTarget_np).long()

print(f"Train Data Shape: {trainData_tensor.shape}")
print(f"Test Data Shape: {testData_tensor.shape}")

# 데이터 로더 생성
train_dataset = TensorDataset(trainData_tensor, trainTarget_tensor)
test_dataset = TensorDataset(testData_tensor, testTarget_tensor)

train_loader = DataLoader(dataset=train_dataset, batch_size=100, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=100, shuffle=False)

# ======================================================================
# 2. 모델 정의 (CNN)
# ======================================================================
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)   # 150 → 75
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)   # 75 → 37
        
        # 64ch × 37 × 37 = 875,776
        self.fc_input_size = 64 * 37 * 37

        self.relu = nn.ReLU()

        self.fc1 = nn.Linear(self.fc_input_size, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 5)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool1(x)

        x = self.relu(self.conv2(x))
        x = self.pool2(x)

        x = x.view(-1, self.fc_input_size)

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x

model = SimpleCNN()
print(model)


# ======================================================================
# 3. 모델 컴파일 및 학습
# ======================================================================

# 손실 함수와 옵티마이저 정의
criterion = nn.CrossEntropyLoss() # Keras의 'categorical_crossentropy'와 동일
optimizer = optim.SGD(model.parameters(), lr=0.01) # learning rate는 기본값으로 설정
#오티마니저 SGD, Adam, RMSProp
epochs = 30

# 학습 루프
for epoch in range(epochs):
    model.train() # 모델을 학습 모드로 설정
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        # 그래디언트 초기화
        optimizer.zero_grad()
        
        # 순전파 (forward pass)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # 역전파 (backward pass) 및 옵티마이저 스텝
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
    print(f'Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader):.4f}')

# ======================================================================
# 4. 모델 평가
# ======================================================================

# 모델을 평가 모드로 설정
model.eval()
train_correct = 0
train_total = 0
test_correct = 0
test_total = 0

# 훈련셋 평가
with torch.no_grad(): # 평가 시에는 그래디언트를 계산하지 않음
    for inputs, labels in train_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()

# 테스트셋 평가
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()

train_acc = 100 * train_correct / train_total
test_acc = 100 * test_correct / test_total

print(f'\n훈련셋 정확도: {train_acc:.2f}%')
print(f'테스트셋 정확도: {test_acc:.2f}%')