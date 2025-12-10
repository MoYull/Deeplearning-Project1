print("딥러닝 코드 실행됩니다!!")

# 1. 라이브러리 불러오기
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np

# 2. penguins 데이터 로드
penguins = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv")
penguins = penguins.dropna()

print("penguins 데이터 shape:", penguins.shape)   # ← 이제 정상적으로 출력됨

# 3. feature/label 분리
feature_cols = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
X = penguins[feature_cols].values

le = LabelEncoder()
y = le.fit_transform(penguins["species"])

print("X shape:", X.shape)
print("y shape:", len(y))

#데이터 스케일링
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#5 Train/Test  분리
X_train, X_test, y_train, y_test  = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Torch Tensor로 변환
X_train = torch.FloatTensor(X_train)
X_test  = torch.FloatTensor(X_test)
y_train = torch.LongTensor(y_train)
y_test  = torch.LongTensor(y_test)

# CNN 입력 형태로 변환: (batch, channels=1, length=4)
X_train_cnn = X_train.unsqueeze(1)
X_test_cnn = X_test.unsqueeze(1)

print("CNN X_train shape:", X_train_cnn.shape)

class PenguinCNN(nn.Module):
    def __init__(self):
        super(PenguinCNN, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=2),  # → (batch,16,3)
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=2), # → (batch,32,2)
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 2, 32),  # 32채널 × 길이 2
            nn.ReLU(),
            nn.Linear(32, 3)        # 클래스 3개 (Adelie, Chinstrap, Gentoo)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

model = PenguinCNN()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

epochs = 200

for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X_train_cnn)
    loss = criterion(outputs, y_train)
    
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 20 == 0:
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {loss.item():.4f}")

with torch.no_grad():
    test_outputs = model(X_test_cnn)
    predicted = torch.argmax(test_outputs, dim=1)
    accuracy = (predicted == y_test).float().mean()

print("CNN 모델 최종 정확도: {:.2f}%".format(accuracy.item() * 100))
