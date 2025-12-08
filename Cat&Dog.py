import os
import shutil
import warnings
from PIL import Image as pilimg
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

warnings.filterwarnings('ignore')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# 1. 경로 설정
original_dataset_dir = './data/cats_and_dogs/train'
base_dir = './data/cats_and_dogs_small'

train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

batch_size = 16
img_height = 180
img_width = 180
num_epochs = 20
learning_rate = 0.001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Device] {device}")

# 2. 이미지 분리 함수
def ImageCopyMove():
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir, ignore_errors=True)

    os.makedirs(train_dir)
    os.makedirs(validation_dir)
    os.makedirs(test_dir)

    dirs = {
        'train_cats': os.path.join(train_dir, 'cats'),
        'train_dogs': os.path.join(train_dir, 'dogs'),
        'val_cats': os.path.join(validation_dir, 'cats'),
        'val_dogs': os.path.join(validation_dir, 'dogs'),
        'test_cats': os.path.join(test_dir, 'cats'),
        'test_dogs': os.path.join(test_dir, 'dogs')
    }

    for d in dirs.values():
        os.makedirs(d)

    # cat 0~999 → train
    for i in range(1000):
        shutil.copyfile(
            os.path.join(original_dataset_dir, f"cat.{i}.jpg"),
            os.path.join(dirs['train_cats'], f"cat.{i}.jpg")
        )

    # cat 1000~1499 → validation
    for i in range(1000, 1500):
        shutil.copyfile(
            os.path.join(original_dataset_dir, f"cat.{i}.jpg"),
            os.path.join(dirs['val_cats'], f"cat.{i}.jpg")
        )

    # cat 1500~1999 → test
    for i in range(1500, 2000):
        shutil.copyfile(
            os.path.join(original_dataset_dir, f"cat.{i}.jpg"),
            os.path.join(dirs['test_cats'], f"cat.{i}.jpg")
        )

    # dog 동일
    for i in range(1000):
        shutil.copyfile(
            os.path.join(original_dataset_dir, f"dog.{i}.jpg"),
            os.path.join(dirs['train_dogs'], f"dog.{i}.jpg")
        )
    for i in range(1000, 1500):
        shutil.copyfile(
            os.path.join(original_dataset_dir, f"dog.{i}.jpg"),
            os.path.join(dirs['val_dogs'], f"dog.{i}.jpg")
        )
    for i in range(1500, 2000):
        shutil.copyfile(
            os.path.join(original_dataset_dir, f"dog.{i}.jpg"),
            os.path.join(dirs['test_dogs'], f"dog.{i}.jpg")
        )

    print("이미지 분류 완료!")

# 3. Transform 정의
transform = transforms.Compose([
    transforms.Resize((img_height, img_width)),
    transforms.ToTensor()
])

# 4. 데이터 로더 생성
def create_dataloaders():
    train_data = datasets.ImageFolder(train_dir, transform=transform)
    val_data = datasets.ImageFolder(validation_dir, transform=transform)
    test_data = datasets.ImageFolder(test_dir, transform=transform)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

# 5. CNN 모델 정의
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Conv block 1
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)

        # Conv block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)

        # Fully connected
        self.fc1 = nn.Linear(64 * 45 * 45, 128)
        self.fc2 = nn.Linear(128, 2)  # cat=0, dog=1

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool1(self.relu(self.conv1(x)))
        x = self.pool2(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 45 * 45)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 6. 모델 학습
def train_model():
    ImageCopyMove()
    train_loader, val_loader, test_loader = create_dataloaders()

    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print("학습 시작...")
    for epoch in range(num_epochs):
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"[Epoch {epoch+1}] Loss: {running_loss/len(train_loader):.4f}")

    print("학습 완료!")

    # 모델 저장
    torch.save(model.state_dict(), "cats_and_dogs_model.pth")

    # 테스트 정확도
    total = 0
    correct = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"테스트 정확도: {correct/total*100:.2f}%")

# 실행
if __name__ == "__main__":
    train_model()
