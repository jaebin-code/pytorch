import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    transform = transforms.Compose(
        [transforms.Resize((32, 32)),  # 이미지 크기를 32x32로 조정
        transforms.ToTensor(),  # 이미지를 PyTorch 텐서로 변환
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # CIFAR10 훈련 데이터 로드
    trainset = torchvision.datasets.CIFAR10(root = './data', train = True, download = True, transform = transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size = 4, shuffle = True, num_workers = 2)

    # 테스트 데이터 로드
    testset = torchvision.datasets.CIFAR10(root = './data', train = False, download = True, transform = transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size = 4, shuffle = True, num_workers = 2)

    # 클래스 이름 튜플로 정의
    classes = ('airplane','automobile','bird','cat','deer','dog', 'frog', 'horse', 'ship', 'truck')


    class LeNet5(nn.Module):
        def __init__(self):
            super(LeNet5, self).__init__()
            # 초기 RGB 채널 3, 6개의 채널로, 5X5 필터
            # 32x32에 5짜리 필터 적용 -> 28x28 전환
            self.conv1 = nn.Conv2d(3, 6, 5) 
            # 2x2 평균 풀링, 보통 맥스 풀링 사용하기도 함 -> 14x14 전환
            self.pool = nn.AvgPool2d(2, 2) 
            # 이거는 논문에서는 선택 기준이 있던데, 알아서 해주는건지 확인, random일 확률이 높을듯
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16*5*5, 120) #120x1 차원 conv한 것을 완전 연결 계층으로 표현
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10) # 최종적으로 각 column이 클래스를 나타낼 수 있게 구성

        def forward(self,x):
            # 풀링, 활성화함수로 relu 활용
            x = self.pool(torch.relu(self.conv1(x)))
            x = self.pool(torch.relu(self.conv2(x)))
            # 완전 연결 계층 전에 flatten하고 적용
            x = torch.flatten(x,1)
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = self.fc3(x)
            return x
            
    # 객체 생성
    net = LeNet5()

    # loss function이랑 optimizer

    criterion = nn.CrossEntropyLoss()

    # 확률적 경사하강법, momentum은 이전 변화량 어느정도 반영할지
    optimizer = optim.SGD(net.parameters(), lr=0.003,momentum=0.8)

    for epoch in range(4):
        running_loss = 0
        for i, data in enumerate(trainloader, 0):
            # 입력 데이터
            inputs, labels = data

            # 매개변수 경사도를 0으로 설정
            optimizer.zero_grad()

            # 순전파 + 역전파 + 최적화
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # 통계 출력
            running_loss += loss.item()
            if i % 2000 == 1999:    # 매 2000 미니배치마다 출력, 한 배치당 이미지 4개씩, 50000개의 데이터 12500번
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    # 테스트 데이터셋을 사용하여 네트워크의 성능 평가
    correct = 0 
    total = 0 
    with torch.no_grad():
        for data in testloader: 
            images, labels = data 
            outputs = net(images)  
            _, predicted = torch.max(outputs.data, 1)  
            total += labels.size(0)  
            correct += (predicted == labels).sum().item()  

    print(f'Accuracy of the network on the 10000 test images: {100 * correct / total:.2f}%')
