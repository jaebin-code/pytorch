import torch.nn as nn

class LeNet5(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2d(3, 6, 5),
            nn.ReLU(),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.AvgPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(16*5*5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10),
        )

class LeNet7(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2d(3, 6, 5),
            nn.ReLU(),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(16, 32, 3),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Flatten(),
            nn.Linear(32*3*3, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10),
        )

class LeNet9(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2d(3, 12, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(12, 16, 5),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(16*5*5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10),
        )

def get_model(name):
    if name == "LeNet5":
        return LeNet5()
    elif name == "LeNet7":
        return LeNet7()
    elif name == "LeNet9":
        return LeNet9()
    else:
        raise ValueError("Unsupported model")