import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import scipy.io as scio

# 손실 함수
def Regularized_loss(model, n, y_pred, y_true, p=4, lam=0.01, l1_ratio=0.5):
    classification_loss = -torch.mean(y_true * torch.log_softmax(y_pred, dim=1))
    
    # 첫 번째 Linear 레이어의 가중치 사용
    L2_loss = 1/n * torch.norm(model.fc1.weight.unsqueeze(1) - model.fc1.weight.unsqueeze(0), p=2, dim=2).pow(p).sum()
    # loss = classification_loss + lam * RG_loss
    L1_loss = torch.norm(model.fc1.weight, p=1)
    loss = classification_loss + lam * ((1 - l1_ratio) * L2_loss + l1_ratio * L1_loss)
    return loss

# 다층 퍼셉트론(MLP) 모델 정의
class MLPModel(torch.nn.Module):
    def __init__(self, input_size, num_classes):
        super(MLPModel, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, 128)
        self.fc2 = torch.nn.Linear(128, 64)
        self.fc3 = torch.nn.Linear(64, num_classes)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.5)  # 과적합 방지용

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 데이터 전처리 함수
def preprocess_data(X, scale_type='standard'):
    if scale_type == 'standard':
        scaler = StandardScaler()
    elif scale_type == 'minmax':
        scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    return X

# R_MLR 함수 수정
def R_MLR(para):
    path = f'./dataset/{para.data}.mat'
    X = scio.loadmat(path)['X']
    y = scio.loadmat(path)['Y'].squeeze()
    print(X.shape, y.shape)

    n, d = X.shape[0], X.shape[1]
    num_class = len(np.unique(y))

    if para.If_scale == True:
        X = preprocess_data(X, scale_type=para.scale_type)  # Scaler 적용

    y = y - 1

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    X_train, X_test, y_train, y_test = \
        train_test_split(X_tensor, y_tensor, test_size=para.test_size, random_state=para.state)

    y_train = torch.nn.functional.one_hot(torch.tensor(y_train))
    y_test = torch.nn.functional.one_hot(torch.tensor(y_test))

    # Define the MLP model and optimizer
    model = MLPModel(d, num_class)
    optimizer = torch.optim.Adam(model.parameters(), lr=para.lr, weight_decay=para.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

    # 조기 종료 설정
    best_loss = float('inf')
    patience = 10
    early_stopping_counter = 0

    # Save the loss function values on the training set and the accuracy on the test set
    loss_list = []
    test_acc_list = []

    for epoch in range(para.num_epoch):
        model.train()
        y_pred = model(X_train)
        loss = Regularized_loss(model, n, y_pred, y_train, para.p, para.lam)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()  # Learning rate 업데이트

        if loss.item() < best_loss:
            best_loss = loss.item()
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1

        if early_stopping_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{para.num_epoch}], Loss: {loss.item():.4f}")

            model.eval()
            with torch.no_grad():
                y_pred = model(X_test)
                correct = (torch.argmax(y_pred, dim=1) == torch.argmax(y_test, dim=1)).sum().item()
                test_acc = correct / len(X_test)
                print(f"Test Accuracy: {test_acc:.4f}")

            loss_list.append(loss.item())
            test_acc_list.append(test_acc)

    print(f"Total Test Accuracy for {para.data}: {test_acc:.4f}")

    epochs = np.arange(1, len(loss_list) + 1) * 5
    # conver_plot(epochs, test_acc_list, loss_list)