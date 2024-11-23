import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import scipy.io as scio

def preprocess_data(X, scale_type='standard'):
    if scale_type == 'standard':
        scaler = StandardScaler()
    elif scale_type == 'minmax':
        scaler = MinMaxScaler()
    return scaler.fit_transform(X)

class MLPModel(torch.nn.Module):
    def __init__(self, input_size, num_classes):
        super(MLPModel, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, 128)
        self.fc2 = torch.nn.Linear(128, 64)
        self.fc3 = torch.nn.Linear(64, num_classes)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def Regularized_loss(model, n, y_pred, y_true, p=4, lam=0.01, l1_ratio=0.5):
    classification_loss = -torch.mean(y_true * torch.log_softmax(y_pred, dim=1))
    L2_loss = 1/n * torch.norm(model.fc1.weight.unsqueeze(1) - model.fc1.weight.unsqueeze(0), p=2, dim=2).pow(p).sum()
    L1_loss = torch.norm(model.fc1.weight, p=1)
    loss = classification_loss + lam * ((1 - l1_ratio) * L2_loss + l1_ratio * L1_loss)
    return loss

def R_MLR(para):
    path = f'./dataset/{para.data}.mat'
    data = scio.loadmat(path)
    X = data['X']
    y = data['Y'].squeeze()
    print(X.shape, y.shape)

    n, d = X.shape[0], X.shape[1]
    num_class = len(np.unique(y))

    if para.If_scale:
        X = preprocess_data(X, scale_type=para.scale_type)

    y = y - 1
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=para.test_size, random_state=para.state)
    y_train = torch.nn.functional.one_hot(torch.tensor(y_train))
    y_test = torch.nn.functional.one_hot(torch.tensor(y_test))

    model = MLPModel(d, num_class)
    optimizer = torch.optim.Adam(model.parameters(), lr=para.lr, weight_decay=para.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

    best_loss = float('inf')
    patience = 10
    early_stopping_counter = 0
    test_acc = 0

    for epoch in range(para.num_epoch):
        model.train()
        y_pred = model(X_train)
        loss = Regularized_loss(model, n, y_pred, y_train, para.p, para.lam)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1

        if early_stopping_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

        if (epoch + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                y_pred = model(X_test)
                correct = (torch.argmax(y_pred, dim=1) == torch.argmax(y_test, dim=1)).sum().item()
                test_acc = correct / len(X_test)
                print(f"Epoch [{epoch+1}/{para.num_epoch}], Test Accuracy: {test_acc:.4f}")

    print(f"Total Test Accuracy for {para.data}: {test_acc:.4f}")
    return test_acc  # 최종 테스트 정확도 반환