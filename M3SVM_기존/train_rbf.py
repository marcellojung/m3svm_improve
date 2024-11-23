import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import scipy.io as scio

def Regularized_loss(model, n, y_pred, y_true, p=4, lam=0.01):
    # Hinge loss for M3SVM, logistic regression loss for ISM3
    # y_true = torch.argmax(y_true, dim=1)
    # classification_loss = F.multi_margin_loss(y_pred, y_true) # Hinge loss

    classification_loss = -torch.mean(y_true * torch.log_softmax(y_pred, dim=1))  # Logits

    RG_loss = 1/n * torch.norm(model.weight.unsqueeze(1) - model.weight.unsqueeze(0), p=2, dim=2).pow(p).sum()
    loss = classification_loss + lam * RG_loss
    return loss

def R_MLR(para):
    # Load dataset
    path = f'./dataset/{para.data}.mat'
    X = scio.loadmat(path)['X']
    y = scio.loadmat(path)['Y'].squeeze()
    print(X.shape, y.shape)

    n, d = X.shape[0], X.shape[1]
    num_class = len(np.unique(y))

    if para.If_scale:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    y = y - 1  # Adjust labels if necessary

    # Convert data to PyTorch tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=para.test_size, random_state=para.state)

    y_train = torch.nn.functional.one_hot(torch.tensor(y_train))
    y_test = torch.nn.functional.one_hot(torch.tensor(y_test))

    # Define the model and optimizer
    model = torch.nn.Linear(d, num_class)
    optimizer = torch.optim.Adam(model.parameters(), lr=para.lr, weight_decay=para.weight_decay)

    # Save the loss function values on the training set and the accuracy on the test set
    loss_list = []
    test_acc_list = []

    for epoch in range(para.num_epoch):
        # Training step
        y_pred = model(X_train)
        loss = Regularized_loss(model, n, y_pred, y_train, para.p, para.lam)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{para.num_epoch}], Loss: {loss.item():.4f}")

            # Evaluation on the test set
            with torch.no_grad():
                y_pred = model(X_test)
                correct = (torch.argmax(y_pred, dim=1) == torch.argmax(y_test, dim=1)).sum().item()
                test_acc = correct / len(X_test)
                print(f"Test Accuracy: {test_acc:.4f}")

            loss_list.append(loss.item())
            test_acc_list.append(test_acc)

    print(f"Total Test Accuracy for {para.data}: {test_acc:.4f}")

    # Return the final test accuracy for potential usage in main scripts
    return test_acc