import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import logging
import time

torch.manual_seed(13)
np.random.seed(13)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.basicConfig(filename="svm_gpu.log", level=logging.INFO, format="%(asctime)s - %(message)s")
logging.info("starting svm training")
data_dictionary="1_3/data"
X, y = make_classification(
    n_samples=10000,
    n_features=64,
    n_informative=50,
    n_redundant=5,
    n_classes=10,
    random_state=13
)

scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=13)

X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.long).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test, dtype=torch.long).to(device)

class RBFKernelLayer(nn.Module):
    def __init__(self, in_features, num_centers, gamma):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(num_centers, in_features))
        self.gamma = gamma

    def forward(self, x):
        x_expand = x.unsqueeze(1)
        centers_expand = self.centers.unsqueeze(0)
        dist = torch.sum((x_expand - centers_expand) ** 2, dim=2)
        return torch.exp(-self.gamma * dist)

class TorchSVM(nn.Module):
    def __init__(self, input_dim, num_classes, kernel_type="linear", gamma=0.01):
        super().__init__()
        self.kernel_type = kernel_type
        if kernel_type == "linear":
            self.fc = nn.Linear(input_dim, num_classes)
        else:
            self.kernel = RBFKernelLayer(input_dim, num_centers=128, gamma=gamma)
            self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        if self.kernel_type == "linear":
            return self.fc(x)
        else:
            features = self.kernel(x)
            return self.fc(features)

def hinge_loss(outputs, targets, C):
    one_hot = torch.zeros_like(outputs).scatter_(1, targets.unsqueeze(1), 1)
    margins = 1 - outputs * (2 * one_hot - 1)
    return 0.5 * torch.norm(outputs) ** 2 + C * torch.mean(torch.clamp(margins, min=0))

param_grid = {
    "lr": [1e-3, 5e-4],
    "C": [0.1, 1.0, 10.0],
    "epochs": [100],
    "batch_size": [64],
    "kernel": ["linear", "rbf"],
    "gamma": [0.01, 0.05]
}

best_acc = 0
best_params = None

for params in ParameterGrid(param_grid):
    model = TorchSVM(
        input_dim=X_train.shape[1],
        num_classes=len(torch.unique(y_train)),
        kernel_type=params["kernel"],
        gamma=params["gamma"]
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=params["lr"])
    logging.info(f"Training with params: {params}")
    start = time.time()

    for epoch in range(params["epochs"]):
        model.train()
        perm = torch.randperm(X_train.size(0))
        epoch_loss = 0
        for i in range(0, X_train.size(0), params["batch_size"]):
            idx = perm[i:i + params["batch_size"]]
            x_batch, y_batch = X_train[idx], y_train[idx]
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = hinge_loss(outputs, y_batch, params["C"])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        model.eval()
        with torch.no_grad():
            preds = model(X_test).argmax(1)
            acc = accuracy_score(y_test.cpu(), preds.cpu())

        logging.info(f"Epoch [{epoch+1}/{params['epochs']}], Loss: {epoch_loss:.4f}, Acc: {acc:.4f}")

    total_time = time.time() - start
    logging.info(f"Training time: {total_time:.2f}s")

    if acc > best_acc:
        best_acc = acc
        best_params = params
        torch.save(model.state_dict(), "best_svm_model.pt")
        logging.info(f"New best model saved: acc={best_acc:.4f}")

logging.info(f"Best params: {best_params}")
logging.info(f"Best accuracy: {best_acc:.4f}")

model = TorchSVM(X_train.shape[1], len(torch.unique(y_train)),
                 kernel_type=best_params["kernel"], gamma=best_params["gamma"]).to(device)
model.load_state_dict(torch.load("best_svm_model.pt", weights_only=True))
model.eval()

with torch.no_grad():
    preds = model(X_test).argmax(1)
    report = classification_report(y_test.cpu(), preds.cpu())
    logging.info("Final classification report:\n" + report)
    print("Best Parameters:", best_params)
    print("Final Accuracy:", best_acc)
