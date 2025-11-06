import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import numpy as np
from tqdm import tqdm
import joblib
import logging

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

svm = {
    "model": {},
    "data": {},
    "features": {},
    "results": {},
    "logs": {}
}

os.makedirs("svm/logs", exist_ok=True)
os.makedirs("svm/models", exist_ok=True)
os.makedirs("svm/features", exist_ok=True)

logging.basicConfig(
    filename="svm/logs/svm.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder("1_3/data", transform=transform)
indices_per_class = {}

for idx, (_, label) in enumerate(dataset.samples):
    if label not in indices_per_class:
        indices_per_class[label] = []
    indices_per_class[label].append(idx)

train_indices, test_indices = [], []
np.random.seed(13)

for label, indices in indices_per_class.items():
    np.random.shuffle(indices)
    n_train = int(0.8 * len(indices))
    train_indices.extend(indices[:n_train])
    test_indices.extend(indices[n_train:])

train_subset = torch.utils.data.Subset(dataset, train_indices)
test_subset = torch.utils.data.Subset(dataset, test_indices)

train_loader = DataLoader(train_subset, batch_size=32, shuffle=False, num_workers=13)
test_loader = DataLoader(test_subset, batch_size=32, shuffle=False, num_workers=13)

svm["data"]["train_size"] = len(train_subset)
svm["data"]["test_size"] = len(test_subset)
svm["data"]["num_classes"] = len(dataset.classes)

resnet = models.resnet50(pretrained=True)
resnet.fc = nn.Identity()
resnet = resnet.to(device)
resnet.eval()

def extract_features(dataloader):
    features, labels = [], []
    with torch.no_grad():
        for imgs, lbls in tqdm(dataloader, desc="Extracting Features"):
            imgs = imgs.to(device)
            out = resnet(imgs)
            features.append(out.cpu().numpy())
            labels.append(lbls.numpy())
    return np.concatenate(features), np.concatenate(labels)

train_features, train_labels = extract_features(train_loader)
test_features, test_labels = extract_features(test_loader)

svm["features"]["train_shape"] = train_features.shape
svm["features"]["test_shape"] = test_features.shape

np.save("svm/features/train_features.npy", train_features)
np.save("svm/features/train_labels.npy", train_labels)
np.save("svm/features/test_features.npy", test_features)
np.save("svm/features/test_labels.npy", test_labels)

scaler = StandardScaler()
train_features = scaler.fit_transform(train_features)
test_features = scaler.transform(test_features)

param_grid = {
    "C": [0.1, 1, 10],
    "gamma": ["scale", 0.01, 0.001],
    "kernel": ["rbf", "poly"]
}

svm_model = SVC(random_state=13)
grid = GridSearchCV(svm_model, param_grid, cv=3, n_jobs=13, verbose=2)
grid.fit(train_features, train_labels)

best_svm = grid.best_estimator_

preds = best_svm.predict(test_features)
acc = accuracy_score(test_labels, preds)
report = classification_report(test_labels, preds, target_names=dataset.classes, output_dict=True)
cm = confusion_matrix(test_labels, preds)

svm["model"]["best_params"] = grid.best_params_
svm["results"]["accuracy"] = acc
svm["results"]["classification_report"] = report
svm["results"]["confusion_matrix"] = cm.tolist()

joblib.dump(best_svm, "svm/models/best_svm_model.pkl")
joblib.dump(scaler, "svm/models/scaler.pkl")

logging.info(f"Training samples: {len(train_subset)}")
logging.info(f"Testing samples: {len(test_subset)}")
logging.info(f"Classes: {dataset.classes}")
logging.info(f"Best Parameters: {grid.best_params_}")
logging.info(f"Accuracy: {acc:.4f}")
logging.info("Classification Report:")
logging.info(classification_report(test_labels, preds, target_names=dataset.classes))
logging.info("Confusion Matrix:")
logging.info(cm)

print("Best Parameters:", grid.best_params_)
print(f"Accuracy: {acc:.4f}")
print("Classification Report:")
print(classification_report(test_labels, preds, target_names=dataset.classes))
print("Confusion Matrix:")
print(cm)
