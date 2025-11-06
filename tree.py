import os
import numpy as np
import joblib
import torch
import torch.nn as nn
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import logging

os.makedirs("tree", exist_ok=True)
logging.basicConfig(filename="tree/tree.log", level=logging.INFO, format="%(asctime)s - %(message)s")

data_dir = "1_3/data"
batch_size = 32
num_workers = 13
random_state = 13
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
logging.info(f"Device: {device}")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

full_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
num_classes = len(full_dataset.classes)
labels = [label for _, label in full_dataset.samples]

train_idx, test_idx = train_test_split(
    np.arange(len(full_dataset)),
    test_size=0.2,
    stratify=labels,
    random_state=random_state
)

train_dataset = Subset(full_dataset, train_idx)
test_dataset = Subset(full_dataset, test_idx)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

logging.info(f"Train size: {len(train_dataset)}, Test size: {len(test_dataset)}")

resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
resnet50.eval()
resnet50 = resnet50.to(device)
for p in resnet50.parameters():
    p.requires_grad = False

def extract_features_from_loader(model, loader):
    feats, labels = [], []
    with torch.no_grad():
        for imgs, labs in loader:
            imgs = imgs.to(device)
            x = model.conv1(imgs)
            x = model.bn1(x)
            x = model.relu(x)
            x = model.maxpool(x)
            x = model.layer1(x)
            x = model.layer2(x)
            x = model.layer3(x)
            x = model.layer4(x)
            x = model.avgpool(x)
            x = torch.flatten(x, 1)
            feats.append(x.cpu().numpy())
            labels.append(labs.numpy())
    return np.concatenate(feats), np.concatenate(labels)

logging.info("feature extraction started")
X_train, y_train = extract_features_from_loader(resnet50, train_loader)
logging.info("feature extraction completed")
X_test, y_test = extract_features_from_loader(resnet50, test_loader)
logging.info(f"Feature shapes: Train={X_train.shape}, Test={X_test.shape}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, "tree/resnet_penultimate_scaler.pkl")
logging.info("Scaler saved")

param_grid = {
    "criterion": ["gini", "entropy"],
    "max_depth": [None, 5, 10, 20, 50],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}

dt = DecisionTreeClassifier(random_state=random_state)
grid = GridSearchCV(dt, param_grid, cv=5, n_jobs=-1, verbose=2)
logging.info("starting DT hyperparameter tuning")
grid.fit(X_train_scaled, y_train)
best_dt = grid.best_estimator_
joblib.dump(best_dt, "tree/best_resnet_dt.pkl")
logging.info(f"Best DT Params: {grid.best_params_}")
logging.info(f"Best CV Score: {grid.best_score_:.4f}")

y_pred = best_dt.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=full_dataset.classes)
cm = confusion_matrix(y_test, y_pred)

logging.info(f"Test Accuracy: {acc:.4f}")
logging.info("Classification Report:\n" + report)
logging.info(f"Confusion Matrix:\n{cm}")

with open("tree/dt_classification_report.txt", "w") as f:
    f.write("Best DT Params:\n")
    f.write(str(grid.best_params_) + "\n\n")
    f.write(f"Test Accuracy: {acc:.4f}\n\n")
    f.write(report)

plt.figure(figsize=(10, 8))
plt.imshow(cm, interpolation='nearest')
plt.title(f"Confusion Matrix (Acc={acc:.4f})")
plt.colorbar()
tick_marks = np.arange(len(full_dataset.classes))
plt.xticks(tick_marks, full_dataset.classes, rotation=45, ha="right")
plt.yticks(tick_marks, full_dataset.classes)
thresh = cm.max() / 2.
for i, j in np.ndindex(cm.shape):
    plt.text(j, i, format(cm[i, j], 'd'),
             horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black")
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.savefig("tree/confusion_matrix_resnet_dt.png", bbox_inches="tight")
plt.close()
logging.info("Confusion matrix saved")

plt.figure(figsize=(20, 12))
plot_tree(best_dt, filled=True, feature_names=[f"feat_{i}" for i in range(X_train.shape[1])],
          class_names=full_dataset.classes, max_depth=4, fontsize=8)
plt.title("Decision Tree (top levels)")
plt.savefig("tree/decision_tree_diagram.png", bbox_inches="tight")
plt.close()
logging.info("Decision tree diagram saved")

print("done")
print("Best DT Params:", grid.best_params_)
print("Test Acc:", acc)
logging.info("Training completed successfully")
