import os
import shutil
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

DATA_DIR = "Brain_Tumour_Detection" 
TRAIN_DIR = "TRAIN/"
VAL_DIR = "VAL/"
TEST_DIR = "TEST/"
RESULTS_DIR = "results/"
SAMPLE_DIR = "sample_images/" 
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(SAMPLE_DIR, exist_ok=True)

classes = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
for split in ['TRAIN', 'VAL', 'TEST']:
    for class_name in classes:
        os.makedirs(f"{split}/{class_name}", exist_ok=True)

for class_name in classes:
    img_files = os.listdir(os.path.join(DATA_DIR, class_name))
    np.random.shuffle(img_files)
    
    sample_image = img_files[0]
    shutil.copy(os.path.join(DATA_DIR, class_name, sample_image), os.path.join(SAMPLE_DIR, sample_image))
    
    train_split = int(0.7 * len(img_files))
    val_split = int(0.85 * len(img_files))
    
    for i, file in enumerate(img_files):
        src_path = os.path.join(DATA_DIR, class_name, file)
        
        if i < train_split:
            dest_path = os.path.join(TRAIN_DIR, class_name, file)
        elif i < val_split:
            dest_path = os.path.join(VAL_DIR, class_name, file)
        else:
            dest_path = os.path.join(TEST_DIR, class_name, file)
        
        shutil.copy(src_path, dest_path)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_data = ImageFolder(TRAIN_DIR, transform=transform)
val_data = ImageFolder(VAL_DIR, transform=transform)
test_data = ImageFolder(TEST_DIR, transform=transform)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=16, shuffle=False)
test_loader = DataLoader(test_data, batch_size=16, shuffle=False)

#model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
model = models.resnet50(weights=None)

num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_ftrs, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, len(classes)),
    nn.Softmax(dim=1)
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)


def train_model(model, train_loader, val_loader, patience=10, target_acc=0.80):
    best_val_acc = 0
    no_improve_epochs = 0
    epoch = 0
    reached_target = False
    
    while True:
        epoch += 1
        model.train()
        running_loss = 0
        correct, total = 0, 0
        
        for images, labels in tqdm(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        
        train_acc = correct / total
        val_acc = evaluate_model(model, val_loader)
        print(f"Epoch {epoch}: Train Loss: {running_loss/len(train_loader):.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
        
        if val_acc >= target_acc:
            reached_target = True 
        
        if reached_target and no_improve_epochs >= patience:
            print("Early stopping: Validation accuracy has not improved for 10 consecutive epochs after reaching 80%.")
            break
        
        if val_acc > best_val_acc:
            torch.save({
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict()
            }, "best_model.pth")
            best_val_acc = val_acc
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

def evaluate_model(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total

train_model(model, train_loader, val_loader, patience=10, target_acc=0.80)

model.load_state_dict(torch.load("best_model.pth", map_location=device)["model_state"])

def test_model(model, test_loader):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.cpu().numpy()
            outputs = model(images).cpu().numpy()
            preds = np.argmax(outputs, axis=1)
            y_true.extend(labels)
            y_pred.extend(preds)
    return np.array(y_true), np.array(y_pred)

y_true, y_pred = test_model(model, test_loader)

acc = accuracy_score(y_true, y_pred)
cm = confusion_matrix(y_true, y_pred)
report = classification_report(y_true, y_pred, target_names=classes)
print(f"Test Accuracy: {acc:.2f}")
print(report)

plt.figure(figsize=(6,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"))
