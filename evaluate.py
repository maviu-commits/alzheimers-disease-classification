import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# -------- DATA PATH (WINDOWS) --------
data_dir = r"C:\Users\indum\Downloads\archive (2)\AugmentedAlzheimerDataset"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading data...")

# -------- TRANSFORM --------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# -------- LOAD FULL DATA --------
full_dataset = datasets.ImageFolder(data_dir, transform=transform)
class_names = full_dataset.classes

print("Classes:", class_names)

# -------- BALANCED SAMPLING (FIX) --------
class_indices = {i: [] for i in range(len(class_names))}

for idx, (_, label) in enumerate(full_dataset):
    class_indices[label].append(idx)

selected_indices = []

# 🔥 Equal samples from each class
for cls in class_indices:
    selected_indices += class_indices[cls][:500]

dataset = torch.utils.data.Subset(full_dataset, selected_indices)

loader = DataLoader(dataset, batch_size=32, shuffle=False)

print("Data loaded")

# -------- MODEL --------
model = models.resnet50(weights=None)

model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 512),
    nn.ReLU(),
    nn.BatchNorm1d(512),
    nn.Dropout(0.5),
    nn.Linear(512, 4)
)

model = model.to(device)
model.load_state_dict(torch.load("alz_model.pth", map_location=device))
model.eval()

print("Model loaded")

# -------- EVALUATION --------
y_true = []
y_pred = []

with torch.no_grad():
    for i, (images, labels) in enumerate(loader):

        if i % 20 == 0:
            print(f"Processing batch {i}")

        images = images.to(device)

        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        y_true.extend(labels.numpy())
        y_pred.extend(preds.cpu().numpy())

# -------- RESULTS --------
acc = accuracy_score(y_true, y_pred)
print("\nFinal Accuracy:", acc * 100)

cm = confusion_matrix(y_true, y_pred)

print("\nConfusion Matrix:\n", cm)

print("\nClassification Report:\n")
print(classification_report(y_true, y_pred, target_names=class_names))

# -------- GRAPH --------
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix Heatmap")

plt.savefig("confusion_matrix.png")
print("\nGraph saved as confusion_matrix.png")

plt.show()