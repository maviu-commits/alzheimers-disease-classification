import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import getpass

def main():

    # -------- PATH --------
    user = getpass.getuser()
    data_dir = f"/home/{user}/ad_mri_project/data/archive/AugmentedAlzheimerDataset"

    # -------- TRANSFORMS --------
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor()
    ])

    # -------- DATA --------
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)

    print("Classes:", dataset.classes)
    print("Total images:", len(dataset))

    # -------- SPLIT --------
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

    # -------- MODEL --------
    model = models.resnet50(weights="IMAGENET1K_V1")

    # 🔥 IMPROVED CLASSIFIER (THIS IS KEY)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 512),
        nn.ReLU(),
        nn.BatchNorm1d(512),
        nn.Dropout(0.5),
        nn.Linear(512, 4)
    )

    # -------- DEVICE --------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # -------- LOSS --------
    criterion = nn.CrossEntropyLoss()

    # 🔥 OPTIMIZER (stable)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    epochs = 10   # enough to reach 96

    best_acc = 0

    for epoch in range(epochs):

        model.train()
        running_loss = 0

        print(f"\nEpoch {epoch+1}")

        for batch_idx, (images, labels) in enumerate(train_loader):

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if batch_idx % 50 == 0:
                print(f"Batch {batch_idx} Loss {loss.item():.4f}")

        print(f"Total Loss: {running_loss:.4f}")

        # -------- VALIDATION --------
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:

                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                _, predicted = torch.max(outputs, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        acc = 100 * correct / total
        print(f"Validation Accuracy: {acc:.2f}%")

        # 🔥 SAVE BEST MODEL
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "alz_model.pth")

    print("\nBest Accuracy:", best_acc)
    print("Model saved")

if __name__ == "__main__":
    main()
