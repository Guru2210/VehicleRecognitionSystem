import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import timm
import os
from sklearn.model_selection import train_test_split

# -----------------------------
# Dataset Class
# (Unchanged)
# -----------------------------
class VMMRdbDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


# -----------------------------
# ArcFace Layer
# (Unchanged)
# -----------------------------
class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.5):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings, labels=None):
        cosine = nn.functional.linear(
            nn.functional.normalize(embeddings),
            nn.functional.normalize(self.weight)
        )
        # Safe clamp to avoid NaNs
        cosine = torch.clamp(cosine, -1.0 + 1e-6, 1.0 - 1e-6)

        if labels is None:
            return cosine * self.s

        theta = torch.acos(cosine)
        target_logits = torch.cos(theta + self.m)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)

        output = self.s * (one_hot * target_logits + (1.0 - one_hot) * cosine)
        return output


# -----------------------------
# Windows-safe entry point
# -----------------------------
if __name__ == "__main__":
    dataset_path = "D:/Freelancing/Vehicle Recognition with VMMRdb vectorized/archive/Dataset"
    image_paths, labels, class_names = [], [], []

    # Load images
    print("Loading dataset...")
    for make in os.listdir(dataset_path):
        make_dir = os.path.join(dataset_path, make)
        if os.path.isdir(make_dir):
            for model in os.listdir(make_dir):
                model_dir = os.path.join(make_dir, model)
                if os.path.isdir(model_dir):
                    class_name = f"{make}_{model}"
                    class_names.append(class_name)
                    for img_file in os.listdir(model_dir):
                        if img_file.endswith(".jpg"):
                            image_paths.append(os.path.join(model_dir, img_file))
                            labels.append(class_name)

    class_to_idx = {cls: i for i, cls in enumerate(class_names)}
    label_indices = [class_to_idx[lbl] for lbl in labels]

    train_paths, test_paths, train_labels, test_labels = train_test_split(
        image_paths, label_indices, test_size=0.2, stratify=label_indices, random_state=42
    )

    print(f"Training images: {len(train_paths)}, Testing images: {len(test_paths)}, Classes: {len(class_names)}")

    # -----------------------------
    # Transforms
    # (MODIFIED: Added Data Augmentation to train_transform)
    # -----------------------------
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),  # Flips image
        transforms.RandomRotation(15),         # Rotates image
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), # Tweaks lighting
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # -----------------------------
    # Datasets & Dataloaders
    # -----------------------------
    train_dataset = VMMRdbDataset(train_paths, train_labels, train_transform)
    test_dataset = VMMRdbDataset(test_paths, test_labels, val_transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    # -----------------------------
    # Model & ArcFace
    # (MODIFIED: Added Dropout layer)
    # -----------------------------
    resnet_model = timm.create_model("resnet50d.ra2_in1k", pretrained=True)
    if hasattr(resnet_model, 'fc'):
        resnet_model.fc = nn.Identity()
    elif hasattr(resnet_model, 'head'):
        resnet_model.head = nn.Identity()

    feature_dim = 2048
    num_classes = len(class_names)
    
    # NEW: Dropout layer for regularization
    dropout_layer = nn.Dropout(p=0.2) # 20% dropout
    
    arc_margin = ArcMarginProduct(in_features=feature_dim, out_features=num_classes, s=30.0, m=0.5)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resnet_model = resnet_model.to(device)
    dropout_layer = dropout_layer.to(device) # Move dropout to device
    arc_margin = arc_margin.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        # NEW: Add dropout parameters to optimizer
        list(resnet_model.parameters()) + list(dropout_layer.parameters()) + list(arc_margin.parameters()),
        lr=1e-4, weight_decay=1e-5
    )

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # -----------------------------
    # Training Loop
    # (MODIFIED: Added Dropout and Early Stopping)
    # -----------------------------
    num_epochs = 40
    
    # NEW: Variable for Early Stopping
    best_accuracy = 0.0
    
    print(f"\nStarting ArcFace fine-tuning on {device}...\n")

    for epoch in range(num_epochs):
        # Training
        resnet_model.train()
        dropout_layer.train() # Set dropout to train mode
        arc_margin.train()
        running_loss = 0.0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device).long().view(-1)
            optimizer.zero_grad()
            
            embeddings = resnet_model(imgs)
            embeddings = dropout_layer(embeddings) # NEW: Apply dropout
            
            logits = arc_margin(embeddings, labels)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        scheduler.step()

        # Validation
        resnet_model.eval()
        dropout_layer.eval() # NEW: Set dropout to eval mode
        arc_margin.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs, labels = imgs.to(device), labels.to(device).long().view(-1)
                embeddings = resnet_model(imgs)
                # Note: No dropout during validation
                logits = arc_margin(embeddings)  # Inference mode
                _, predicted = torch.max(logits, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {avg_loss:.4f} - Val Accuracy: {accuracy:.2f}%")

        # --- NEW: Early Stopping Logic ---
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            print(f"  🎉 New best model! Saving to 'best_model.pth' with {accuracy:.2f}% accuracy.")
            torch.save({
                'backbone': resnet_model.state_dict(),
                'arc_margin': arc_margin.state_dict(),
                'class_to_idx': class_to_idx
            }, "best_model.pth") # Save the *best* model

    print(f"\n✅ Training complete. Best model saved to 'best_model.pth' with {best_accuracy:.2f}% accuracy.")