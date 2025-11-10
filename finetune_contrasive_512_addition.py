import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import timm
import os
import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score
from tqdm import tqdm
from collections import Counter
from sklearn.metrics import roc_auc_score
# ============================================================
# Set Random Seeds for Reproducibility (FIX 1 - IMPROVED)
# ============================================================
def set_seed(seed=42, deterministic=True):
    """Set random seeds for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Guard CUDA calls - only if CUDA is available
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Optional deterministic mode (can slow training)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            torch.backends.cudnn.benchmark = True  # Faster but non-deterministic


# ============================================================
# Dataset Class for Contrastive Learning (FIXED)
# ============================================================
class VMMRdbContrastiveDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None, failure_log=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.failure_log = failure_log  # FIX 9: Optional failure logging

        # Create mapping: label -> list of image paths
        self.label_to_images = {}
        for path, label in zip(image_paths, labels):
            if label not in self.label_to_images:
                self.label_to_images[label] = []
            self.label_to_images[label].append(path)

        self.classes = list(self.label_to_images.keys())

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Anchor image
        img1_path = self.image_paths[idx]
        label = self.labels[idx]

        # Positive or negative pair
        if random.random() < 0.5:
            # FIXED: Ensure positive pair is NOT the same image
            same_class_images = [p for p in self.label_to_images[label] if p != img1_path]
            
            if len(same_class_images) > 0:
                # Valid positive pair available
                img2_path = random.choice(same_class_images)
                target = 1
            else:
                # Only one image in this class, create negative pair instead
                neg_label = random.choice([c for c in self.classes if c != label])
                img2_path = random.choice(self.label_to_images[neg_label])
                target = 0
        else:
            # Negative pair
            neg_label = random.choice([c for c in self.classes if c != label])
            img2_path = random.choice(self.label_to_images[neg_label])
            target = 0

        # FIX D & FIX 9: Robust image loading with optional failure logging
        try:
            img1 = Image.open(img1_path).convert("RGB")
        except Exception as e:
            if self.failure_log is not None:
                self.failure_log.append((img1_path, str(e)))
            else:
                print(f"⚠️  Warning: Failed to load {img1_path}: {e}. Using black image.")
            img1 = Image.new("RGB", (224, 224), (0, 0, 0))
        
        try:
            img2 = Image.open(img2_path).convert("RGB")
        except Exception as e:
            if self.failure_log is not None:
                self.failure_log.append((img2_path, str(e)))
            else:
                print(f"⚠️  Warning: Failed to load {img2_path}: {e}. Using black image.")
            img2 = Image.new("RGB", (224, 224), (0, 0, 0))

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, torch.tensor(target, dtype=torch.float32)


# ============================================================
# Contrastive Loss with Epsilon for Numerical Stability
# ============================================================
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0, eps=1e-9):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = eps

    def forward(self, output1, output2, label):
        euclidean_distance = torch.sqrt(
            torch.sum((output1 - output2) ** 2, dim=1) + self.eps
        )
        loss = torch.mean(
            label * torch.pow(euclidean_distance, 2) +
            (1 - label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        )
        return loss


# ============================================================
# Main Training Pipeline
# ============================================================
if __name__ == "__main__":
    # -----------------------------
    # CLI Arguments
    # -----------------------------
    import argparse
    parser = argparse.ArgumentParser(description='Train contrastive model on VMMRdb dataset')
    parser.add_argument('--dataset', '-d', default='archive', help='Path to archive root (default: archive)')
    parser.add_argument('--batch-size', '-b', type=int, default=16, help='Training batch size (default: 16)')
    parser.add_argument('--epochs', '-e', type=int, default=15, help='Number of training epochs (default: 15)')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate (default: 5e-4)')
    parser.add_argument('--no-pin-memory', action='store_true', help='Disable pin_memory in DataLoader')
    parser.add_argument('--min-samples', type=int, default=2, help='Minimum samples per class (default: 2)')
    parser.add_argument('--margin', type=float, default=1.0, help='Contrastive loss margin (default: 1.0)')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay (default: 1e-4)')
    parser.add_argument('--grad-clip', type=float, default=5.0, help='Gradient clipping max norm (default: 5.0)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--no-deterministic', action='store_true', 
                        help='Disable deterministic mode for faster training (less reproducible)')
    parser.add_argument('--embedding-dim', type=int, default=512, 
                        help='Embedding dimension for projection head (default: 512, recommended for vehicles)')
    parser.add_argument('--output', '-o', default='resnet50d_vmmrdb_contrastive_final.pth', 
                        help='Output model filename (default: resnet50d_vmmrdb_contrastive_final.pth)')
    parser.add_argument('--log-failures', action='store_true',
                        help='Log image loading failures to file (default: False)')
    args = parser.parse_args()

    # FIX 1: Set random seeds with optional deterministic mode
    set_seed(args.seed, deterministic=not args.no_deterministic)
    if args.no_deterministic:
        print(f"🎲 Random seed set to: {args.seed} (fast mode, non-deterministic)")
    else:
        print(f"🎲 Random seed set to: {args.seed} (deterministic mode)")
    
    # Initialize failure log if requested (FIX 9)
    failure_log = []
    if args.log_failures:
        print("📝 Image loading failures will be logged")

    # -----------------------------
    # Dataset Loading (FIXED: Efficient class collection)
    # -----------------------------
    vmmrdb_dir = args.dataset
    dataset_path = os.path.join(vmmrdb_dir, "Dataset")

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

    image_paths, labels = [], []
    class_names_set = set()

    print("\n📂 Loading dataset...")
    for make in sorted(os.listdir(dataset_path)):
        make_dir = os.path.join(dataset_path, make)
        if os.path.isdir(make_dir):
            for model in sorted(os.listdir(make_dir)):
                model_dir = os.path.join(make_dir, model)
                if os.path.isdir(model_dir):
                    class_name = f"{make}_{model}"
                    class_names_set.add(class_name)
                    
                    for img_file in os.listdir(model_dir):
                        if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                            image_paths.append(os.path.join(model_dir, img_file))
                            labels.append(class_name)

    # Sort for reproducibility
    class_names = sorted(list(class_names_set))
    class_to_idx = {cls: i for i, cls in enumerate(class_names)}
    label_indices = [class_to_idx[lbl] for lbl in labels]

    print(f"   Total images loaded: {len(image_paths)}")
    print(f"   Total classes: {len(class_names)}")

    # -----------------------------
    # Filter Classes with Too Few Samples (FIXED: Simplified logic)
    # -----------------------------
    label_counts = Counter(label_indices)
    
    # Identify classes with sufficient samples
    min_samples = max(args.min_samples, 2)  # Need at least 2 for train/test split
    valid_classes = {cls for cls, count in label_counts.items() if count >= min_samples}
    
    if len(valid_classes) < len(class_names):
        removed_count = len(class_names) - len(valid_classes)
        print(f"\n⚠️  Warning: Removing {removed_count} classes with fewer than {min_samples} samples")
        
        # Filter data
        filtered_data = [(path, lbl) for path, lbl in zip(image_paths, label_indices) 
                         if lbl in valid_classes]
        
        if len(filtered_data) < 10:
            raise ValueError(f"Too few samples after filtering ({len(filtered_data)}). "
                           f"Consider lowering --min-samples or checking your dataset.")
        
        image_paths, label_indices = zip(*filtered_data)
        image_paths = list(image_paths)
        label_indices = list(label_indices)
        
        print(f"   Filtered dataset size: {len(image_paths)} images")

    # -----------------------------
    # Train/Test Split with Stratification
    # -----------------------------
    try:
        train_paths, test_paths, train_labels, test_labels = train_test_split(
            image_paths, label_indices, test_size=0.2, stratify=label_indices, random_state=args.seed
        )
        print("\n✅ Successfully created stratified train/test split")
    except ValueError as e:
        print(f"\n⚠️  Stratified split failed: {e}")
        print("   Falling back to random split (non-stratified)")
        train_paths, test_paths, train_labels, test_labels = train_test_split(
            image_paths, label_indices, test_size=0.2, random_state=args.seed
        )

    print(f"   Training images: {len(train_paths)}")
    print(f"   Testing images: {len(test_paths)}")
    print(f"   Active classes: {len(set(train_labels))}")

    # -----------------------------
    # Data Transforms
    # -----------------------------
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # -----------------------------
    # Dataset & DataLoader
    # -----------------------------
    train_dataset = VMMRdbContrastiveDataset(train_paths, train_labels, train_transform, 
                                             failure_log if args.log_failures else None)
    test_dataset = VMMRdbContrastiveDataset(test_paths, test_labels, val_transform,
                                            failure_log if args.log_failures else None)

    pin_memory = not args.no_pin_memory and torch.cuda.is_available()
    
    # FIX 8: Improved num_workers selection
    num_workers = 0  # Safe default for Windows
    if os.name != 'nt' and torch.cuda.is_available():  # Linux/Mac with GPU
        num_workers = max(1, min(4, os.cpu_count() or 1))
        print(f"ℹ️  Using {num_workers} workers for data loading (Linux/Mac)")
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=pin_memory
    )

    # -----------------------------
    # Model Definition with Projection Head
    # -----------------------------
    print("\n🏗️  Initializing ResNet50d model with projection head...")
    backbone = timm.create_model("resnet50d.ra2_in1k", pretrained=True)
    
    # Get backbone output dimension
    if hasattr(backbone, 'fc'):
        backbone_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
    elif hasattr(backbone, 'head'):
        backbone_dim = backbone.head.in_features
        backbone.head = nn.Identity()
    else:
        print("⚠️  Warning: Could not find classification head, assuming 2048 dim")
        backbone_dim = 2048
    
    # Create projection head (2048 -> 512 by default)
    # FIX 2: Using LayerNorm instead of BatchNorm1d for stability with small batches
    class ProjectionHead(nn.Module):
        def __init__(self, in_dim, hidden_dim, out_dim):
            super().__init__()
            self.projection = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),  # More stable than BatchNorm for small batches
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, out_dim)
            )
        
        def forward(self, x):
            return self.projection(x)
    
    projection_head = ProjectionHead(
        in_dim=backbone_dim,
        hidden_dim=2048,  # Same as input for symmetry
        out_dim=args.embedding_dim
    )
    
    # Combine backbone + projection head
    class ContrastiveModel(nn.Module):
        def __init__(self, backbone, projection):
            super().__init__()
            self.backbone = backbone
            self.projection = projection
        
        def forward(self, x):
            features = self.backbone(x)
            embeddings = self.projection(features)
            return embeddings
    
    resnet_model = ContrastiveModel(backbone, projection_head)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Using device: {device}")
    print(f"   Backbone output: {backbone_dim}D → Projection head: {args.embedding_dim}D")
    resnet_model = resnet_model.to(device)

    # -----------------------------
    # Loss, Optimizer, Scheduler
    # -----------------------------
    criterion = ContrastiveLoss(margin=args.margin)
    optimizer = optim.AdamW(
        resnet_model.parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # -----------------------------
    # Training Configuration Summary
    # -----------------------------
    print(f"\n{'='*60}")
    print(f"🚀 TRAINING CONFIGURATION")
    print(f"{'='*60}")
    print(f"  Epochs:           {args.epochs}")
    print(f"  Batch size:       {args.batch_size}")
    print(f"  Learning rate:    {args.lr}")
    print(f"  Weight decay:     {args.weight_decay}")
    print(f"  Margin:           {args.margin}")
    print(f"  Gradient clip:    {args.grad_clip}")
    print(f"  Random seed:      {args.seed}")
    print(f"  Embedding dim:    {args.embedding_dim}D")
    print(f"  Pin memory:       {pin_memory}")
    print(f"  Num workers:      {num_workers}")
    print(f"{'='*60}\n")

    # -----------------------------
    # Helper Functions: Metrics Calculation
    # -----------------------------
    def calculate_cosine_similarity_accuracy(output1, output2, labels, threshold=0.7):
        """
        Calculate accuracy using cosine similarity (FIX B - Better metric).
        
        Args:
            output1: Normalized embeddings for first image
            output2: Normalized embeddings for second image
            labels: Binary labels (1 for same class, 0 for different)
            threshold: Similarity threshold (default: 0.7)
        
        Returns:
            accuracy: Percentage of correct predictions
            cosine_sim: Cosine similarity scores
        """
        # For L2-normalized vectors, cosine similarity = dot product
        cosine_sim = torch.sum(output1 * output2, dim=1)
        
        # Predict: if similarity > threshold, predict same class (1)
        predictions = (cosine_sim > threshold).float()
        
        # Calculate accuracy
        correct = (predictions == labels).float().sum()
        accuracy = correct / labels.size(0)
        
        return accuracy.item(), cosine_sim
    
    def compute_optimal_threshold(similarities, labels):
        """
        Compute optimal threshold using Youden's J statistic (FIX B).
        
        Args:
            similarities: List of similarity scores
            labels: List of ground truth labels
        
        Returns:
            best_threshold: Optimal threshold
            best_j_score: Best Youden's J score
        """
        try:
            # FIX 4: Check for label variety
            if len(set(labels)) < 2:
                print("⚠️  Warning: All labels are the same, cannot compute ROC. Using default threshold.")
                return 0.7, 0.0
            
            fpr, tpr, thresholds = roc_curve(labels, similarities)
            j_scores = tpr - fpr
            best_idx = np.argmax(j_scores)
            return thresholds[best_idx], j_scores[best_idx]
        except Exception as e:
            print(f"⚠️  Warning: ROC computation failed ({e}). Using default threshold 0.7")
            return 0.7, 0.0  # Default fallback

    # -----------------------------
    # Training Loop
    # -----------------------------
    best_val_loss = float('inf')
    best_val_acc = 0.0
    best_threshold = 0.7  # Will be updated based on validation
    best_model_state = None

    for epoch in range(args.epochs):
        # ==================== Training Phase ====================
        resnet_model.train()
        running_loss = 0.0
        # FIX 5: Per-sample accuracy tracking (more accurate)
        correct_samples = 0
        total_samples = 0

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for img1, img2, labels in train_pbar:
            # FIX B: Explicit dtype handling for labels
            img1 = img1.to(device)
            img2 = img2.to(device)
            labels = labels.to(device, dtype=torch.float32)  # Explicit float32 for loss

            optimizer.zero_grad()
            
            # Forward pass
            emb1 = resnet_model(img1)
            emb2 = resnet_model(img2)

            # L2 normalize embeddings
            out1 = nn.functional.normalize(emb1, p=2, dim=1)
            out2 = nn.functional.normalize(emb2, p=2, dim=1)

            # Compute loss
            loss = criterion(out1, out2, labels)
            
            # Backward pass
            loss.backward()
            
            # FIX C: Gradient clipping with configurable max_norm (default 5.0)
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(resnet_model.parameters(), max_norm=args.grad_clip)
            
            optimizer.step()
            
            # FIX 5: Calculate per-sample accuracy using cosine similarity
            with torch.no_grad():
                cosine_sim = torch.sum(out1 * out2, dim=1)
                predictions = (cosine_sim > best_threshold).float()
                correct_samples += (predictions == labels).sum().item()
                total_samples += labels.size(0)
                batch_acc = (predictions == labels).float().mean().item()
            
            running_loss += loss.item()
            
            # Update progress bar with both loss and accuracy
            train_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{batch_acc:.4f}'
            })

        avg_train_loss = running_loss / len(train_loader)
        avg_train_acc = correct_samples / total_samples  # FIX 5: Per-sample accuracy

        # ==================== Validation Phase ====================
        resnet_model.eval()
        val_loss = 0.0
        val_accuracy = 0.0
        
        # Collect all similarities and labels for threshold optimization (FIX B)
        all_similarities = []
        all_labels = []
        
        with torch.no_grad():
            val_pbar = tqdm(test_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]  ")
            for img1, img2, labels in val_pbar:
                # FIX B: Explicit dtype handling for labels
                img1 = img1.to(device)
                img2 = img2.to(device)
                labels = labels.to(device, dtype=torch.float32)  # Explicit float32 for loss
                
                emb1 = resnet_model(img1)
                emb2 = resnet_model(img2)
                
                out1 = nn.functional.normalize(emb1, p=2, dim=1)
                out2 = nn.functional.normalize(emb2, p=2, dim=1)
                
                loss = criterion(out1, out2, labels)
                val_loss += loss.item()
                
                # Calculate accuracy using cosine similarity
                batch_acc, cosine_sims = calculate_cosine_similarity_accuracy(out1, out2, labels, threshold=best_threshold)
                val_accuracy += batch_acc
                
                # Store for threshold optimization
                all_similarities.extend(cosine_sims.cpu().numpy().tolist())
                all_labels.extend(labels.cpu().numpy().tolist())
                
                # Update progress bar with both loss and accuracy
                val_pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{batch_acc:.4f}'
                })
        
        avg_val_loss = val_loss / len(test_loader)
        avg_val_acc = val_accuracy / len(test_loader)
        
        # Compute optimal threshold from validation set (FIX B)
        optimal_threshold, j_score = compute_optimal_threshold(all_similarities, all_labels)
        
        # Recalculate accuracy with optimal threshold
        predictions_optimal = [1 if s > optimal_threshold else 0 for s in all_similarities]
        acc_optimal = sum([1 for p, l in zip(predictions_optimal, all_labels) if p == l]) / len(all_labels)
        
        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # Print epoch summary with accuracy
        # FIX 3: Clarify that displayed acc uses current threshold, optimal acc uses ROC-based threshold
        print(f"Epoch [{epoch+1}/{args.epochs}] - "
              f"Train Loss: {avg_train_loss:.4f} | Train Acc: {avg_train_acc:.4f} (thresh={best_threshold:.3f}) | "
              f"Val Loss: {avg_val_loss:.4f} | Val Acc: {avg_val_acc:.4f} | "
              f"Optimal Acc: {acc_optimal:.4f} (thresh={optimal_threshold:.3f}) | "
              f"LR: {current_lr:.6f}")
        
        # SANITY CHECK: Display separation metrics
        try:
            pos_sims = [s for s, l in zip(all_similarities, all_labels) if l == 1]
            neg_sims = [s for s, l in zip(all_similarities, all_labels) if l == 0]
            mean_pos_sim = np.mean(pos_sims) if pos_sims else 0
            mean_neg_sim = np.mean(neg_sims) if neg_sims else 0
            sim_separation = mean_pos_sim - mean_neg_sim
            auc_score = roc_auc_score(all_labels, all_similarities) if len(set(all_labels)) > 1 else 0
        except Exception:
            mean_pos_sim, mean_neg_sim, sim_separation, auc_score = 0, 0, 0, 0
        print(f"  📊 Pos Sim: {mean_pos_sim:.3f} | Neg Sim: {mean_neg_sim:.3f} | "
              f"Separation: {sim_separation:.3f} | AUC: {auc_score:.4f}")

        # Save best model based on validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_val_acc = acc_optimal  # Use optimal accuracy
            best_threshold = optimal_threshold  # Update best threshold
            # FIX D: Deep copy to CPU to avoid GPU memory issues and ensure immutability
            best_model_state = {k: v.cpu().clone() for k, v in resnet_model.state_dict().items()}
            print(f"  ✓ New best validation loss: {best_val_loss:.4f} (Optimal Acc: {best_val_acc:.4f}, Threshold: {best_threshold:.3f})")
        
        print()  # Empty line for readability

    # -----------------------------
    # Save Best Model & Log Failures
    # -----------------------------
    print(f"\n{'='*60}")
    
    # FIX 9: Save failure log if requested
    if args.log_failures and len(failure_log) > 0:
        failure_log_path = args.output.replace('.pth', '_failures.txt')
        with open(failure_log_path, 'w') as f:
            f.write(f"Image Loading Failures ({len(failure_log)} total)\n")
            f.write("="*60 + "\n\n")
            for path, error in failure_log:
                f.write(f"{path}\n  Error: {error}\n\n")
        print(f"⚠️  {len(failure_log)} image loading failures logged to: {failure_log_path}")
        print()
    if best_model_state is not None:
        # Save model with metadata
        checkpoint = {
            'model_state_dict': best_model_state,
            'best_val_loss': best_val_loss,
            'best_val_acc': best_val_acc,
            'optimal_threshold': best_threshold,
            'margin': args.margin,
            'embedding_dim': args.embedding_dim,
            'architecture': 'resnet50d',
        }
        torch.save(checkpoint, args.output)
        print(f"✅ Training complete! Best model saved as '{args.output}'")
        print(f"   Best validation loss: {best_val_loss:.4f}")
        print(f"   Best validation accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
        print(f"   Optimal similarity threshold: {best_threshold:.3f}")
    else:
        # FIX D: Save to CPU
        final_state = {k: v.cpu().clone() for k, v in resnet_model.state_dict().items()}
        checkpoint = {
            'model_state_dict': final_state,
            'embedding_dim': args.embedding_dim,
            'architecture': 'resnet50d',
        }
        torch.save(checkpoint, args.output)
        print(f"✅ Training complete! Final model saved as '{args.output}'")
    print(f"{'='*60}\n")
    
    print("📝 Usage notes:")
    print("   - To load: checkpoint = torch.load('model.pth')")
    print("   - Then: model.load_state_dict(checkpoint['model_state_dict'])")
    print("   - For inference: Apply L2 normalization to embeddings")
    print("   - Compute similarity: Use cosine similarity (dot product of normalized vectors)")
    print(f"   - Optimal threshold for same/different class: {best_threshold:.3f}")
    print("   - Classification: similarity > threshold → same class, else different class")
    print("\n💡 Performance tips:")
    print("   - Use --no-deterministic for ~20% faster training (less reproducible)")
    print("   - Increase --batch-size to 32 or 64 if you have GPU memory")
    print("   - Use --log-failures to identify corrupted images in your dataset")
    
    print("\n📊 Sanity Check Interpretation:")
    print("   - Pos Sim: Should be high (>0.7 for good separation)")
    print("   - Neg Sim: Should be low (<0.3 for good separation)")
    print("   - Separation: Should be >0.4 (larger = better)")
    print("   - AUC: Should be >0.9 (perfect = 1.0)")
    print()