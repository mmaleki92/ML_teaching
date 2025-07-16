import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
from torch.utils.data import Subset
import numpy as np
import copy

# --- 1. Data Augmentations ---
# BYOL relies on two augmented views of the same image.
# We'll use the same augmentation pipeline as SimCLR.
class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]

def get_cifar10_dataloader(batch_size=512, num_workers=4, subset_percentage=1.0):
    """
    Returns a DataLoader for the CIFAR-10 dataset.
    Can use a subset of the data if subset_percentage < 1.0.
    """
    # Augmentation pipeline
    transform = transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])

    # Load the full CIFAR-10 training dataset
    full_train_dataset = torchvision.datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=TwoCropTransform(transform)
    )

    # --- NEW: Create a subset of the dataset ---
    if subset_percentage < 1.0:
        num_samples = int(len(full_train_dataset) * subset_percentage)
        indices = np.random.choice(len(full_train_dataset), num_samples, replace=False)
        train_dataset = Subset(full_train_dataset, indices)
        print(f"Using a subset of {num_samples} images ({subset_percentage*100:.1f}% of the training data).")
    else:
        train_dataset = full_train_dataset
        print(f"Using the full training dataset with {len(train_dataset)} images.")


    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    return train_loader

# --- 2. Model Architecture ---

# MLP for projection and prediction heads
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

# BYOL model
class BYOL(nn.Module):
    def __init__(self, backbone, backbone_in_features, projection_size=256, projection_hidden_size=4096, prediction_size=256, prediction_hidden_size=4096, momentum=0.99):
        super().__init__()
        self.momentum = momentum

        # --- Online Network ---
        self.online_encoder = backbone
        # The projection head
        self.online_projector = MLP(backbone_in_features, projection_hidden_size, projection_size)
        # The prediction head
        self.online_predictor = MLP(projection_size, prediction_hidden_size, prediction_size)

        # --- Target Network ---
        # The target network has the same architecture as the online network
        self.target_encoder = copy.deepcopy(self.online_encoder)
        self.target_projector = copy.deepcopy(self.online_projector)

        # Freeze the target network
        for param in self.target_encoder.parameters():
            param.requires_grad = False
        for param in self.target_projector.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def _update_target_network(self):
        """
        Momentum update of the target network's weights.
        """
        for param_q, param_k in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1. - self.momentum)

        for param_q, param_k in zip(self.online_projector.parameters(), self.target_projector.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1. - self.momentum)

    def forward(self, x1, x2):
        # --- Online Network Forward Pass ---
        online_y1 = self.online_encoder(x1)
        online_z1 = self.online_projector(online_y1)
        online_p1 = self.online_predictor(online_z1)

        online_y2 = self.online_encoder(x2)
        online_z2 = self.online_projector(online_y2)
        online_p2 = self.online_predictor(online_z2)

        # --- Target Network Forward Pass ---
        with torch.no_grad():
            self._update_target_network() # Update the target network
            target_y1 = self.target_encoder(x1)
            target_z1 = self.target_projector(target_y1)

            target_y2 = self.target_encoder(x2)
            target_z2 = self.target_projector(target_y2)

        return (online_p1, online_p2), (target_z1.detach(), target_z2.detach())

# --- 3. Loss Function ---
def byol_loss_fn(p, z):
    p = F.normalize(p, dim=-1)
    z = F.normalize(z, dim=-1)
    return 2 - 2 * (p * z).sum(dim=-1)

# --- 4. Training ---
def train_byol(epochs=100, batch_size=512, subset_percentage=1.0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get the data loader with the specified subset percentage
    train_loader = get_cifar10_dataloader(batch_size, subset_percentage=subset_percentage)

    # Create the ResNet backbone
    # We modify the first conv layer for CIFAR-10 as it's a small dataset
    backbone = resnet18(weights=None, num_classes=10)
    backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    backbone.maxpool = nn.Identity()
    
    # Get the feature dimension from the original fc layer BEFORE replacing it
    backbone_in_features = backbone.fc.in_features
    backbone.fc = nn.Identity() # We will use our own projector and predictor

    # Pass the feature dimension to the BYOL constructor
    model = BYOL(backbone, backbone_in_features=backbone_in_features).to(device)

    # Create the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    print("Starting BYOL training...")
    for epoch in range(epochs):
        total_loss = 0
        # Check if the dataloader is empty (can happen with very small subsets and large batches)
        if len(train_loader) == 0:
            print(f"Skipping epoch {epoch+1} due to empty dataloader. Your subset might be too small for the batch size.")
            continue

        for i, (images, _) in enumerate(train_loader):
            images1 = images[0].to(device)
            images2 = images[1].to(device)

            # Forward pass
            (p1, p2), (z1, z2) = model(images1, images2)

            # Calculate loss
            loss = byol_loss_fn(p1, z2) + byol_loss_fn(p2, z1)
            loss = loss.mean()

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

    print("BYOL training finished.")
    # You can now save the trained encoder for downstream tasks
    # torch.save(model.online_encoder.state_dict(), 'byol_encoder.pth')

if __name__ == '__main__':
    # --- Use only 10% of the dataset for a quick run ---
    train_byol(epochs=20, subset_percentage=0.1)
