import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
from torch.utils.data import Subset
import numpy as np

# --- 1. Data Augmentations ---
# SimCLR, like BYOL, relies on two augmented views of the same image.
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
    # Augmentation pipeline, same as used in the BYOL example
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

    # Create a subset of the dataset if specified
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
class SimCLR(nn.Module):
    def __init__(self, backbone, projection_dim=128):
        super().__init__()
        self.backbone = backbone
        
        # Get the feature dimension from the original fc layer
        backbone_in_features = self.backbone.fc.in_features
        # Replace the fc layer with an Identity layer
        self.backbone.fc = nn.Identity()

        # Add a projection head (MLP)
        self.projector = nn.Sequential(
            nn.Linear(backbone_in_features, backbone_in_features),
            nn.ReLU(),
            nn.Linear(backbone_in_features, projection_dim)
        )

    def forward(self, x):
        features = self.backbone(x)
        projections = self.projector(features)
        return projections

# --- 3. Loss Function (NT-Xent) ---
class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5, batch_size=512, device='cuda'):
        super().__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def forward(self, z_i, z_j):
        """
        z_i and z_j are the projections of the two augmented views.
        Shape: [batch_size, projection_dim]
        """
        # Concatenate the projections
        z = torch.cat((z_i, z_j), dim=0)

        # Calculate cosine similarity
        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature

        # Create the labels for contrastive loss
        # The positive pairs are (i, i+batch_size) and (i+batch_size, i)
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(self.batch_size * 2, 1)
        
        # *** FIX: The mask for indexing must be a boolean tensor, not a float. Removed .float() ***
        mask = (~torch.eye(self.batch_size * 2, self.batch_size * 2, dtype=bool)).to(self.device)
        negative_samples = sim[mask].reshape(self.batch_size * 2, -1)

        # Combine positive and negative samples for CrossEntropyLoss
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        labels = torch.zeros(self.batch_size * 2).to(self.device).long()

        loss = self.criterion(logits, labels)
        loss /= (2 * self.batch_size)
        return loss


# --- 4. Training ---
def train_simclr(epochs=100, batch_size=512, subset_percentage=1.0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get the data loader
    train_loader = get_cifar10_dataloader(batch_size, subset_percentage=subset_percentage)

    # Create the ResNet backbone
    backbone = resnet18(weights=None, num_classes=10)
    backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    backbone.maxpool = nn.Identity()

    # Create the SimCLR model
    model = SimCLR(backbone).to(device)

    # Create the loss function and optimizer
    loss_fn = NTXentLoss(temperature=0.5, batch_size=batch_size, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    print("Starting SimCLR training...")
    for epoch in range(epochs):
        total_loss = 0
        if len(train_loader) == 0:
            print(f"Skipping epoch {epoch+1} due to empty dataloader.")
            continue

        for i, (images, _) in enumerate(train_loader):
            images1 = images[0].to(device)
            images2 = images[1].to(device)

            # Forward pass
            proj1 = model(images1)
            proj2 = model(images2)

            # Calculate loss
            loss = loss_fn(proj1, proj2)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

    print("SimCLR training finished.")
    # You can now save the trained encoder (backbone) for downstream tasks
    # torch.save(model.backbone.state_dict(), 'simclr_backbone.pth')

if __name__ == '__main__':
    # --- Use only 10% of the dataset for a quick run ---
    train_simclr(epochs=50, batch_size=256, subset_percentage=0.1)