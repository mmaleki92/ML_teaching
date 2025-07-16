import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset
import numpy as np
import copy
import matplotlib.pyplot as plt
from einops import rearrange
from einops.layers.torch import Rearrange

# --- 1. Vision Transformer (ViT) Components ---

class PatchEmbedding(nn.Module):
    """Splits image into patches and embeds them."""
    def __init__(self, in_channels=3, patch_size=4, emb_size=512):
        super().__init__()
        self.patch_size = patch_size
        # This layer will split the image into patches and perform a convolution
        self.projection = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear(patch_size * patch_size * in_channels, emb_size)
        )

    def forward(self, x):
        x = self.projection(x)
        return x

class MultiHeadAttention(nn.Module):
    """Multi-Head Attention mechanism."""
    def __init__(self, emb_size=512, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.scale = emb_size ** -0.5
        # Q, K, V projections
        self.to_qkv = nn.Linear(emb_size, emb_size * 3, bias=False)
        # Output projection
        self.out = nn.Linear(emb_size, emb_size)
        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(dropout)
        # Store attention weights
        self.attn_weights = None

    def forward(self, x):
        b, n, _, h = *x.shape, self.num_heads
        # Project to Q, K, V
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        # Rearrange for multi-head attention
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        # Calculate attention
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        attn = dots.softmax(dim=-1)
        # Store attention weights for visualization
        self.attn_weights = attn
        attn = self.attn_dropout(attn)

        # Apply attention to values
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.out(out)
        out = self.dropout(out)
        return out

class TransformerEncoderBlock(nn.Module):
    """A single block of the Transformer Encoder."""
    def __init__(self, emb_size=512, num_heads=8, forward_expansion=4, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(emb_size, num_heads, dropout)
        self.norm1 = nn.LayerNorm(emb_size)
        self.norm2 = nn.LayerNorm(emb_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(emb_size, forward_expansion * emb_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(forward_expansion * emb_size, emb_size),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.feed_forward(self.norm2(x))
        return x

class VisionTransformer(nn.Module):
    """Vision Transformer model."""
    def __init__(self, *, image_size=32, patch_size=4, num_classes=10, emb_size=512,
                 depth=6, num_heads=8, forward_expansion=4, dropout=0.1):
        super().__init__()
        num_patches = (image_size // patch_size) ** 2
        self.patch_embedding = PatchEmbedding(patch_size=patch_size, emb_size=emb_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, emb_size))
        self.dropout = nn.Dropout(dropout)
        
        self.transformer_encoder = nn.ModuleList([
            TransformerEncoderBlock(emb_size, num_heads, forward_expansion, dropout) for _ in range(depth)
        ])

        # This will be replaced by the BYOL projector, but we define it for completeness
        self.to_latent = nn.Identity()
        self.linear_head = nn.Linear(emb_size, num_classes)

    def forward(self, x):
        x = self.patch_embedding(x)
        b, n, _ = x.shape

        # Prepend the [CLS] token
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        # Pass through Transformer encoder blocks
        for block in self.transformer_encoder:
            x = block(x)

        # Extract the [CLS] token for classification
        cls_token_final = x[:, 0]
        
        # This is the output that will be fed to the BYOL projector
        return self.to_latent(cls_token_final)


# --- 2. Data Augmentations (Same as before) ---
class TwoCropTransform:
    def __init__(self, transform):
        self.transform = transform
    def __call__(self, x):
        return [self.transform(x), self.transform(x)]

def get_cifar10_dataloader(batch_size=256, num_workers=4, subset_percentage=1.0):
    transform = transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])
    full_train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=TwoCropTransform(transform))
    
    if subset_percentage < 1.0:
        num_samples = int(len(full_train_dataset) * subset_percentage)
        indices = np.random.choice(len(full_train_dataset), num_samples, replace=False)
        train_dataset = Subset(full_train_dataset, indices)
        print(f"Using a subset of {num_samples} images ({subset_percentage*100:.1f}% of training data).")
    else:
        train_dataset = full_train_dataset
        print(f"Using the full training dataset with {len(train_dataset)} images.")

    return torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)


# --- 3. BYOL Model (Adapted for ViT) ---
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(inplace=True), nn.Linear(hidden_dim, output_dim))
    def forward(self, x):
        return self.net(x)

class BYOL(nn.Module):
    def __init__(self, backbone, emb_size=512, projection_size=256, projection_hidden_size=4096, prediction_size=256, prediction_hidden_size=4096, momentum=0.99):
        super().__init__()
        self.momentum = momentum
        self.online_encoder = backbone
        self.online_projector = MLP(emb_size, projection_hidden_size, projection_size)
        self.online_predictor = MLP(projection_size, prediction_hidden_size, prediction_size)
        self.target_encoder = copy.deepcopy(self.online_encoder)
        self.target_projector = copy.deepcopy(self.online_projector)
        for param in self.target_encoder.parameters(): param.requires_grad = False
        for param in self.target_projector.parameters(): param.requires_grad = False

    @torch.no_grad()
    def _update_target_network(self):
        for param_q, param_k in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1. - self.momentum)
        for param_q, param_k in zip(self.online_projector.parameters(), self.target_projector.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1. - self.momentum)

    def forward(self, x1, x2):
        online_y1 = self.online_encoder(x1)
        online_z1 = self.online_projector(online_y1)
        online_p1 = self.online_predictor(online_z1)
        online_y2 = self.online_encoder(x2)
        online_z2 = self.online_projector(online_y2)
        online_p2 = self.online_predictor(online_z2)
        with torch.no_grad():
            self._update_target_network()
            target_y1 = self.target_encoder(x1)
            target_z1 = self.target_projector(target_y1)
            target_y2 = self.target_encoder(x2)
            target_z2 = self.target_projector(target_y2)
        return (online_p1, online_p2), (target_z1.detach(), target_z2.detach())

def byol_loss_fn(p, z):
    p = F.normalize(p, dim=-1)
    z = F.normalize(z, dim=-1)
    return 2 - 2 * (p * z).sum(dim=-1)


# --- 4. Training ---
def train_byol_vit(epochs=50, batch_size=256, subset_percentage=0.1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = get_cifar10_dataloader(batch_size, subset_percentage=subset_percentage)
    
    # Instantiate the Vision Transformer
    vit_backbone = VisionTransformer(
        image_size=32, patch_size=4, num_classes=10, emb_size=512,
        depth=6, num_heads=8, forward_expansion=4, dropout=0.1
    )
    
    model = BYOL(vit_backbone, emb_size=512).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    print("Starting BYOL-ViT training...")
    for epoch in range(epochs):
        total_loss = 0
        if len(train_loader) == 0:
            print(f"Skipping epoch {epoch+1} due to empty dataloader.")
            continue
        for i, (images, _) in enumerate(train_loader):
            images1, images2 = images[0].to(device), images[1].to(device)
            (p1, p2), (z1, z2) = model(images1, images2)
            loss = (byol_loss_fn(p1, z2) + byol_loss_fn(p2, z1)).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

    print("BYOL-ViT training finished.")
    return model.online_encoder # Return the trained encoder


# --- 5. Attention Map Visualization ---
def visualize_attention(model, device):
    print("\nVisualizing attention maps...")
    # Get a single image from CIFAR-10 to test
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])
    dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    img, _ = dataset[np.random.randint(0, len(dataset))] # Get a random image
    img_tensor = img.unsqueeze(0).to(device)

    # Put model in eval mode
    model.eval()
    
    # Forward pass to get attention weights
    with torch.no_grad():
        _ = model(img_tensor)

    # We want the attention from the last block
    # Note: This accesses the stored attention weights from the forward pass
    attention_map = model.transformer_encoder[-1].attention.attn_weights
    
    # Average the attention weights across all heads
    attention_map = attention_map.mean(dim=1).squeeze(0) # Shape: [num_patches+1, num_patches+1]
    
    # Get the attention for the [CLS] token
    cls_attention = attention_map[0, 1:].reshape(8, 8) # Reshape to image patch grid
    cls_attention = nn.functional.interpolate(
        cls_attention.unsqueeze(0).unsqueeze(0),
        scale_factor=4,
        mode="bilinear"
    ).squeeze()

    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    # Unnormalize and show original image
    img_display = img.permute(1, 2, 0).cpu().numpy()
    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.2023, 0.1994, 0.2010])
    img_display = std * img_display + mean
    img_display = np.clip(img_display, 0, 1)

    ax1.imshow(img_display)
    ax1.set_title("Original Image")
    ax1.axis('off')

    ax2.imshow(img_display)
    ax2.imshow(cls_attention.cpu().numpy(), cmap='jet', alpha=0.5)
    ax2.set_title("Attention Map Overlay")
    ax2.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # Set a seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Train the model
    trained_vit_encoder = train_byol_vit(epochs=1, subset_percentage=0.01) # Use 10% of data for 25 epochs for a quick demo

    # Visualize the attention
    visualize_attention(trained_vit_encoder, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))