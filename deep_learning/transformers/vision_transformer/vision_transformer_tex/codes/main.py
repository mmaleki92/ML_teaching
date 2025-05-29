import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import math
import time

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Simple Patch Embedding layer
class PatchEmbedding(nn.Module):
    def __init__(self, image_size=8, patch_size=2, in_channels=1, embed_dim=64):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        
        self.projection = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
        # x shape: [batch_size, channels, image_size, image_size]
        x = self.projection(x)  # [batch_size, embed_dim, image_size//patch_size, image_size//patch_size]
        x = x.flatten(2)  # [batch_size, embed_dim, num_patches]
        x = x.transpose(1, 2)  # [batch_size, num_patches, embed_dim]
        return x

# Multi-headed Self Attention with hooks for visualization
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attention_maps = None
        
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch_size, num_heads, seq_len, head_dim]
        
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        
        # Save attention probabilities for visualization
        self.attention_maps = attn_probs.detach()
        
        # Apply attention
        context = torch.matmul(attn_probs, v)
        context = context.permute(0, 2, 1, 3)  # [batch_size, seq_len, num_heads, head_dim]
        context = context.reshape(batch_size, seq_len, self.embed_dim)
        
        output = self.proj(context)
        return output

"""
import numpy as np

def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)  # for numerical stability
    e_x = np.exp(x)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def multi_head_self_attention(X, num_heads):
    '''
    X: Input tensor of shape (B, N, D)
       B = batch size
       N = sequence length (e.g., num of patches + 1 CLS token)
       D = embedding dimension
    num_heads: number of attention heads (H)
    '''

    B, N, D = X.shape
    assert D % num_heads == 0, "Embedding dimension must be divisible by number of heads."
    d_head = D // num_heads

    # 1. Initialize weights (random for demonstration)
    W_q = np.random.randn(D, D)
    W_k = np.random.randn(D, D)
    W_v = np.random.randn(D, D)
    W_o = np.random.randn(D, D)

    # 2. Linear projections: (B, N, D) → (B, N, D)
    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v

    # 3. Reshape for multi-heads: (B, N, H, d_head) → (B, H, N, d_head)
    def reshape_for_heads(x):
        return x.reshape(B, N, num_heads, d_head).transpose(0, 2, 1, 3)

    Q = reshape_for_heads(Q)
    K = reshape_for_heads(K)
    V = reshape_for_heads(V)

    # 4. Scaled dot-product attention
    scale = np.sqrt(d_head)
    attention_scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / scale   # (B, H, N, N)
    attention_weights = softmax(attention_scores, axis=-1)
    attention_output = np.matmul(attention_weights, V)                 # (B, H, N, d_head)

    # 5. Concatenate heads: (B, H, N, d_head) → (B, N, D)
    attention_output = attention_output.transpose(0, 2, 1, 3).reshape(B, N, D)

    # 6. Final output projection: (B, N, D) → (B, N, D)
    output = attention_output @ W_o

    return output, attention_weights  # return attention weights for inspection

"""

# Transformer Encoder Block
class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        # Attention Block with residual connection
        attn_output = self.attention(self.norm1(x))
        
        # residual connection (a.k.a. skip connection).
        # Instead of directly replacing x, it adds the attention output to the original input.
        x = x + attn_output
        
        # MLP Block with residual connection
        mlp_output = self.mlp(self.norm2(x))
        x = x + mlp_output
        
        return x

# Vision Transformer
class VisionTransformer(nn.Module):
    def __init__(
        self,
        image_size=8,
        patch_size=2,
        in_channels=1,
        num_classes=10,
        embed_dim=64,
        depth=3,
        num_heads=4,
        mlp_dim=128,
        dropout=0.1
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim
        )
        
        num_patches = self.patch_embed.num_patches
        
        # Add class token
        self.class_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Positional embedding, learnable
        self.pos_embedding = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        
        self.dropout = nn.Dropout(dropout)
        
        # Transformer encoder blocks, stack of all the transformers
        self.transformer_blocks = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads, mlp_dim, dropout)
            for _ in range(depth)
        ])
        
        # Classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        nn.init.normal_(self.pos_embedding, std=0.02)
        nn.init.normal_(self.class_token, std=0.02)
    
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Create patch embeddings
        x = self.patch_embed(x)  # [batch_size, num_patches, embed_dim]
        
        # Add class token, a learned tensor, It acts as a placeholder that will attend to all patches,
        #  and be used for classification at the end.

        class_tokens = self.class_token.expand(batch_size, -1, -1)
        x = torch.cat((class_tokens, x), dim=1)  # [batch_size, 1+num_patches, embed_dim]

        # good to know:
        # expand: creates a view, doesn't use extra memory.
        # repeat: actually copies data, and is memory-inefficient for this case.


        # Add positional embeddings
        x = x + self.pos_embedding
        x = self.dropout(x)
        
        # Pass through transformer blocks
        for block in self.transformer_blocks:
            x = block(x)
        
        # Classification token
        x = self.norm(x)
        x = x[:, 0]  # Take only the class token for classification
        x = self.head(x)
        
        return x
    
    def get_attention_maps(self):
        attention_maps = []
        for block in self.transformer_blocks:
            attention_maps.append(block.attention.attention_maps)
        return attention_maps

"""
| Step                         | Shape (example)                          | Explanation                                                     |
| ---------------------------- | ---------------------------------------- | --------------------------------------------------------------- |
| Input `x`                    | `[batch_size, C, H, W]`                  | Batch of images                                                 |
| Patch embeddings `x`         | `[batch_size, num_patches, embed_dim]`   | Each image → patches                                            |
| Class token (before expand)  | `[1, 1, embed_dim]`                      | One learned token                                               |
| Class tokens (after expand)  | `[batch_size, 1, embed_dim]`             | Same class token for all images in batch (broadcasted)          |
| Concatenated `x`             | `[batch_size, 1+num_patches, embed_dim]` | Class token + patches, per image sequence                       |
| After transformer blocks     | `[batch_size, 1+num_patches, embed_dim]` | Class token now different per image (due to attention)          |
| Extract class token for head | `[batch_size, embed_dim]`                | Take first token’s embedding from each image for classification |

"""
# Load scikit-learn digits dataset
def load_sklearn_digits(batch_size=128):
    # Load the digits dataset
    digits = load_digits()
    X = digits.data
    y = digits.target
    
    # Reshape data to [samples, channels, height, width]
    X = X.reshape(-1, 1, 8, 8)
    
    # Normalize the data (scale to [0, 1])
    X = X / 16.0
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    
    # Convert to torch tensors
    X_train = torch.FloatTensor(X_train)
    y_train = torch.LongTensor(y_train)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.LongTensor(y_test)
    
    # Create TensorDatasets
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, digits

# Training function
def train(model, train_loader, optimizer, criterion, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    print(f'Epoch [{epoch+1}], Loss: {running_loss/len(train_loader):.4f}, Acc: {100.*correct/total:.2f}%')
    return 100. * correct / total

# Evaluation function
def evaluate(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    accuracy = 100. * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')
    return accuracy

# Visualize attention maps
def visualize_attention(model, test_loader, digits_dataset, num_images=5):
    # Get some test images
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    images = images[:num_images].to(device)
    labels = labels[:num_images]
    
    # Get model predictions and attention maps
    with torch.no_grad():
        _ = model(images)
        all_attention_maps = model.get_attention_maps()
    
    # Original images for reference
    images_np = images.cpu().numpy()
    
    # Create a figure for displaying images and attention maps
    fig, axes = plt.subplots(num_images, model.transformer_blocks[0].attention.num_heads + 1, 
                            figsize=(3 * (model.transformer_blocks[0].attention.num_heads + 1), 3 * num_images))
    
    for img_idx in range(num_images):
        # Display original image
        axes[img_idx, 0].imshow(images_np[img_idx, 0], cmap='gray')
        axes[img_idx, 0].set_title(f"Digit: {labels[img_idx].item()}")
        axes[img_idx, 0].axis('off')
        
        # Get attention maps from final transformer block
        # We focus on the attention paid to each patch by the class token
        attn_map = all_attention_maps[-1][img_idx]  # [num_heads, seq_len, seq_len]
        
        for head_idx in range(model.transformer_blocks[0].attention.num_heads):
            # Extract attention from class token to patches
            # Skip the class token itself (index 0)
            attention_from_class = attn_map[head_idx, 0, 1:].reshape(4, 4)  # 8/2 = 4
            
            # Resize attention map to match image dimensions
            attention_resized = torch.nn.functional.interpolate(
                attention_from_class.unsqueeze(0).unsqueeze(0),
                size=(8, 8),
                mode='bilinear',
                align_corners=False
            ).squeeze().cpu().numpy()
            
            # Create heatmap overlay
            ax = axes[img_idx, head_idx + 1]
            ax.imshow(images_np[img_idx, 0], cmap='gray')
            ax.imshow(attention_resized, alpha=0.5, cmap='hot')
            ax.set_title(f"Head {head_idx+1}")
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('sklearn_digits_attention_visualization.png')
    plt.show()

def main():
    # Hyperparameters
    batch_size = 32  # Using a smaller batch size for this smaller dataset
    num_epochs = 10
    learning_rate = 3e-4
    
    # Load data
    train_loader, test_loader, digits_dataset = load_sklearn_digits(batch_size)
    
    # Initialize model - adapted for 8x8 images with 2x2 patches
    model = VisionTransformer(
        image_size=8,
        patch_size=2,  # 2x2 patches
        in_channels=1,  # Grayscale images
        num_classes=10,
        embed_dim=64,
        depth=3,
        num_heads=4,
        mlp_dim=128
    ).to(device)
    
    print(f"Created ViT with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Train the model
    start_time = time.time()
    for epoch in range(num_epochs):
        train_acc = train(model, train_loader, optimizer, criterion, epoch)
    
    print(f"Training took {(time.time() - start_time):.2f} seconds")
    
    # Evaluate the model
    evaluate(model, test_loader)
    
    # Visualize attention maps
    visualize_attention(model, test_loader, digits_dataset, num_images=5)
    
    # Save the model
    torch.save(model.state_dict(), 'vit_sklearn_digits.pth')
    print("Model saved as 'vit_sklearn_digits.pth'")

if __name__ == '__main__':
    main()