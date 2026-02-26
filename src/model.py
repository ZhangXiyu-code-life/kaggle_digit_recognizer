import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """
    A simple CNN for MNIST digit classification.

    Input:
        Tensor with shape [batch_size, 1, 28, 28]
    Output:
        Logits with shape [batch_size, 10]
    """

    def __init__(self) -> None:
        super().__init__()

        # First convolution layer:
        # - in_channels=1 because MNIST images are grayscale.
        # - out_channels=32 to extract 32 feature maps.
        # - kernel_size=3 and padding=1 keep spatial size at 28x28.
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)

        # Second convolution layer:
        # - takes 32 feature maps and outputs 64 richer feature maps.
        # - same kernel/padding setup keeps spatial size before pooling.
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)

        # Shared max-pooling layer:
        # - kernel_size=2, stride=2 halves height/width each time.
        # - after two pooling ops: 28x28 -> 14x14 -> 7x7.
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Final classifier:
        # - flattened feature size is 64 * 7 * 7.
        # - output size is 10 for classes [0..9].
        self.fc = nn.Linear(in_features=64 * 7 * 7, out_features=10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input x shape: [N, 1, 28, 28]

        # Conv block 1: convolution -> ReLU -> max pooling
        # Shape: [N, 1, 28, 28] -> [N, 32, 28, 28] -> [N, 32, 14, 14]
        x = F.relu(self.conv1(x))
        x = self.pool(x)

        # Conv block 2: convolution -> ReLU -> max pooling
        # Shape: [N, 32, 14, 14] -> [N, 64, 14, 14] -> [N, 64, 7, 7]
        x = F.relu(self.conv2(x))
        x = self.pool(x)

        # Flatten from [N, 64, 7, 7] to [N, 64*7*7].
        # start_dim=1 keeps batch dimension N unchanged.
        x = torch.flatten(x, start_dim=1)

        # Output logits for 10 classes.
        # No softmax here because nn.CrossEntropyLoss expects raw logits.
        x = self.fc(x)
        return x
