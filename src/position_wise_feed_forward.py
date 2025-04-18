import torch.nn as nn


# Position-wise fully connected layers
# Transforms the attention outputs, adding complexity
class PositionWiseFeedForward(nn.Module):
    """
    Position-wise Feed Forward Network (FFN) layer used in Transformer architecture.

    This implements the FFN equation: FFN(x) = max(0, xW₁ + b₁)W₂ + b₂
    It consists of two linear transformations with a ReLU activation in between.

    Args:
        d_model (int): The input and output dimension (model dimension)
        d_ff (int): The hidden dimension of the feed-forward layer, typically 4*d_model

    Attributes:
        fc1 (nn.Linear): First linear transformation (d_model → d_ff)
        fc2 (nn.Linear): Second linear transformation (d_ff → d_model)
        relu (nn.ReLU): ReLU activation function
    """

    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Forward pass of the feed-forward network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, d_model)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_length, d_model)
        """
        return self.fc2(self.relu(self.fc1(x)))
