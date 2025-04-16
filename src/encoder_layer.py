import torch.nn as nn

from multi_head_attention import MultiHeadAttention
from position_wise_feed_forward import PositionWiseFeedForward


# The EncoderLayer encapsulates a multi-head self-attention mechanism followed by the position-wise feed-forward neural network,
# with residual connections, layer normalization, and dropout applied as appropriate.
class EncoderLayer(nn.Module):
    """
    A single layer of the Transformer encoder.

    Each encoder layer consists of two main sub-layers:
    1. Multi-Head Self-Attention mechanism
    2. Position-wise Feed-Forward Network
    Each sub-layer is followed by a residual connection and layer normalization.

    Args:
        d_model (int): The dimension of the model (input and output dimension)
        num_heads (int): Number of attention heads in the multi-head attention layer
        d_ff (int): Hidden dimension of the feed-forward network
        dropout (float): Dropout rate for regularization

    Attributes:
        self_attention (MultiHeadAttention): Multi-head self-attention mechanism
        feed_forward (PositionWiseFeedForward): Position-wise feed-forward network
        norm1 (nn.LayerNorm): Layer normalization after attention
        norm2 (nn.LayerNorm): Layer normalization after feed-forward
        dropout (nn.Dropout): Dropout layer for regularization
    """

    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()

        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        """
        Forward pass of the encoder layer.

        The forward pass follows this sequence:
        1. Self-attention on the input
        2. Residual connection and layer normalization
        3. Feed-forward network
        4. Another residual connection and layer normalization

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, d_model)
            mask (torch.Tensor): Attention mask to prevent attention to padding tokens,
                               shape (batch_size, 1, seq_length, seq_length)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_length, d_model)

        Note:
            The residual connections (x + ...) help with training deep networks
            Layer normalization helps stabilize the network
        """
        attention_output = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attention_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x
