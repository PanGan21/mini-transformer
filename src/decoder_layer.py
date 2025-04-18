import torch.nn as nn

from src.multi_head_attention import MultiHeadAttention
from src.position_wise_feed_forward import PositionWiseFeedForward


# The EncoderLayer consists off a multi-head self-attention mechanism, a multi-head cross-attention mechanism
# (that attends to the encoder's output), a position-wise feed-forward neural network, and the corresponding residual connections,
# layer normalization, and dropout layers.
class DecoderLayer(nn.Module):
    """
    A single layer of the Transformer decoder.

    Each decoder layer consists of three main sub-layers:
    1. Masked Multi-Head Self-Attention mechanism
    2. Multi-Head Cross-Attention mechanism (attending to encoder output)
    3. Position-wise Feed-Forward Network
    Each sub-layer is followed by a residual connection and layer normalization.

    Args:
        d_model (int): The dimension of the model (input and output dimension)
        num_heads (int): Number of attention heads in both attention layers
        d_ff (int): Hidden dimension of the feed-forward network
        dropout (float): Dropout rate for regularization

    Attributes:
        self_attention (MultiHeadAttention): Masked multi-head self-attention mechanism
        cross_attention (MultiHeadAttention): Multi-head cross-attention mechanism for
                                            attending to encoder outputs
        feed_forward (PositionWiseFeedForward): Position-wise feed-forward network
        norm1 (nn.LayerNorm): Layer normalization after self-attention
        norm2 (nn.LayerNorm): Layer normalization after cross-attention
        norm3 (nn.LayerNorm): Layer normalization after feed-forward
        dropout (nn.Dropout): Dropout layer for regularization
    """

    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()

        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.cross_attention = MultiHeadAttention(d_model, num_heads)

        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask, tgt_mask):
        """
        Forward pass of the decoder layer.

        The forward pass follows this sequence:
        1. Masked self-attention on decoder input (prevents attending to future tokens)
        2. Residual connection and layer normalization
        3. Cross-attention between decoder state and encoder output
        4. Another residual connection and layer normalization
        5. Feed-forward network
        6. Final residual connection and layer normalization

        Args:
            x (torch.Tensor): Input tensor from previous decoder layer or embedding
                             shape: (batch_size, target_seq_length, d_model)
            enc_output (torch.Tensor): Output from the encoder
                                     shape: (batch_size, source_seq_length, d_model)
            src_mask (torch.Tensor): Mask for encoder outputs to ignore padding
                                   shape: (batch_size, 1, 1, source_seq_length)
            tgt_mask (torch.Tensor): Mask for decoder self-attention to prevent attending
                                   to future tokens and padding
                                   shape: (batch_size, 1, target_seq_length, target_seq_length)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, target_seq_length, d_model)

        Note:
            - The self-attention uses masking to prevent attending to future positions
            - The cross-attention allows the decoder to attend to relevant parts of the input sequence
            - Residual connections help with training deep networks
            - Layer normalization helps stabilize training
        """
        attention_output = self.self_attention(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attention_output))

        attention_output = self.cross_attention(
            x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attention_output))

        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))

        return x
