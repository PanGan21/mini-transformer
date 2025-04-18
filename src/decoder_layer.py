import torch.nn as nn

from multi_head_attention import MultiHeadAttention
from position_wise_feed_forward import PositionWiseFeedForward


# The EncoderLayer consists off a multi-head self-attention mechanism, a multi-head cross-attention mechanism
# (that attends to the encoder's output), a position-wise feed-forward neural network, and the corresponding residual connections,
# layer normalization, and dropout layers.
class DecoderLayer(nn.Module):
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
        attention_output = self.self_attention(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attention_output))

        attention_output = self.cross_attention(
            x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attention_output))

        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))

        return x
