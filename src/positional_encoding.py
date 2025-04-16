import torch
import torch.nn as nn
import math


# Positional Encoding is used to inject the position information of each token in the input sequence.
# It uses sine and cosine functions of different frequencies to generate the positional encoding.
class PositionalEncoding(nn.Module):
    """
    Positional Encoding layer that adds position information to input embeddings.

    This implementation uses sine and cosine functions of different frequencies to generate
    unique position encodings for each position in the sequence. The encoding follows the formula:

    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    where pos is the position and i is the dimension.

    Args:
        d_model (int): The dimension of the model/embeddings
        max_seq_length (int): Maximum sequence length to generate positional encodings for

    Attributes:
        pe (torch.Tensor): Positional encoding matrix of shape (1, max_seq_length, d_model)
                          stored as a buffer (not a parameter)

    Note:
        The positional encodings are fixed and not learned during training.
        They are added to the input embeddings during the forward pass.
    """

    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(
            0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        """
        Add positional encodings to the input embeddings.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, d_model)
                             containing the token embeddings

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_length, d_model)
                         with positional encodings added to the input embeddings

        Note:
            The positional encodings are truncated or expanded to match the input sequence length.
            Only the first seq_length positions from the pre-computed encodings are used.
        """
        return x + self.pe[:, :x.size(1)]
