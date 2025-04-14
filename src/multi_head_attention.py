import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import copy


# Mechanism to focus on different parts of the input
# Captures dependencies across different positions in the sequence
class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism that allows the model to jointly attend to information
    from different representation subspaces at different positions.

    This implementation splits the input into multiple heads, applies scaled dot-product
    attention independently on each head, and then concatenates and transforms the results.

    Args:
        d_model (int): The dimension of the model (input and output dimension)
        num_heads (int): Number of attention heads to use

    Attributes:
        d_model (int): The dimension of the model
        num_heads (int): Number of attention heads
        d_k (int): Dimension of each head's key, query, and value
        W_q (nn.Linear): Query transformation matrix
        W_k (nn.Linear): Key transformation matrix
        W_v (nn.Linear): Value transformation matrix
        W_o (nn.Linear): Output transformation matrix
    """

    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        # Ensure that the model dimension (d_model) is divisible by the number of heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        # Initialize dimensions
        self.d_model = d_model  # Model's dimension
        self.num_heads = num_heads  # Number of attention heads
        self.d_k = d_model // num_heads  # Dimension of each head's key, query, value

        # Linear layers for transforming inputs
        self.W_q = nn.Linear(d_model, d_model)  # Query transformation
        self.W_k = nn.Linear(d_model, d_model)  # Key transformation
        self.W_v = nn.Linear(d_model, d_model)  # Value transformation
        self.W_o = nn.Linear(d_model, d_model)  # Output transformation

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """
        Applies scaled dot-product attention mechanism on the input tensors.

        The attention is computed as: softmax(QK^T/sqrt(d_k))V

        Args:
            Q (torch.Tensor): Query tensor of shape (batch_size, num_heads, seq_length, d_k)
            K (torch.Tensor): Key tensor of shape (batch_size, num_heads, seq_length, d_k)
            V (torch.Tensor): Value tensor of shape (batch_size, num_heads, seq_length, d_k)
            mask (torch.Tensor, optional): Mask tensor to prevent attention to certain positions.
                                         Shape: (batch_size, 1,
                                                 seq_length, seq_length)

        Returns:
            torch.Tensor: Output tensor after applying attention.
                         Shape: (batch_size, num_heads, seq_length, d_k)
        """

        # Calculate attention scores
        attention_scores = torch.matmul(
            Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Apply mask if provided (useful for preventing attention to certain parts like padding)
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)

        # Softmax is applied to obtain attention probabilities
        attention_probabilitites = torch.softmax(attention_scores, dim=-1)

        # Multiply by values to obtain the final output
        output = torch.matmul(attention_probabilitites, V)
        return output

    def split_heads(self, x):
        """
        Splits the input tensor into multiple heads by reshaping.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, d_model)

        Returns:
            torch.Tensor: Reshaped tensor of shape (batch_size, num_heads, seq_length, d_k)
        """

        # Reshape the input to have num_heads for multi-head attention
        batch_size, seq_length, _ = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        """
        Combines the multiple heads back into original shape.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_heads, seq_length, d_k)

        Returns:
            torch.Tensor: Combined tensor of shape (batch_size, seq_length, d_model)
        """

        # Combine the multiple heads back to original shape
        batch_size, _, seq_legth, _ = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_legth, self.d_model)

    def forward(self, Q, K, V, mask=None):
        """
        Forward pass of the multi-head attention mechanism.

        This method:
        1. Linearly projects the queries, keys and values
        2. Splits them into multiple heads
        3. Applies scaled dot-product attention
        4. Combines the heads
        5. Applies final linear transformation

        Args:
            Q (torch.Tensor): Query tensor of shape (batch_size, seq_length, d_model)
            K (torch.Tensor): Key tensor of shape (batch_size, seq_length, d_model)
            V (torch.Tensor): Value tensor of shape (batch_size, seq_length, d_model)
            mask (torch.Tensor, optional): Mask tensor to prevent attention to certain positions

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_length, d_model)
        """

        # Apply linear transformations and split heads
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))

        # Performa scaled dot-product attention
        attention_output = self.scaled_dot_product_attention(Q, K, V, mask)

        # Combine heads and apply output transformation
        output = self.W_o(self.combine_heads(attention_output))
        return output
