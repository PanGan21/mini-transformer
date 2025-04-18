import torch
import torch.nn as nn

from positional_encoding import PositionalEncoding
from encoder_layer import EncoderLayer
from decoder_layer import DecoderLayer


# The Transformer class brings together the various components of a Transformer model,
# including the embeddings, positional encoding, encoder layers, and decoder layers.
# It provides a convenient interface for training and inference, encapsulating the complexities of multi-head attention,
# feed-forward networks, and layer normalization.
class Transformer(nn.Module):
    """
    Complete Transformer architecture implementing the 'Attention is All You Need' paper.

    The Transformer model consists of an encoder and decoder stack, each with their own
    embedding layers, positional encodings, and multiple layers of attention and feed-forward networks.

    Args:
        src_vocab_size (int): Size of the source vocabulary
        tgt_vocab_size (int): Size of the target vocabulary
        d_model (int): Dimension of the model's embeddings and layers
        num_heads (int): Number of attention heads in multi-head attention layers
        num_layers (int): Number of encoder and decoder layers in their respective stacks
        d_ff (int): Dimension of the feed-forward network in encoder/decoder layers
        max_seq_length (int): Maximum sequence length for positional encoding
        dropout (float): Dropout rate for regularization

    Attributes:
        encoder_embedding (nn.Embedding): Embedding layer for source sequences
        decoder_embedding (nn.Embedding): Embedding layer for target sequences
        positional_encoding (PositionalEncoding): Adds positional information to embeddings
        encoder_layers (nn.ModuleList): List of encoder layers
        decoder_layers (nn.ModuleList): List of decoder layers
        fc (nn.Linear): Final linear layer to project to vocabulary size
        dropout (nn.Dropout): Dropout layer for regularization
    """

    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout):
        super(Transformer, self).__init__()

        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

        self.fc = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, src, tgt):
        """
        Generate padding and sequence masks for source and target sequences.

        Creates two types of masks:
        1. Padding mask: To ignore padding tokens in source/target sequences
        2. Sequence mask: To prevent the decoder from attending to future tokens

        Args:
            src (torch.Tensor): Source sequence tensor of shape (batch_size, src_seq_length)
            tgt (torch.Tensor): Target sequence tensor of shape (batch_size, tgt_seq_length)

        Returns:
            tuple: A tuple containing:
                - src_mask (torch.Tensor): Source padding mask of shape 
                                         (batch_size, 1, 1, src_seq_length)
                - tgt_mask (torch.Tensor): Combined target padding and sequence mask of shape
                                         (batch_size, 1, tgt_seq_length, tgt_seq_length)
        """
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
        seq_length = tgt.size(1)
        nopeak_mask = (
            1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()
        tgt_mask = tgt_mask & nopeak_mask
        return src_mask, tgt_mask

    def forward(self, src, tgt):
        """
        Forward pass of the Transformer model.

        The process follows these steps:
        1. Generate masks for source and target sequences
        2. Convert input tokens to embeddings and add positional encoding
        3. Pass through encoder stack
        4. Pass through decoder stack, using encoder outputs
        5. Project final decoder output to vocabulary size

        Args:
            src (torch.Tensor): Source sequence tensor of shape 
                               (batch_size, src_seq_length)
            tgt (torch.Tensor): Target sequence tensor of shape
                               (batch_size, tgt_seq_length)

        Returns:
            torch.Tensor: Output logits of shape 
                         (batch_size, tgt_seq_length, tgt_vocab_size)

        Note:
            - Source and target sequences should be padded with 0s if necessary
            - The model processes the entire sequence in parallel during training
            - During inference, the target sequence is typically processed 
              one token at a time
        """
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        src_embedded = self.dropout(
            self.positional_encoding(self.encoder_embedding(src)))
        tgt_embedded = self.dropout(
            self.positional_encoding(self.decoder_embedding(tgt)))

        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)

        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, tgt_mask)

        output = self.fc(dec_output)
        return output
