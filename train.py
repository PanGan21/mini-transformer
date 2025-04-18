import torch
import torch.nn as nn
import torch.optim as optim

from src.transformer import Transformer

"""
Transformer Model Hyperparameter Guide
+----------------+------------------+------------------------------------------------+
| Hyperparameter | Typical values   | Impact on performance                          |
+----------------+------------------+------------------------------------------------+
| d_model        | 256, 512, 1024  | Higher values increase model capacity but      |
|                |                  | require more computation                       |
+----------------+------------------+------------------------------------------------+
| num_heads      | 8, 12, 16       | More heads can capture diverse aspects of      |
|                |                  | data, but are computationally intensive        |
+----------------+------------------+------------------------------------------------+
| num_layers     | 6, 12, 24       | More layers improve representation power,      |
|                |                  | but can lead to overfitting                    |
+----------------+------------------+------------------------------------------------+
| d_ff           | 2048, 4096      | Larger feed-forward networks increase model    |
|                |                  | robustness                                     |
+----------------+------------------+------------------------------------------------+
| dropout        | 0.1, 0.3        | Regularizes the model to prevent overfitting   |
+----------------+------------------+------------------------------------------------+
| learning rate  | 0.0001 - 0.001  | Impacts convergence speed and stability        |
+----------------+------------------+------------------------------------------------+
| batch size     | 32, 64, 128     | Larger batch sizes improve learning stability  |
|                |                  | but require more memory                        |
+----------------+------------------+------------------------------------------------+
"""

src_vocab_size = 5000
tgt_vocab_size = 5000
d_model = 512
num_heads = 8
num_layers = 6
d_ff = 2048
max_seq_length = 100
dropout = 0.1


transformer = Transformer(src_vocab_size, tgt_vocab_size, d_model,
                          num_heads, num_layers, d_ff, max_seq_length, dropout)


# Generate random sample data
# (batch_size, seq_length)
src_data = torch.randint(1, src_vocab_size, (64, max_seq_length))
# (batch_size, seq_length)
tgt_data = torch.randint(1, tgt_vocab_size, (64, max_seq_length))


criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(transformer.parameters(),
                       lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

transformer.train()

for epoch in range(100):
    optimizer.zero_grad()
    output = transformer(src_data, tgt_data[:, :-1])
    loss = criterion(output.contiguous().view(-1, tgt_vocab_size),
                     tgt_data[:, 1:].contiguous().view(-1))
    loss.backward()
    optimizer.step()
    print(f"Epoch: {epoch+1}, Loss: {loss.item()}")

criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(transformer.parameters(),
                       lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

transformer.train()

for epoch in range(100):
    optimizer.zero_grad()
    output = transformer(src_data, tgt_data[:, :-1])
    loss = criterion(output.contiguous().view(-1, tgt_vocab_size),
                     tgt_data[:, 1:].contiguous().view(-1))
    loss.backward()
    optimizer.step()
    print(f"Epoch: {epoch+1}, Loss: {loss.item()}")

transformer.eval()

# Generate random sample validation data
# (batch_size, seq_length)
val_src_data = torch.randint(1, src_vocab_size, (64, max_seq_length))
# (batch_size, seq_length)
val_tgt_data = torch.randint(1, tgt_vocab_size, (64, max_seq_length))

with torch.no_grad():

    val_output = transformer(val_src_data, val_tgt_data[:, :-1])
    val_loss = criterion(val_output.contiguous(
    ).view(-1, tgt_vocab_size), val_tgt_data[:, 1:].contiguous().view(-1))
    print(f"Validation Loss: {val_loss.item()}")
