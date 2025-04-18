# mini-transformer Implementation

A PyTorch implementation of the Transformer architecture based on the "Attention Is All You Need" paper. This project implements a scaled-down version of the Transformer model, suitable for learning and experimentation.

## Architecture

The implementation includes the following key components:

- Multi-Head Attention
- Position-wise Feed Forward Networks
- Positional Encoding
- Encoder and Decoder Layers
- Full Transformer Model

## Model Configuration

Default hyperparameters used in the implementation:
| Hyperparameter | Value | Description |
|----------------|-------|-------------|
| src_vocab_size | 5000 | Source vocabulary size |
| tgt_vocab_size | 5000 | Target vocabulary size |
| d_model | 512 | Model dimension |
| num_heads | 8 | Number of attention heads |
| num_layers | 6 | Number of encoder/decoder layers |
| d_ff | 2048 | Feed-forward network dimension |
| max_seq_length | 100 | Maximum sequence length |
| dropout | 0.1 | Dropout rate |

## Training Results

The model was trained for 200 epochs in total (2 runs of 100 epochs each). Training results show:

- Initial loss: ~8.68
- Final training loss: 0.391
- Validation loss: 9.253

The decreasing loss pattern indicates that the model successfully learned to:

1. First run (epochs 1-100): Loss decreased from 8.68 to 2.71
2. Second run (epochs 1-100): Loss further decreased from 2.68 to 0.39

## Project Structure

mini-transformer/
├── src/
│ ├── multi_head_attention.py
│ ├── position_wise_feed_forward.py
│ ├── positional_encoding.py
│ ├── encoder_layer.py
│ ├── decoder_layer.py
│ └── transformer.py
├── train.py
├── requirements.txt
└── README.md

## Requirements

- PyTorch
- Python 3.x
- Additional dependencies in `requirements.txt`

## Setup locally

- Create a virtual environment: `python -m venv .venv`
- Activate the virtual environemt: `source .venv/bin/activate`
- Install the required dependencies: `pip install -r requirements.txt`

## Usage

To train the model:

```bash
python train.py
```

## References

- "Attention Is All You Need" paper

## License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.
