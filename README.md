# English-to-Arabic Machine Translation using Seq2Seq LSTM (PyTorch)

This project implements a sequence-to-sequence (Seq2Seq) neural network for English-to-Arabic translation using PyTorch. The model is built without attention and uses an encoder-decoder architecture with LSTM layers.

## Model Overview

- **Encoder**: Bidirectional LSTM
- **Decoder**: Unidirectional LSTM
- **Embedding Layers**: Learnable source and target embeddings
- **Loss Function**: Label smoothed cross-entropy (optional)
- **Training Strategy**:
  - Teacher Forcing with optional decaying ratio
  - Padding handling via `ignore_index`
  - Early stopping based on validation loss (manual)

##  Features

- Custom vocabulary and tokenization
- Training/validation/test split
- Label smoothing for regularization
- Decaying teacher forcing ratio to reduce overfitting
- Greedy decoding during inference
- BLEU score evaluation on test set

## 🗂 Dataset

The dataset is based on the [Tatoeba English-Arabic parallel corpus](https://opus.nlpl.eu/Tatoeba.php), preprocessed to remove repeated samples and heavily filtered for unknown tokens.

