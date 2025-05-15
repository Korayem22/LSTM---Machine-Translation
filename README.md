# BiLSTM Sequence-to-Sequence Machine Translation with Attention

This project implements an English-to-Italian machine translation model using a bidirectional LSTM encoder, attention mechanism, and teacher forcing in PyTorch. It includes data preprocessing, vocabulary building, custom dataset class, training loop, evaluation via BLEU score, and translation inference.

---

##  Features

-  Bidirectional LSTM Encoder
-  Attention-based Decoder
-  Token-level vocabulary with rare-word filtering and UNK handling
-  Evaluation using BLEU Score
-  Inference with greedy decoding
-  Preprocessing pipeline: cleaning, tokenization, length filtering

---

##  Dataset

- Source: [OPUS-100](https://huggingface.co/datasets/opus100) (`en-it` split)
- Used Hugging Face `datasets` library for loading
- Training limited to N samples (default: 100k)
- Sentences longer than 10 tokens are excluded

---

##  Dependencies

- `torch`
- `datasets`
- `nltk`
- `tqdm`

