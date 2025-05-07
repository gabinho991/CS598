
# Seq2Seq Model for MIMIC Data

This project trains a sequence-to-sequence (Seq2Seq) model with Luong attention on MIMIC-III data. The model is designed for tasks such as question answering or sequence prediction in the clinical domain.

## 🔧 Hyperparameters

| Hyperparameter        | Value                    | Notes                                      |
|-----------------------|--------------------------|--------------------------------------------|
| Embedding size        | `256`                    | Used in both encoder and decoder           |
| Hidden size           | `256`                    | BiLSTM encoder splits this across 2 dirs   |
| Encoder dropout       | `0.3`                    | Applies to encoder LSTM                    |
| Decoder dropout       | `0.3`                    | Applies to decoder LSTM                    |
| Attention type        | `Luong`                  | Used in decoder                            |
| Max decoding length   | `150`                    | During inference                           |
| Learning rate         | `1e-3`                   | Adam optimizer                             |
| Batch size            | `32` (default)           | Can be overridden via `--bs`               |
| Epochs                | `20` (default)           | Can be overridden via `--epochs`           |
| LR scheduler step     | `2`                      | Every 2 epochs                             |
| LR scheduler gamma    | `0.8`                    | Multiplicative decay                       |
| Gradient clipping     | `1.0`                    | Applied during training                    |
| Scheduled sampling    | `0.9 * (0.95^(epoch-1))` | Lowered each epoch, min capped at 0.1      |

## 🚀 Training

Use the `train.py` script to train the model:

```bash
python train.py --train path/to/train.jsonl --dev path/to/dev.jsonl --epochs 20 --bs 32
```

Model and vocabulary will be saved automatically in the same directory as the training data.

