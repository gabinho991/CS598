# MIMIC-SPARQL
This repository provides our replicate project of mimic-sparql dataset implementation of the following paper: [Knowledge Graph-based Question Answering with Electronic Health Records](https://arxiv.org/abs/2010.09394) accepted at Machine Learning in Health Care (MLHC) 2021.

## Datasets

3. __MIMICSQL__  
MIMICSQL consists of 5 merged table of MIMIC-III. Dataset and codes can be found in https://github.com/wangpinggl/TREQS
4. __MIMIC-SPARQL__  
MIMIC-SPARQL is a graph-based counterpart of MIMICSQL.
3. __MIMICSQL*__  
MIMICSQL* is extended version of MIMICSQL. The database consists of 9 table of MIMIC-III instead of 5 merged tables.  
4. __MIMIC-SPARQL*__  
MIMIC-SPARQL is a graph-based counterpart of MIMICSQL*. 

### Prepare the Datasets

First, you need to access the MIMIC-III data. This requires certification from https://mimic.physionet.org/ 
And then, `mimic.db` is necessary to create MIMICSQL following the https://github.com/wangpinggl/TREQS README.md 
We also provide script to create `mimic.db` through Bigquery, which require your google cloud project with access the MIMIC-III.
The following process to create MIMICSQL* & MIMIC-SPARQL* can be find in https://github.com/junwoopark92/mimic-sparql/blob/master/README.md

## Seq2Seq Model for MIMIC-SPARQL & MIMIC* Data

This project trains a sequence-to-sequence (Seq2Seq) model with Luong attention on MIMIC-III data. The model is designed for tasks such as question answering or sequence prediction in the clinical domain.

### 🔧 Hyperparameters

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

