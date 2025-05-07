# CS598

## 🧪 Experimental Setup

### 📁 Dataset and Splits
We use the MIMICSQL* and MIMIC-SPARQL* datasets derived from the MIMIC-III database. These datasets support structured QA in both SQL and SPARQL formats.

- **Training set**: 8,000 QA pairs  
- **Validation set**: 1,000 QA pairs  
- **Test set**: 1,000 QA pairs

### 🛠️ Preprocessing
- Tokenization includes special symbols: `<pad>`, `<sos>`, `<eos>`
- Lowercasing and punctuation normalization are applied
- Structural accuracy metrics remove literals (e.g., numbers, strings)
- SPARQL queries are automatically converted from SQL using schema-aware path finding
- All SQL-SPARQL pairs are validated for semantic equivalence

### ⚙️ Implementation Details
- Framework: PyTorch  
- Optimizer: Adam with gradient clipping  
- Evaluation: RDFlib (SPARQL), SQLite (SQL)  
- Training device: NVIDIA TITAN Xp GPU  
- Average training time: ~1 hour  
- Teacher forcing with scheduled decay  
- All code and data will be open-sourced after review
