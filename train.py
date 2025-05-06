import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from utils import load_data
from vocab import Vocab
from model.encoder import Encoder
from model.decoder import Decoder
from model.attention import Attention
from model.seq2seq import Seq2Seq

# ====== Configuration ======
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
EMB_DIM = 128
HID_DIM = 256
EPOCHS = 15
LR = 1e-3

# ====== Load data & vocab ======
train_data = load_data("data/train.json")
q_vocab = Vocab.load("data/question_vocab.json")
s_vocab = Vocab.load("data/query_vocab.json")

# ====== Dataset prep ======
def prepare_pair(item):
    q_ids = torch.tensor(q_vocab.encode(item["question"]), dtype=torch.long)
    s_ids = torch.tensor(s_vocab.encode(["<sos>"] + item["query"] + ["<eos>"]), dtype=torch.long)
    return q_ids, s_ids

pairs = [prepare_pair(d) for d in train_data]

def collate_fn(batch):
    src, trg = zip(*batch)
    src_pad = pad_sequence(src, padding_value=0)
    trg_pad = pad_sequence(trg, padding_value=0)
    return src_pad.to(DEVICE), trg_pad.to(DEVICE)

train_loader = DataLoader(pairs, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

# ====== Model setup ======
INPUT_DIM = len(q_vocab)
OUTPUT_DIM = len(s_vocab)

attention = Attention(HID_DIM)
encoder = Encoder(INPUT_DIM, EMB_DIM, HID_DIM)
decoder = Decoder(OUTPUT_DIM, EMB_DIM, HID_DIM, attention)

model = Seq2Seq(encoder, decoder).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss(ignore_index=0)

# ====== Training Loop ======
for epoch in range(1, EPOCHS + 1):
    model.train()
    epoch_loss = 0
    for src, trg in train_loader:
        optimizer.zero_grad()
        
        # Forward pass
        output = model(src, trg)  # output: [trg_len, batch, output_dim]
        
        # Shift output and target for loss
        output = output[1:].reshape(-1, OUTPUT_DIM)       # skip <sos>
        trg = trg[1:].reshape(-1)                         # skip <sos>
        
        loss = criterion(output, trg)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
    
    print(f"[Epoch {epoch}] Loss: {epoch_loss:.4f}")
