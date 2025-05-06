import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils import load_data
from vocab import Vocab
from model.encoder import Encoder
from model.decoder import Decoder
from model.attention import Attention
from model.seq2seq import Seq2Seq
from torch.nn.utils.rnn import pad_sequence

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data


q_vocab = Vocab.load("data/question_vocab.json")
s_vocab = Vocab.load("data/query_vocab.json")

def prepare_pair(item):
    q_ids = torch.tensor(q_vocab.encode(item["question"]), dtype=torch.long)
    s_ids = torch.tensor(s_vocab.encode(["<sos>"] + item["query"] + ["<eos>"]), dtype=torch.long)
    return q_ids, s_ids

pairs = [prepare_pair(d) for d in data]

def collate_fn(batch):
    src, trg = zip(*batch)
    src_pad = pad_sequence(src, padding_value=0)
    trg_pad = pad_sequence(trg, padding_value=0)
    return src_pad.to(DEVICE), trg_pad.to(DEVICE)

loader = DataLoader(pairs, batch_size=32, shuffle=True, collate_fn=collate_fn)

# Hyperparameters
INPUT_DIM = len(q_vocab)
OUTPUT_DIM = len(s_vocab)
EMB_DIM = 128
HID_DIM = 256
EPOCHS = 15

# Model
attention = Attention(HID_DIM)
encoder = Encoder(INPUT_DIM, EMB_DIM, HID_DIM)
decoder = Decoder(OUTPUT_DIM, EMB_DIM, HID_DIM, attention)
model = Seq2Seq(encoder, decoder).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss(ignore_index=0)

# Training loop
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    for src, trg in loader:
        optimizer.zero_grad()
        output = model(src, trg)
        output = output.view(-1, OUTPUT_DIM)
        trg_flat = trg[1:].reshape(-1)
        loss = criterion(output, trg_flat)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"Epoch {epoch+1}: Loss = {epoch_loss:.4f}")
