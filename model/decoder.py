import torch.nn as nn
import torch
from model.attention import Attention

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, attention):
        super().__init__()
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim + hid_dim * 2, hid_dim)
        self.fc_out = nn.Linear(hid_dim * 3, output_dim)
        self.attention = attention

    def forward(self, input, hidden, encoder_outputs):
        input = input.unsqueeze(0)
        embedded = self.embedding(input)
        attn_weights = self.attention(hidden, encoder_outputs)
        attn_applied = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs.permute(1, 0, 2))
        rnn_input = torch.cat((embedded, attn_applied.permute(1, 0, 2)), dim=2)
        output, (hidden, cell) = self.rnn(rnn_input, (hidden, torch.zeros_like(hidden)))
        output = self.fc_out(torch.cat((output.squeeze(0), attn_applied.squeeze(1)), dim=1))
        return output, hidden
