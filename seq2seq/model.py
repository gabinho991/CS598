# seq2seq/model.py
import torch, torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

class Encoder(nn.Module):
    def __init__(self, vocab, emb=256, hid=512):
        super().__init__()
        self.emb = nn.Embedding(len(vocab), emb, padding_idx=vocab["<pad>"])
        self.rnn = nn.LSTM(emb, hid // 2, batch_first=True,
                           bidirectional=True)

    def forward(self, src, src_lens):
        # src: [B, T]
        emb = self.emb(src)
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, src_lens, batch_first=True, enforce_sorted=False)
        h, (hn, cn) = self.rnn(packed)
        enc_out, _ = nn.utils.rnn.pad_packed_sequence(h, batch_first=True)
        return enc_out, (hn, cn)

class LuongAttention(nn.Module):
    def __init__(self, hid):
        super().__init__()
        self.W = nn.Linear(hid, hid, bias=False)

    def forward(self, dec_h, enc_out, mask):
        # dec_h: [B, 1, H]   enc_out: [B, S, H]
        scores = torch.bmm(self.W(dec_h), enc_out.transpose(1, 2))  # [B,1,S]
        scores.masked_fill_(mask[:,None,:]==0, -1e9)
        attn = torch.softmax(scores, dim=-1)         # [B,1,S]
        ctx  = torch.bmm(attn, enc_out)              # [B,1,H]
        return ctx, attn.squeeze(1)

class Decoder(nn.Module):
    def __init__(self, vocab, hid=512, emb=256, attn=True):
        super().__init__()
        self.emb = nn.Embedding(len(vocab), emb, padding_idx=vocab["<pad>"])
        self.rnn = nn.LSTM(emb+hid, hid, batch_first=True)
        self.attn = LuongAttention(hid) if attn else None
        self.fc   = nn.Linear(hid*2 if attn else hid, len(vocab))

    def forward(self, tgt, enc_out, enc_mask, hidden):
    # tgt: [B, 1]
        emb = self.emb(tgt)  # [B,1,E]
        
        # compute context vector from attention
        context, _ = self.attn(hidden[0].transpose(0, 1), enc_out, enc_mask)  # [B,1,H]

        # concatenate embedding and context, pass to RNN
        rnn_input = torch.cat([emb, context], dim=-1)  # [B,1,E+H]
        dec_h, hidden = self.rnn(rnn_input, hidden)    # [B,1,H]

        # output projection
        out = torch.cat([dec_h, context], dim=-1)      # [B,1,2H]
        logits = self.fc(out.squeeze(1))               # [B,V]
        return logits, hidden

class Seq2Seq(nn.Module):
    def __init__(self, vocab, emb=256, hid=512, sos="<sos>", eos="<eos>"):
        super().__init__()
        self.vocab, self.sos, self.eos = vocab, vocab[sos], vocab[eos]
        self.enc = Encoder(vocab, emb, hid)
        self.dec = Decoder(vocab, hid, emb, attn=True)

    def forward(self, src, src_lens, tgt=None, tf_ratio=0.9, max_len=150):
        enc_out, (hn,cn) = self.enc(src, src_lens)
        mask = (src!=self.vocab["<pad>"]).int()
        batch = src.size(0)
        hidden = (torch.cat([hn[-2],hn[-1]],dim=-1).unsqueeze(0),
                  torch.cat([cn[-2],cn[-1]],dim=-1).unsqueeze(0))
        dec_inp = torch.full((batch,1), self.sos, dtype=torch.long, device=src.device)
        outputs=[]
        steps = tgt.size(1) if tgt is not None else max_len
        for t in range(steps):
            logits, hidden = self.dec(dec_inp, enc_out, mask, hidden)
            outputs.append(logits)
            _, topi = logits.max(1, keepdim=True)
            dec_inp = tgt[:,t:t+1] if (tgt is not None and torch.rand(1).item()<tf_ratio) else topi
        return torch.stack(outputs, dim=1)  # [B,T,V]
