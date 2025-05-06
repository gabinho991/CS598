from collections import Counter
import json

class Vocab:
    def __init__(self, min_freq=1):
        self.token2idx = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3}
        self.idx2token = {0: "<pad>", 1: "<sos>", 2: "<eos>", 3: "<unk>"}
        self.min_freq = min_freq
        self.freqs = Counter()
        self.frozen = False

    def build(self, token_lists):
        for tokens in token_lists:
            self.freqs.update(tokens)
        idx = len(self.token2idx)
        for token, freq in self.freqs.items():
            if freq >= self.min_freq and token not in self.token2idx:
                self.token2idx[token] = idx
                self.idx2token[idx] = token
                idx += 1
        self.frozen = True

    def __len__(self):
        return len(self.token2idx)

    def encode(self, tokens):
        return [self.token2idx.get(t, self.token2idx["<unk>"]) for t in tokens]

    def decode(self, ids):
        return [self.idx2token.get(i, "<unk>") for i in ids]

    def save(self, filepath):
        with open(filepath, "w") as f:
            json.dump({"token2idx": self.token2idx}, f)

    @classmethod
    def load(cls, filepath):
        with open(filepath) as f:
            data = json.load(f)
        vocab = cls()
        vocab.token2idx = data["token2idx"]
        vocab.idx2token = {int(i): t for t, i in vocab.token2idx.items()}
        vocab.frozen = True
        return vocab
