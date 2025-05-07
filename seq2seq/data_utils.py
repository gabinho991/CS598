# seq2seq/data_utils.py
import json, torch, re
from torch.nn.utils.rnn import pad_sequence

SPECIAL = ["<pad>", "<sos>", "<eos>", "<unk>"]

def load_pairs_jsonl(path):
    pairs = []
    with open(path, 'r') as f:
        for line in f:
            data = json.loads(line)
            question = data.get("question_refine_tok", [])
            raw_sql = data.get("sql_tok", [])
            
            # Clean SQL tokens
            cleaned_sql = []
            for token in raw_sql:
                token = token.replace('"', '') # Remove quotes/backslashes
                if token.startswith("^^<http"):  # preserve entire type annotation
                    cleaned_sql.append(token)
                else:
                    # Token-split around SPARQL operators (while keeping them)
                    parts = re.split(r'([()\{\}=<>])', token)
                    cleaned_sql.extend([t for t in parts if t and not t.isspace()])
            
            pairs.append((question, cleaned_sql))
    return pairs

def build_vocab(pairs, min_freq=1):
    freq={}
    for q, s in pairs:
        for tk in q+s: freq[tk]=freq.get(tk,0)+1
    itos = SPECIAL + [w for w,c in freq.items() if c>=min_freq]
    return {w:i for i,w in enumerate(itos)}

def tokens2ids(tokens, vocab):
    return [vocab.get(t,"<unk>") for t in tokens]

def batchify(batch, vocab):
    qs, sqls = zip(*batch)
    q_ids = [torch.tensor(tokens2ids(q,vocab)+[vocab["<eos>"]]) for q in qs]
    s_ids = [torch.tensor(tokens2ids(s,vocab)+[vocab["<eos>"]]) for s in sqls]
    q_pad = pad_sequence(q_ids, batch_first=True, padding_value=vocab["<pad>"])
    s_pad = pad_sequence(s_ids, batch_first=True, padding_value=vocab["<pad>"])
    q_lens = torch.tensor([len(x) for x in q_ids])
    return q_pad, q_lens, s_pad
