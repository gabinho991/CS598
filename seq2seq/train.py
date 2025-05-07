import argparse, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
import os
from model      import Seq2Seq
from data_utils import load_pairs_jsonl, build_vocab, batchify
import pickle
# --------------------------------------------------------
# one pass over data
# --------------------------------------------------------
def run_epoch(model, data_loader, vocab, optimiser=None, epoch=1):
    ce          = nn.CrossEntropyLoss(ignore_index=vocab["<pad>"], reduction="none")
    tf_ratio    = max(0.1, 0.9 * (0.95 ** (epoch - 1)))       # scheduled‑sampling
    tok_correct = tok_total = 0
    total_loss  = 0

    for q_pad, q_len, s_pad in data_loader:
        logits = model(q_pad, q_len, s_pad[:, :-1], tf_ratio=tf_ratio)     # [B,T,V]

        tgt     = s_pad[:, 1:]                           # gold without <sos>
        eosmask = (tgt != vocab["<eos>"]).float()        # mask out tokens after first <eos>
        loss    = ce(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
        loss    = (loss * eosmask.reshape(-1)).sum() / eosmask.sum()

        if optimiser is not None:                        # training mode
            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()

        # ----- token‑level accuracy --------------------------------------
        preds      = logits.argmax(-1)
        valmask    = (tgt != vocab["<pad>"])
        tok_correct+= ((preds == tgt) & valmask).sum().item()
        tok_total  += valmask.sum().item()
        total_loss += loss.item()
        
    return tok_correct / max(1, tok_total), total_loss / len(data_loader)


# --------------------------------------------------------
# main training script
# --------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--train', required=True)
    p.add_argument('--dev',   required=True)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--bs',     type=int, default=32)
    args = p.parse_args()
    best_acc = 0.0
    patience = 3          # how many epochs to wait after no improvement
    tolerance = 1e-3      # minimum change to consider an improvement
    patience_counter = 0
    train_pairs = load_pairs_jsonl(args.train)
    dev_pairs   = load_pairs_jsonl(args.dev)
    vocab       = build_vocab(train_pairs + dev_pairs)
    save_dir = os.path.dirname(args.train)
    model_path = os.path.join(save_dir, "best_model.pt")
    model      = Seq2Seq(vocab)
    optimiser  = optim.Adam(model.parameters(), lr=1e-3)
    scheduler  = optim.lr_scheduler.StepLR(optimiser, step_size=2, gamma=0.8)

    train_loader = DataLoader(train_pairs, batch_size=args.bs, shuffle=True,
                              collate_fn=lambda b: batchify(b, vocab))
    dev_loader   = DataLoader(dev_pairs,   batch_size=args.bs, shuffle=False,
                              collate_fn=lambda b: batchify(b, vocab))

    for ep in range(1, args.epochs + 1):
        model.train()
        run_epoch(model, train_loader, vocab, optimiser, epoch=ep)
        scheduler.step()

        model.eval()
        dev_acc, dev_loss = run_epoch(model, dev_loader, vocab, optimiser=None, epoch=ep)
        print(f"Epoch {ep:2d}: Dev TokAcc={dev_acc:.3f}  |  Loss={dev_loss:.3f}")

        if dev_acc > best_acc + tolerance:
            best_acc = dev_acc
            patience_counter = 0
            # Optionally: save model
            vocab_path = os.path.join(save_dir, "best_vocab.pkl")
            torch.save(model.state_dict(), model_path)
            with open(vocab_path, "wb") as f:
                pickle.dump(vocab, f)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Stopping early at epoch {ep} (no improvement in {patience} rounds)")
                break

if __name__ == "__main__":
    main()