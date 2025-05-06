from nltk.translate.bleu_score import sentence_bleu
from utils import load_data
from vocab import Vocab
from model.seq2seq import Seq2Seq
import torch

def evaluate(model, data, q_vocab, s_vocab, max_len=50):
    model.eval()
    exact_matches = 0
    total = 0
    bleu_scores = []

    with torch.no_grad():
        for sample in data:
            input_ids = torch.tensor(q_vocab.encode(sample["question"]), dtype=torch.long).unsqueeze(1).to(model.device)
            trg_ids = s_vocab.encode(["<sos>"] + sample["query"] + ["<eos>"])
            reference = sample["query"]

            # Generate prediction
            output_ids = model.predict(input_ids, max_len=max_len)
            pred_tokens = s_vocab.decode(output_ids)

            # Clean output (stop at <eos>)
            if "<eos>" in pred_tokens:
                pred_tokens = pred_tokens[:pred_tokens.index("<eos>")]

            if pred_tokens == reference:
                exact_matches += 1

            bleu = sentence_bleu([reference], pred_tokens, weights=(0.5, 0.5))
            bleu_scores.append(bleu)
            total += 1

    em_score = exact_matches / total * 100
    avg_bleu = sum(bleu_scores) / len(bleu_scores)

    print(f"Exact Match: {em_score:.2f}%")
    print(f"Average BLEU: {avg_bleu:.4f}")
