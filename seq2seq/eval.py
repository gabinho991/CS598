import argparse, json, re, torch, sqlite3, os
from model import Seq2Seq
from data_utils import load_pairs_jsonl, build_vocab, batchify
from torch.utils.data import DataLoader
from rdflib import Graph
import pickle

def remove_literals(tokens):
    # Remove string and number literals
    return [t for t in tokens if not re.fullmatch(r'".*?"', t) and not re.fullmatch(r'\d+(\.\d+)?', t)]

def is_logic_form_correct(pred, gold):
    return pred == gold

def is_structural_match(pred, gold):
    return remove_literals(pred) == remove_literals(gold)

def exec_sql_query(query, conn):
    try:
        cursor = conn.execute(query)
        return sorted([tuple(row) for row in cursor.fetchall()])
    except Exception:
        return None

def exec_sparql_query(query, graph):
    try:
        return sorted([tuple(r.values()) for r in graph.query(query)])
    except Exception:
        return None

def run_eval(model, data_loader, vocab, sql_db=None, kg_file=None):
    inv_vocab = {v: k for k, v in vocab.items()}
    acc_lf, acc_st, acc_ex, total = 0, 0, 0, 0
    conn = sqlite3.connect(sql_db) if sql_db else None
    graph = Graph().parse(kg_file, format='ttl') if kg_file else None

    for q_pad, q_len, s_pad in data_loader:
        model.eval()
        with torch.no_grad():
            logits = model(q_pad, q_len, None, tf_ratio=0.0)
            preds = logits.argmax(-1)

        for i in range(q_pad.size(0)):
            pred_tokens = [inv_vocab[t.item()] for t in preds[i] if t.item() != vocab["<pad>"]]
            gold_tokens = [inv_vocab[t.item()] for t in s_pad[i][1:] if t.item() != vocab["<pad>"]]
            pred_str = ' '.join(pred_tokens).replace('<eos>', '').strip()
            gold_str = ' '.join(gold_tokens).replace('<eos>', '').strip()

            if is_logic_form_correct(pred_tokens, gold_tokens):
                acc_lf += 1
            if is_structural_match(pred_tokens, gold_tokens):
                acc_st += 1

            # Execution accuracy
            gold_out, pred_out = None, None
            if sql_db:
                gold_out = exec_sql_query(gold_str, conn)
                pred_out = exec_sql_query(pred_str, conn)
            elif kg_file:
                gold_out = exec_sparql_query(gold_str, graph)
                pred_out = exec_sparql_query(pred_str, graph)

            if gold_out is not None and pred_out is not None and gold_out == pred_out:
                acc_ex += 1
            total += 1

    if conn:
        conn.close()
    return acc_lf / total, acc_st / total, acc_ex / total

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help="Path to test jsonl file")
    parser.add_argument('--model', required=False, default='best_model.pt')
    parser.add_argument('--sql_db', help='Path to SQL database file')
    parser.add_argument('--kg_file', help='Path to RDF knowledge graph file')
    args = parser.parse_args()

    pairs = load_pairs_jsonl(args.data)
    model_dir = os.path.dirname(args.model)
    vocab_path = os.path.join(model_dir, "best_vocab.pkl")
    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)
    model = Seq2Seq(vocab)
    model.load_state_dict(torch.load(args.model, map_location='cpu'))

    loader = DataLoader(pairs, batch_size=32, shuffle=False, collate_fn=lambda b: batchify(b, vocab))
    acc_lf, acc_st, acc_ex = run_eval(model, loader, vocab, args.sql_db, args.kg_file)

    print(f"Logic Form Accuracy (AccLF):      {acc_lf:.3f}")
    print(f"Structural Accuracy (AccST):       {acc_st:.3f}")
    print(f"Execution Accuracy (AccEX):        {acc_ex:.3f}")

if __name__ == "__main__":
    main()
