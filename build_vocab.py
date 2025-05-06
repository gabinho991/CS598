from utils import load_data
from vocab import Vocab

data = load_data("data/train.json")

question_tokens = [item["question"] for item in data]
query_tokens = [item["query"] for item in data]

question_vocab = Vocab(min_freq=1)
query_vocab = Vocab(min_freq=1)

question_vocab.build(question_tokens)
query_vocab.build(query_tokens)

print(f"Question vocab size: {len(question_vocab)}")
print(f"Query vocab size: {len(query_vocab)}")

question_vocab.save("data/question_vocab.json")
query_vocab.save("data/query_vocab.json")
