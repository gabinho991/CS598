# save as create_subset.py
import json

def create_subset(in_path, out_path, num_samples=100):
    with open(in_path, 'r') as f:
        lines = f.readlines()
        data = [json.loads(line) for line in lines]

    subset = data[:num_samples]

    with open(out_path, 'w') as f:
        for item in subset:
            f.write(json.dumps(item) + '\n')

# Example usage
create_subset(
    'dataset/mimic_sparqlstar/natural/train.json',
    'dataset/mimic_sparqlstar/natural/train_small.json',
    num_samples=100
)
