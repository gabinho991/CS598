import json
import random
from pathlib import Path
from copy import deepcopy

# Load original dataset
input_path = Path("dataset/mimic_sparqlstar/template/test.json")
with open(input_path, 'r') as f:
    original_data = [json.loads(line) for line in f]

# Define a function to slightly paraphrase questions and reword SQL tokens for variation
def augment_entry(entry):
    q = entry.get("question_refine_tok", [])
    sql = entry.get("sql_tok", [])

    question_variants = {
        "how many": ["count the number of", "give me the number of", "total number of"],
        "calculate": ["compute", "find out", "determine"],
        "patients": ["subjects", "individuals", "cases"],
        "provide": ["show", "return", "list"]
    }

    new_q = [random.choice(question_variants.get(tok, [tok])) for tok in q]

    # Optionally shuffle SQL tokens
    new_sql = deepcopy(sql)
    if random.random() < 0.5:
        random.shuffle(new_sql)

    return {
        "question_refine_tok": new_q,
        "sql_tok": new_sql
    }

# Augment data
augmented_data = [augment_entry(entry) for entry in original_data]

# Save new dataset
output_path = Path("dataset/mimic_augmented/template/test_augmented.json")
output_path.parent.mkdir(parents=True, exist_ok=True)
with open(output_path, 'w') as f:
    for entry in augmented_data:
        f.write(json.dumps(entry) + "\n")
