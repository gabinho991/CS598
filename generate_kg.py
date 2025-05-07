import json
import argparse
from rdflib import Graph, URIRef, Literal, Namespace

def extract_triples_from_sparql(sql_str):
    triples = []
    if "where" not in sql_str.lower():
        return triples
    pattern = sql_str.lower().split("where")[1]
    pattern = pattern.strip(" {}")
    for line in pattern.split('.'):
        if not line.strip():
            continue
        parts = line.strip().split()
        if len(parts) < 3:
            continue
        subj, pred, obj = parts[:3]
        subj = URIRef(subj.strip("?<>\""))
        pred = URIRef(pred.strip("<>\""))
        if obj.startswith("?"):
            obj = URIRef(obj.strip("?<>\""))
        elif "^^" in obj:
            val, _ = obj.split("^^")
            obj = Literal(val.strip("\""))
        else:
            obj = Literal(obj.strip("\""))
        triples.append((subj, pred, obj))
    return triples

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Path to SPARQL JSON dataset')
    parser.add_argument('--output', required=True, help='Path to output .ttl KG file')
    args = parser.parse_args()

    g = Graph()
    with open(args.input, 'r') as f:
        for line in f:
            data = json.loads(line)
            sql = data.get("sql", "")
            for s, p, o in extract_triples_from_sparql(sql):
                g.add((s, p, o))

    g.serialize(destination=args.output, format='turtle')
    print(f"KG saved to {args.output}")

if __name__ == "__main__":
    main()
