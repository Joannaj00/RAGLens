import json
from datasets import load_dataset
import os

os.makedirs("data/raw", exist_ok=True)

OUTPUT_FILE = "data/raw/docs.jsonl"
DATASET_NAME = "gfissore/arxiv-abstracts-2021"
N = 300

def main():
    dataset = load_dataset(DATASET_NAME, split="train") # take the training portion of the dataset
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for i in range(N):
            row = dataset[i]
            doc = {
                "id": row["id"],
                "title": row["title"],
                "abstract": row["abstract"],
                "source": DATASET_NAME
            }
            f.write(json.dumps(doc) + "\n")
    print(f"Wrote {N} documents to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()