import json

from datasets import load_dataset


def parse_labels(labels):
    if isinstance(labels, str):
        return json.loads(labels)
    return labels


def main():
    print("Loading ToolRet queries...")

    queries = load_dataset(
        "mangopy/ToolRet-Queries",
        "apibank",
        split="queries",
    )

    print(f"Loaded {len(queries)} queries.")

    print("\nLoading ToolRet web tools...")

    tools = load_dataset(
        "mangopy/ToolRet-Tools",
        "web",
        split="tools",
    )

    print(f"Loaded {len(tools)} tools.")

    print("\nChecking relevant tool IDs...")

    tool_ids = set(tools["id"])

    relevant_tool_ids = set()

    for query in queries:
        labels = parse_labels(query["labels"])

        for label in labels:
            if label["relevance"] > 0:
                relevant_tool_ids.add(label["id"])

    missing_tool_ids = relevant_tool_ids - tool_ids

    print(f"Relevant tool IDs found in queries: {len(relevant_tool_ids)}")
    print(f"Relevant tool IDs missing from corpus: {len(missing_tool_ids)}")

    if missing_tool_ids:
        print("\nExample missing IDs:")
        for tool_id in sorted(missing_tool_ids)[:10]:
            print(tool_id)
    else:
        print("\nAll relevant tool IDs exist in the ToolRet web corpus. ✅")

    print("\nWriting ToolRet data files...")

    corpus_path = "data/toolret/corpus.jsonl"
    queries_path = "data/toolret/queries.jsonl"

    with open(corpus_path, "w", encoding="utf-8") as corpus_file:
        for tool in tools:
            record = {
                "id": tool["id"],
                "text": tool["documentation"],
            }
            corpus_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    with open(queries_path, "w", encoding="utf-8") as queries_file:
        for query in queries:
            labels = parse_labels(query["labels"])

            relevant_ids = [
                label["id"]
                for label in labels
                if label["relevance"] > 0
            ]

            record = {
                "id": query["id"],
                "query": query["query"],
                "relevant_ids": relevant_ids,
            }

            queries_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote {len(tools)} tools to {corpus_path}")
    print(f"Wrote {len(queries)} queries to {queries_path}")

if __name__ == "__main__":
    main()