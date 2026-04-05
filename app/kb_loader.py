import re

_kb_documents = []

def load_knowledge_base(filepath="data/knowledge_base.txt"):
    global _kb_documents
    with open(filepath, "r", encoding="utf-8") as f:
        raw = f.read()

    pattern = re.compile(r"\[([A-Z_/ ]+)\]", re.MULTILINE)
    parts   = pattern.split(raw)
    entries = []

    for i in range(1, len(parts) - 1, 2):
        header = parts[i].strip().lower().replace(" ", "_").replace("/", "_")
        body   = parts[i + 1].strip()
        if body:
            entries.append({"id": header, "text": body})

    _kb_documents = entries
    print(f"[KB] Loaded {len(entries)} sections: {[e['id'] for e in entries]}")


def retrieve_context(query, n_results=3):
    if not _kb_documents:
        return ""

    query_words = set(query.lower().split())

    scored = []
    for doc in _kb_documents:
        doc_words = set(doc["text"].lower().split())
        score     = len(query_words & doc_words)
        scored.append((score, doc["id"], doc["text"]))

    scored.sort(key=lambda x: x[0], reverse=True)

    # Always include pricing in context — it's the most common query
    top_ids   = [id for _, id, _ in scored[:n_results]]
    top_texts = [text for _, id, text in scored[:n_results]]

    if "pricing_rules" not in top_ids:
        pricing = next((d["text"] for d in _kb_documents if d["id"] == "pricing_rules"), None)
        if pricing:
            top_texts.append(pricing)

    return "\n\n".join(top_texts)