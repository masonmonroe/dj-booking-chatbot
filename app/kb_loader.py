"""
kb_loader.py

Reads knowledge_base.txt, splits it into sections by [HEADER] blocks,
and loads each section as a document into ChromaDB.

Called once at bot startup. Safe to call multiple times — skips sections
that are already loaded (upsert by section ID).
"""

import re
import chromadb

# Shared ChromaDB client and collection
chroma_client = chromadb.Client()
collection    = chroma_client.get_or_create_collection(name="dj_kb")


def parse_knowledge_base(filepath="data/knowledge_base.txt"):
    """
    Parses a .txt file structured with [SECTION_HEADER] blocks.

    Returns a list of dicts:
        [{"id": "profile", "text": "..."}, ...]

    Each [HEADER] becomes the document ID (lowercased, spaces → underscores).
    Everything between two headers is the document body.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        raw = f.read()

    # Split on [HEADER] markers
    pattern = re.compile(r"\[([A-Z_/ ]+)\]", re.MULTILINE)
    parts   = pattern.split(raw)

    # parts alternates: ["preamble", "HEADER1", "body1", "HEADER2", "body2", ...]
    entries = []
    for i in range(1, len(parts) - 1, 2):
        header = parts[i].strip().lower().replace(" ", "_").replace("/", "_")
        body   = parts[i + 1].strip()

        if body:  # skip empty sections
            entries.append({"id": header, "text": body})

    return entries


def load_knowledge_base(filepath="data/knowledge_base.txt"):
    """
    Loads all KB sections into ChromaDB.
    Skips any section whose ID is already in the collection.
    Prints a summary of what was loaded vs skipped.
    """
    entries = parse_knowledge_base(filepath)

    # Get existing IDs to avoid duplicates
    existing_ids = set()
    try:
        existing     = collection.get()
        existing_ids = set(existing["ids"])
    except Exception:
        pass

    loaded  = []
    skipped = []

    for entry in entries:
        if entry["id"] in existing_ids:
            skipped.append(entry["id"])
            continue

        collection.add(
            documents=[entry["text"]],
            ids=[entry["id"]],
        )
        loaded.append(entry["id"])

    print(f"[KB] Loaded  : {loaded  or 'none (already up to date)'}")
    print(f"[KB] Skipped : {skipped or 'none'}")


def retrieve_context(query, n_results=3):
    """
    Query ChromaDB for the most relevant KB sections.
    Returns a single string with results joined by newlines.
    """
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
    )
    docs = results.get("documents", [[]])[0]
    return "\n\n".join(docs) if docs else ""


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    load_knowledge_base()
    print("\n--- Sample retrieval: 'pricing and packages' ---")
    print(retrieve_context("pricing and packages"))
