# deadlockdb – Python Client

A lightweight Python client for the [deadlock-db](https://github.com/deadlock-labs/deadlock-db) vector database.

## Installation

```bash
pip install deadlockdb          # from PyPI (when published)
# or install from source
pip install clients/python      # from repo root
```

## Quick Start

```python
from deadlockdb import DeadlockClient

client = DeadlockClient("http://localhost:8080")

# Create a collection
client.create_collection("embeddings", dimension=128, metric="cosine")

# Insert vectors
client.insert("embeddings", [
    {"id": "vec-1", "values": [0.1] * 128, "metadata": {"category": "tech"}},
    {"id": "vec-2", "values": [0.2] * 128, "metadata": {"category": "science"}},
])

# Search
results = client.search("embeddings", vector=[0.15] * 128, k=5)
for r in results:
    print(f"  {r['id']}: {r['score']:.4f}")

# Search with metadata filter
results = client.search(
    "embeddings",
    vector=[0.15] * 128,
    k=5,
    filter={"category": "tech"},
)

# Get / delete a vector
vec = client.get("embeddings", "vec-1")
client.delete("embeddings", "vec-1")

# Delete the collection
client.delete_collection("embeddings")
```

## API Reference

| Method | Description |
|--------|-------------|
| `health()` | Server health check |
| `version()` | Server version |
| `list_collections()` | List all collections |
| `create_collection(name, dimension, ...)` | Create a collection |
| `get_collection(name)` | Get collection info |
| `delete_collection(name)` | Delete a collection |
| `insert(collection, vectors)` | Insert vectors |
| `get(collection, vector_id)` | Get a vector by ID |
| `delete(collection, vector_id)` | Delete a vector |
| `search(collection, vector, k, filter=)` | Similarity search |
