# Python Client – Setup & Publishing Guide

The **deadlockdb** Python package lives in [`clients/python/`](../clients/python/) and provides a lightweight, zero-dependency HTTP client for the deadlock-db vector database.

---

## Prerequisites

| Requirement | Version |
|-------------|---------|
| Python | ≥ 3.8 |
| pip | any recent version |
| A running deadlock-db server | see [Server Setup](#starting-the-server) |

## Starting the Server

```bash
# Build from source
go build -o deadlock-db ./cmd/deadlock-db

# Run (in-memory, no persistence)
./deadlock-db -addr :8080

# Run with disk persistence
./deadlock-db -addr :8080 -data /path/to/data
```

## Installing the Client

### From the repository (local development)

```bash
# From the repo root
pip install clients/python

# Or with an editable install for development
pip install -e clients/python
```

### From PyPI (when published)

```bash
pip install deadlockdb
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
```

> See [`examples/python/basic.py`](../examples/python/basic.py) for a complete runnable example.

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

### Collection Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | *required* | Collection name |
| `dimension` | `int` | *required* | Vector dimensionality |
| `metric` | `str` | `"cosine"` | Distance metric (`cosine`, `euclidean`, `dotproduct`) |
| `m` | `int` | `16` | HNSW max connections per layer |
| `ef_construction` | `int` | `200` | HNSW construction expansion factor |
| `quantization` | `str` | `"none"` | Quantization type (`none`, `scalar`, `pq`, `binary`) |

## Publishing to PyPI

### 1. Install build tools

```bash
pip install build twine
```

### 2. Build the package

```bash
cd clients/python
python -m build
```

This produces a `dist/` directory with `.tar.gz` and `.whl` files.

### 3. Upload to PyPI

```bash
# Test upload (recommended first)
twine upload --repository testpypi dist/*

# Production upload
twine upload dist/*
```

> You will need a PyPI account and an API token. See <https://pypi.org/help/#apitoken> for details.

### 4. Verify the published package

```bash
pip install deadlockdb
python -c "from deadlockdb import DeadlockClient; print('OK')"
```

## Development

```bash
cd clients/python

# Install in editable mode
pip install -e .

# Run tests (requires a running server on localhost:8080)
python -m pytest tests/
```

## Further Reading

- [Python client README](../clients/python/README.md)
- [TypeScript client guide](typescript-client.md)
- [Main project README](../README.md)
