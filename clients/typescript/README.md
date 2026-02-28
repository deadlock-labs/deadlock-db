# deadlockdb – TypeScript / JavaScript Client

A lightweight TypeScript client for the [deadlock-db](https://github.com/deadlock-labs/deadlock-db) vector database.

## Installation

```bash
npm install deadlockdb        # from npm (when published)
# or install from source
npm install clients/typescript # from repo root
```

## Quick Start

```ts
import { DeadlockClient } from "deadlockdb";

const client = new DeadlockClient("http://localhost:8080");

// Create a collection
await client.createCollection("embeddings", 128, { metric: "cosine" });

// Insert vectors
await client.insert("embeddings", [
  { id: "vec-1", values: Array(128).fill(0.1), metadata: { category: "tech" } },
  { id: "vec-2", values: Array(128).fill(0.2), metadata: { category: "science" } },
]);

// Search
const results = await client.search("embeddings", Array(128).fill(0.15), 5);
for (const r of results) {
  console.log(`  ${r.id}: ${r.score.toFixed(4)}`);
}

// Search with metadata filter
const filtered = await client.search(
  "embeddings",
  Array(128).fill(0.15),
  5,
  { category: "tech" },
);

// Get / delete a vector
const vec = await client.get("embeddings", "vec-1");
await client.delete("embeddings", "vec-1");

// Delete the collection
await client.deleteCollection("embeddings");
```

## API Reference

| Method | Description |
|--------|-------------|
| `health()` | Server health check |
| `version()` | Server version |
| `listCollections()` | List all collections |
| `createCollection(name, dimension, options?)` | Create a collection |
| `getCollection(name)` | Get collection info |
| `deleteCollection(name)` | Delete a collection |
| `insert(collection, vectors)` | Insert vectors |
| `get(collection, vectorId)` | Get a vector by ID |
| `delete(collection, vectorId)` | Delete a vector |
| `search(collection, vector, k, filter?)` | Similarity search |
