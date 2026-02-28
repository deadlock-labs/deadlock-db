# TypeScript / JavaScript Client – Setup & Publishing Guide

The **deadlockdb** npm package lives in [`clients/typescript/`](../clients/typescript/) and provides a lightweight TypeScript client for the deadlock-db vector database.

---

## Prerequisites

| Requirement | Version |
|-------------|---------|
| Node.js | ≥ 18 (uses the built-in `fetch` API) |
| npm | any recent version |
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
npm install clients/typescript

# Or link for development
cd clients/typescript
npm install
npm run build
npm link

# Then in your project
npm link deadlockdb
```

### From npm (when published)

```bash
npm install deadlockdb
```

## Quick Start

### TypeScript

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
```

### JavaScript (CommonJS)

```js
const { DeadlockClient } = require("deadlockdb");

async function main() {
  const client = new DeadlockClient("http://localhost:8080");
  await client.createCollection("embeddings", 128, { metric: "cosine" });

  await client.insert("embeddings", [
    { id: "vec-1", values: Array(128).fill(0.1), metadata: { category: "tech" } },
  ]);

  const results = await client.search("embeddings", Array(128).fill(0.15), 5);
  console.log(results);
}

main();
```

> See [`examples/typescript/basic.ts`](../examples/typescript/basic.ts) for a complete runnable example.

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

### CreateCollectionOptions

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `metric` | `string` | `"cosine"` | Distance metric (`cosine`, `euclidean`, `dotproduct`) |
| `m` | `number` | `16` | HNSW max connections per layer |
| `efConstruction` | `number` | `200` | HNSW construction expansion factor |
| `quantization` | `string` | `"none"` | Quantization type (`none`, `scalar`, `pq`, `binary`) |

## Publishing to npm

### 1. Build the package

```bash
cd clients/typescript
npm install
npm run build
```

### 2. Login to npm

```bash
npm login
```

### 3. Publish

```bash
# Dry run first (recommended)
npm publish --dry-run

# Publish to npm
npm publish
```

> The `prepublishOnly` script in `package.json` automatically runs `npm run build` before publishing.

### 4. Verify the published package

```bash
npm install deadlockdb
node -e "const { DeadlockClient } = require('deadlockdb'); console.log('OK');"
```

## Development

```bash
cd clients/typescript

# Install dependencies
npm install

# Build
npm run build

# Run tests (requires a running server on localhost:8080)
npm test
```

## Further Reading

- [TypeScript client README](../clients/typescript/README.md)
- [Python client guide](python-client.md)
- [Main project README](../README.md)
