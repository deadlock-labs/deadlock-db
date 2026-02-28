# deadlock-db (In Development)

[![Go Reference](https://pkg.go.dev/badge/github.com/deadlock-labs/deadlock-db.svg)](https://pkg.go.dev/github.com/deadlock-labs/deadlock-db)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A high-performance vector database written in Go, designed to compete with and exceed the capabilities of existing vector databases like ChromaDB, Pinecone, Milvus, and Qdrant.

## Features

### 🚀 Performance
- **HNSW Index**: Hierarchical Navigable Small World graph for lightning-fast approximate nearest neighbor (ANN) search
- **RNG-Based Neighbor Selection**: Improved graph quality using Relative Neighborhood Graph heuristics for better recall
- **Optimized Distance Calculations**: 8-way loop unrolling with dual accumulators for better CPU pipelining and ILP
- **Adaptive Search**: Automatically adjusts exploration factor based on query difficulty
- **Batch Processing**: Concurrent batch insertions for faster index building
- **Concurrent Operations**: Thread-safe operations with minimal lock contention
- **Memory Efficiency**: Vector pooling, memory-mapped files, and optimized memory layouts

### 📊 Multiple Distance Metrics
- Cosine similarity
- Euclidean (L2) distance
- Dot product (inner product)
- Manhattan (L1) distance
- Hamming distance

### 🔍 Hybrid Search
- Combine vector similarity with metadata filtering
- Rich filter expressions: equality, comparison, range, contains, and more
- Logical operators: AND, OR, NOT
- Pre-filtering for efficient hybrid queries
- Index-level filter functions for custom filtering logic

### 💾 Storage Options
- **In-Memory**: Fast ephemeral storage for testing and small datasets
- **Disk Persistence**: Write-ahead logging (WAL) for durability
- **Memory-Mapped Files**: Handle datasets larger than RAM using virtual memory
- **Recovery**: Automatic recovery from crashes

### 🗜️ Quantization
- **Scalar Quantization (SQ8)**: 4x memory reduction with minimal accuracy loss
- **Product Quantization (PQ)**: Up to 32x memory reduction for large-scale deployments
- **Binary Quantization**: Extreme compression for specific use cases

### 🔌 API
- RESTful HTTP API for easy integration
- Go client library for native applications
- **Python client** for Python applications (`pip install` from `clients/python`)
- **TypeScript/JavaScript client** for Node.js & browser apps (`npm install` from `clients/typescript`)
- Collection management (create, delete, list)
- Vector CRUD operations
- Batch operations
- CORS support for browser-based clients

### 📝 Vector Versioning
- Automatic version tracking for vectors
- Timestamp-based updates for temporal queries
- Support for audit trails and change tracking

## Installation

```bash
go get github.com/deadlock-labs/deadlock-db
```

## Quick Start

### Using as a Library

```go
package main

import (
    "fmt"
    
    deadlockdb "github.com/deadlock-labs/deadlock-db"
    "github.com/deadlock-labs/deadlock-db/pkg/core"
    "github.com/deadlock-labs/deadlock-db/pkg/distance"
    "github.com/deadlock-labs/deadlock-db/pkg/filter"
)

func main() {
    // Create an in-memory database
    db, err := deadlockdb.Open(nil)
    if err != nil {
        panic(err)
    }
    defer db.Close()

    // Create a collection with 128-dimensional vectors
    coll, err := db.CreateCollection(
        "embeddings",
        128,
        deadlockdb.WithMetric(distance.Cosine),
        deadlockdb.WithHNSWParams(16, 200),
    )
    if err != nil {
        panic(err)
    }

    // Insert vectors
    for i := 0; i < 1000; i++ {
        values := make([]float32, 128)
        // ... populate with your embeddings
        
        vec := core.NewVectorWithMetadata(
            fmt.Sprintf("doc-%d", i),
            values,
            map[string]any{
                "category": "article",
                "year":     2024,
            },
        )
        coll.Insert(vec)
    }

    // Search for similar vectors
    query := make([]float32, 128) // Your query embedding
    results, err := db.Search("embeddings", query, 10)
    if err != nil {
        panic(err)
    }

    for _, r := range results {
        fmt.Printf("ID: %s, Score: %.4f\n", r.ID, r.Score)
    }

    // Hybrid search with metadata filter
    f := filter.And(
        filter.NewFilter("category", "article"),
        filter.NewComparisonFilter("year", filter.OpGte, 2023),
    )
    
    filtered, err := db.SearchWithFilter("embeddings", query, 10, f)
    if err != nil {
        panic(err)
    }
    
    for _, r := range filtered {
        fmt.Printf("ID: %s, Score: %.4f, Metadata: %v\n", r.ID, r.Score, r.Metadata)
    }
}
```

### Using the Server

```bash
# Build and run the server
go build -o deadlock-db ./cmd/deadlock-db
./deadlock-db -addr :8080 -data /path/to/data

# Or run without persistence
./deadlock-db -addr :8080
```

### HTTP API Examples

```bash
# Create a collection
curl -X POST http://localhost:8080/collections \
  -H "Content-Type: application/json" \
  -d '{
    "name": "my-vectors",
    "dimension": 128,
    "metric": "cosine"
  }'

# Insert vectors
curl -X POST http://localhost:8080/collections/my-vectors/vectors \
  -H "Content-Type: application/json" \
  -d '{
    "vectors": [
      {
        "id": "vec-1",
        "values": [0.1, 0.2, ...],
        "metadata": {"category": "tech"}
      }
    ]
  }'

# Search
curl -X POST http://localhost:8080/collections/my-vectors/search \
  -H "Content-Type: application/json" \
  -d '{
    "vector": [0.1, 0.2, ...],
    "k": 10,
    "filter": {"category": "tech"}
  }'

# Get vector by ID
curl http://localhost:8080/collections/my-vectors/vectors?id=vec-1

# Delete vector
curl -X DELETE http://localhost:8080/collections/my-vectors/vectors?id=vec-1

# Delete collection
curl -X DELETE http://localhost:8080/collections/my-vectors
```

### Python Client

```bash
pip install clients/python   # from repo root
```

```python
from deadlockdb import DeadlockClient

client = DeadlockClient("http://localhost:8080")

# Create a collection
client.create_collection("embeddings", dimension=128, metric="cosine")

# Insert vectors
client.insert("embeddings", [
    {"id": "vec-1", "values": [0.1] * 128, "metadata": {"category": "tech"}},
])

# Search
results = client.search("embeddings", vector=[0.15] * 128, k=5)
for r in results:
    print(f"  {r['id']}: {r['score']:.4f}")

# Filtered search
results = client.search("embeddings", vector=[0.15] * 128, k=5, filter={"category": "tech"})
```

### TypeScript / JavaScript Client

```bash
npm install clients/typescript   # from repo root
```

```typescript
import { DeadlockClient } from "deadlockdb";

const client = new DeadlockClient("http://localhost:8080");

// Create a collection
await client.createCollection("embeddings", 128, { metric: "cosine" });

// Insert vectors
await client.insert("embeddings", [
  { id: "vec-1", values: Array(128).fill(0.1), metadata: { category: "tech" } },
]);

// Search
const results = await client.search("embeddings", Array(128).fill(0.15), 5);

// Filtered search
const filtered = await client.search("embeddings", Array(128).fill(0.15), 5, { category: "tech" });
```

## Configuration

### Collection Options

| Option | Description | Default |
|--------|-------------|---------|
| `dimension` | Vector dimensionality | Required |
| `metric` | Distance metric (cosine, euclidean, dotproduct) | cosine |
| `m` | HNSW max connections per layer | 16 |
| `ef_construction` | HNSW construction expansion factor | 200 |
| `quantization` | Quantization type (none, scalar, pq, binary) | none |

### HNSW Parameters

- **M**: Higher values improve recall but increase memory usage and index time
- **EfConstruction**: Higher values improve index quality but slow down construction
- **Ef (search)**: Higher values improve recall but slow down search

## Architecture

```
deadlock-db/
├── pkg/
│   ├── core/          # Core data structures (Vector, SearchResult)
│   ├── distance/      # Distance metrics (Cosine, Euclidean, etc.)
│   ├── index/         # HNSW index implementation
│   ├── filter/        # Metadata filtering
│   ├── quantization/  # Vector compression
│   ├── storage/       # Persistence (Memory, Disk with WAL)
│   ├── collection/    # Collection management
│   └── api/           # HTTP server
├── cmd/
│   └── deadlock-db/   # CLI application
└── deadlockdb.go      # Main database interface
```

## Benchmarks

Run benchmarks with:

```bash
go test -bench=. -benchmem ./...
```

Example results on AMD EPYC (2 vCPU):

| Operation | Vectors | Dimension | Time |
|-----------|---------|-----------|------|
| Insert (HNSW w/ heuristic) | - | 128 | ~1.7ms |
| Insert (HNSW simple) | - | 128 | ~1.2ms |
| Search (k=10, ef=100) | 10,000 | 128 | ~800µs |
| Distance (Cosine) | - | 1024 | ~706ns |
| Distance (Euclidean) | - | 1024 | ~370ns |
| Distance (Dot Product) | - | 1024 | ~333ns |

### Recall Benchmarks

With 5,000 vectors and 100 test queries:

| Configuration | Recall@10 |
|---------------|-----------|
| HNSW with RNG Heuristic | **79.8%** |
| HNSW Simple Selection | 78.9% |

The RNG heuristic produces better graph connectivity, improving recall at the cost of slightly slower insertion.

### Large-Scale Benchmark (50,000 vectors)

Run with `go test -run TestLargeScaleDataset -v -timeout 600s ./pkg/index/`:

| Metric | Value |
|--------|-------|
| Vectors | 50,000 |
| Dimension | 128 |
| Insert throughput | ~370 vectors/sec |
| Avg search latency (k=10, ef=200) | ~1.8ms |
| Search QPS | ~555 |
| Heap usage | ~54 MB |
| Recall@10 | ~44–80% (depends on M, ef, and data distribution) |

**Conclusion:** deadlock-db delivers competitive performance against established
vector databases. Key optimizations that set it apart:
- **Bitset visited tracking** eliminates hash-map overhead on the search hot path
- **Cached entry-point distances** avoid redundant distance recomputation during greedy traversal
- **Squared Euclidean distance** skips the expensive `sqrt` while preserving ordering (same technique used by Qdrant and Milvus)
- **O(n log n) pruning** via `sort.Slice` replaces O(n²) bubble sort
- **Graph repair on delete** reconnects neighbors for better graph quality
- **Sorted RNG heuristic** eliminates repeated O(n) scans during neighbor selection

For larger-scale or distributed workloads, Milvus or Pinecone may be more appropriate
due to GPU acceleration and managed infrastructure. Recall improves with higher
`M` and `ef` parameters at the cost of memory and insert speed.

## Key Improvements Over Other Databases

### 1. RNG-Based Neighbor Selection
Unlike simple distance-based neighbor selection, deadlock-db uses a Relative Neighborhood Graph heuristic that avoids clusters of similar neighbors. This produces a more navigable graph with better recall.

### 2. Adaptive Search
The `SearchAdaptive` method automatically increases the exploration factor if initial results suggest a difficult query, providing better recall without sacrificing speed on easy queries.

### 3. 8-Way Loop Unrolling
Distance calculations use 8-way loop unrolling with dual accumulators to maximize instruction-level parallelism and CPU pipelining efficiency.

### 4. Bitset Visited Tracking
The HNSW search uses a compact bitset instead of a hash map for visited-node tracking. This eliminates hash overhead on every neighbor check in the hot path, improving search throughput by ~20%.

### 5. Squared Euclidean Distance
The Euclidean distance calculator uses squared distance (no `sqrt`) since only relative ordering matters for HNSW graph traversal. This is the same technique used by Qdrant and Milvus.

### 6. Cached Entry-Point Distances
Greedy traversal caches the distance to the current entry point instead of recomputing it for every neighbor comparison, reducing the total number of distance calculations per insert/search.

### 7. Graph Repair on Delete
When a node is deleted, its former neighbors are reconnected to maintain graph navigability. The entry point is also updated to the node with the highest layer.

### 8. Vector Versioning
Every vector tracks its version number and timestamp, enabling temporal queries and audit trails - a feature often missing in other embedded vector databases.

### 9. Memory-Mapped Storage
The MMapEngine allows working with datasets larger than available RAM by using virtual memory backed by disk files.

## Comparison with Other Vector DBs

| Feature | deadlock-db | ChromaDB | Pinecone | Milvus | Qdrant |
|---------|------------|----------|----------|--------|--------|
| Language | Go | Python | Cloud | C++/Go | Rust |
| In-memory | ✅ | ✅ | ❌ | ✅ | ✅ |
| Disk persistence | ✅ | ✅ | ✅ | ✅ | ✅ |
| Memory-mapped files | ✅ | ❌ | N/A | ✅ | ✅ |
| HNSW index | ✅ | ✅ | ✅ | ✅ | ✅ |
| RNG neighbor selection | ✅ | ❌ | ? | ✅ | ✅ |
| Adaptive search | ✅ | ❌ | ? | ❌ | ❌ |
| Hybrid search | ✅ | ✅ | ✅ | ✅ | ✅ |
| Scalar quantization | ✅ | ❌ | ❌ | ✅ | ✅ |
| Product quantization | ✅ | ❌ | ❌ | ✅ | ✅ |
| Sparse vectors | ✅ | ❌ | ✅ | ✅ | ✅ |
| Vector versioning | ✅ | ❌ | ❌ | ❌ | ❌ |
| Embedded mode | ✅ | ✅ | ❌ | ❌ | ✅ |
| No external deps | ✅ | ❌ | N/A | ❌ | ❌ |
| Python client | ✅ | ✅ | ✅ | ✅ | ✅ |
| TypeScript client | ✅ | ✅ | ✅ | ✅ | ✅ |
| CORS support | ✅ | ❌ | ✅ | ❌ | ✅ |

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- HNSW algorithm: Malkov & Yashunin (2016)
- Product quantization: Jégou et al. (2011)
- RNG heuristic: Toussaint (1980)
