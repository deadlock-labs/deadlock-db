# deadlock-db

[![Go Reference](https://pkg.go.dev/badge/github.com/deadlock-labs/deadlock-db.svg)](https://pkg.go.dev/github.com/deadlock-labs/deadlock-db)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A high-performance vector database written in Go, designed to compete with and exceed the capabilities of existing vector databases like ChromaDB, Pinecone, Milvus, and Qdrant.

## Features

### 🚀 Performance
- **HNSW Index**: Hierarchical Navigable Small World graph for lightning-fast approximate nearest neighbor (ANN) search
- **Optimized Distance Calculations**: Loop-unrolled distance computations for better CPU utilization
- **Concurrent Operations**: Thread-safe operations with minimal lock contention
- **Memory Efficiency**: Vector pooling and optimized memory layouts

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

### 💾 Storage Options
- **In-Memory**: Fast ephemeral storage for testing and small datasets
- **Disk Persistence**: Write-ahead logging (WAL) for durability
- **Recovery**: Automatic recovery from crashes

### 🗜️ Quantization
- **Scalar Quantization (SQ8)**: 4x memory reduction with minimal accuracy loss
- **Product Quantization (PQ)**: Up to 32x memory reduction for large-scale deployments
- **Binary Quantization**: Extreme compression for specific use cases

### 🔌 API
- RESTful HTTP API for easy integration
- Go client library for native applications
- Collection management (create, delete, list)
- Vector CRUD operations
- Batch operations

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

Example results on a modern machine:

| Operation | Vectors | Dimension | Time |
|-----------|---------|-----------|------|
| Insert | 1 | 128 | ~15µs |
| Search (k=10, ef=100) | 10,000 | 128 | ~200µs |
| Distance (Cosine) | - | 1024 | ~500ns |

## Comparison with Other Vector DBs

| Feature | deadlock-db | ChromaDB | Pinecone | Milvus | Qdrant |
|---------|------------|----------|----------|--------|--------|
| Language | Go | Python | Cloud | C++/Go | Rust |
| In-memory | ✅ | ✅ | ❌ | ✅ | ✅ |
| Disk persistence | ✅ | ✅ | ✅ | ✅ | ✅ |
| HNSW index | ✅ | ✅ | ✅ | ✅ | ✅ |
| Hybrid search | ✅ | ✅ | ✅ | ✅ | ✅ |
| Scalar quantization | ✅ | ❌ | ❌ | ✅ | ✅ |
| Product quantization | ✅ | ❌ | ❌ | ✅ | ✅ |
| Sparse vectors | ✅ | ❌ | ✅ | ✅ | ✅ |
| Embedded mode | ✅ | ✅ | ❌ | ❌ | ✅ |
| No external deps | ✅ | ❌ | N/A | ❌ | ❌ |

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- HNSW algorithm: Malkov & Yashunin (2016)
- Product quantization: Jégou et al. (2011)
