// Example demonstrates advanced features of deadlock-db.
package main

import (
	"fmt"
	"math/rand"
	"time"

	deadlockdb "github.com/deadlock-labs/deadlock-db"
	"github.com/deadlock-labs/deadlock-db/pkg/core"
	"github.com/deadlock-labs/deadlock-db/pkg/distance"
	"github.com/deadlock-labs/deadlock-db/pkg/filter"
	"github.com/deadlock-labs/deadlock-db/pkg/index"
)

func main() {
	fmt.Println("=== deadlock-db Advanced Features Demo ===")
	fmt.Println()

	// 1. Demonstrate vector versioning
	fmt.Println("1. Vector Versioning")
	fmt.Println("-------------------")
	vec := core.NewVector("doc-1", []float32{0.1, 0.2, 0.3, 0.4})
	fmt.Printf("   New vector - ID: %s, Version: %d, Timestamp: %d\n", vec.ID, vec.Version, vec.Timestamp)

	time.Sleep(10 * time.Millisecond)
	vec.IncrementVersion()
	fmt.Printf("   After update - Version: %d, Timestamp: %d\n\n", vec.Version, vec.Timestamp)

	// 2. Demonstrate adaptive search vs regular search
	fmt.Println("2. Adaptive Search")
	fmt.Println("------------------")

	dim := 64
	numVectors := 2000

	config := index.DefaultHNSWConfig(dim)
	config.UseHeuristic = true
	idx := index.NewHNSW(config)

	// Insert vectors
	rand.Seed(42)
	for i := 0; i < numVectors; i++ {
		v := make([]float32, dim)
		for j := 0; j < dim; j++ {
			v[j] = rand.Float32()*2 - 1
		}
		vec := core.NewVector(fmt.Sprintf("vec-%d", i), v)
		idx.Insert(vec)
	}

	query := make([]float32, dim)
	for j := 0; j < dim; j++ {
		query[j] = rand.Float32()*2 - 1
	}

	// Regular search
	start := time.Now()
	results := idx.Search(query, 10, 100)
	regularTime := time.Since(start)

	// Adaptive search
	start = time.Now()
	adaptiveResults := idx.SearchAdaptive(query, 10, 0.9)
	adaptiveTime := time.Since(start)

	fmt.Printf("   Regular search: %d results in %v\n", len(results), regularTime)
	fmt.Printf("   Adaptive search: %d results in %v\n\n", len(adaptiveResults), adaptiveTime)

	// 3. Demonstrate RNG heuristic vs simple selection
	fmt.Println("3. RNG Heuristic vs Simple Selection")
	fmt.Println("------------------------------------")

	// Already shown in recall benchmarks, but show stats
	stats := idx.Stats()
	fmt.Printf("   Index stats with RNG heuristic enabled:\n")
	fmt.Printf("   - Size: %v\n", stats["size"])
	fmt.Printf("   - Max Layer: %v\n", stats["max_layer"])
	fmt.Printf("   - Total Connections: %v\n", stats["total_connections"])
	fmt.Printf("   - Use Heuristic: %v\n\n", stats["use_heuristic"])

	// 4. Demonstrate batch insert
	fmt.Println("4. Batch Insert")
	fmt.Println("---------------")

	configBatch := index.DefaultHNSWConfig(dim)
	idxBatch := index.NewHNSW(configBatch)

	// Create batch of vectors
	vectors := make([]*core.Vector, 500)
	for i := 0; i < 500; i++ {
		v := make([]float32, dim)
		for j := 0; j < dim; j++ {
			v[j] = rand.Float32()
		}
		vectors[i] = core.NewVector(fmt.Sprintf("batch-vec-%d", i), v)
	}

	start = time.Now()
	err := idxBatch.InsertBatch(vectors)
	batchTime := time.Since(start)

	if err != nil {
		fmt.Printf("   Error: %v\n", err)
	} else {
		fmt.Printf("   Inserted 500 vectors in %v (%.2f vectors/sec)\n\n",
			batchTime, float64(500)/batchTime.Seconds())
	}

	// 5. Demonstrate hybrid search with filter
	fmt.Println("5. Hybrid Search with Collection")
	fmt.Println("--------------------------------")

	db, _ := deadlockdb.Open(nil)
	defer db.Close()

	coll, _ := db.CreateCollection("demo", 32, deadlockdb.WithMetric(distance.Cosine))

	categories := []string{"tech", "science", "arts", "sports"}
	for i := 0; i < 100; i++ {
		values := make([]float32, 32)
		for j := range values {
			values[j] = rand.Float32()
		}
		vec := core.NewVectorWithMetadata(
			fmt.Sprintf("doc-%d", i),
			values,
			map[string]any{
				"category": categories[i%4],
				"year":     2020 + i%5,
				"score":    float64(i) * 0.5,
			},
		)
		coll.Insert(vec)
	}

	q := make([]float32, 32)
	for i := range q {
		q[i] = rand.Float32()
	}

	// Search with filter
	f := filter.And(
		filter.NewFilter("category", "tech"),
		filter.NewComparisonFilter("year", filter.OpGte, 2023),
	)

	filtered, _ := db.SearchWithFilter("demo", q, 5, f)
	fmt.Printf("   Found %d tech articles from 2023+:\n", len(filtered))
	for _, r := range filtered {
		fmt.Printf("   - %s (year: %v, score: %.1f)\n",
			r.ID, r.Metadata["year"], r.Metadata["score"])
	}

	fmt.Println("\n=== Demo Complete ===")
}
