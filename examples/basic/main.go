// Example demonstrates basic usage of deadlock-db.
package main

import (
	"fmt"
	"math/rand"

	deadlockdb "github.com/deadlock-labs/deadlock-db"
	"github.com/deadlock-labs/deadlock-db/pkg/core"
	"github.com/deadlock-labs/deadlock-db/pkg/distance"
	"github.com/deadlock-labs/deadlock-db/pkg/filter"
)

func main() {
	fmt.Println("=== deadlock-db Example ===")
	fmt.Println()

	// Create an in-memory database
	db, err := deadlockdb.Open(nil)
	if err != nil {
		panic(err)
	}
	defer db.Close()

	// Create a collection with 128-dimensional vectors
	fmt.Println("Creating collection...")
	coll, err := db.CreateCollection(
		"documents",
		128,
		deadlockdb.WithMetric(distance.Cosine),
		deadlockdb.WithHNSWParams(16, 200),
	)
	if err != nil {
		panic(err)
	}
	fmt.Printf("Created collection: %s (dimension: %d)\n\n", coll.Name(), coll.Dimension())

	// Insert vectors
	fmt.Println("Inserting 1000 vectors...")
	categories := []string{"tech", "science", "arts", "sports", "health"}
	for i := 0; i < 1000; i++ {
		values := make([]float32, 128)
		for j := range values {
			values[j] = rand.Float32()*2 - 1 // Random values in [-1, 1]
		}

		vec := core.NewVectorWithMetadata(
			fmt.Sprintf("doc-%d", i),
			values,
			map[string]any{
				"category": categories[i%len(categories)],
				"year":     2020 + (i % 5),
				"score":    rand.Float64() * 100,
			},
		)
		if err := coll.Insert(vec); err != nil {
			panic(err)
		}
	}
	fmt.Printf("Inserted %d vectors\n\n", coll.Count())

	// Basic search
	fmt.Println("=== Basic Search ===")
	query := make([]float32, 128)
	for i := range query {
		query[i] = rand.Float32()*2 - 1
	}

	results, err := db.Search("documents", query, 5)
	if err != nil {
		panic(err)
	}

	fmt.Println("Top 5 results:")
	for i, r := range results {
		fmt.Printf("  %d. ID: %s, Score: %.4f\n", i+1, r.ID, r.Score)
	}
	fmt.Println()

	// Filtered search
	fmt.Println("=== Filtered Search (category='tech', year>=2023) ===")
	f := filter.And(
		filter.NewFilter("category", "tech"),
		filter.NewComparisonFilter("year", filter.OpGte, 2023),
	)

	filtered, err := db.SearchWithFilter("documents", query, 5, f)
	if err != nil {
		panic(err)
	}

	fmt.Println("Top 5 filtered results:")
	for i, r := range filtered {
		fmt.Printf("  %d. ID: %s, Score: %.4f, Category: %s, Year: %v\n",
			i+1, r.ID, r.Score, r.Metadata["category"], r.Metadata["year"])
	}
	fmt.Println()

	// Get specific vector
	fmt.Println("=== Get Vector ===")
	vec, err := db.Get("documents", "doc-42")
	if err != nil {
		panic(err)
	}
	fmt.Printf("Retrieved vector: ID=%s, Dimension=%d, Metadata=%v\n\n",
		vec.ID, len(vec.Values), vec.Metadata)

	// Collection stats
	fmt.Println("=== Collection Stats ===")
	stats := coll.Stats()
	fmt.Printf("Name: %s\n", stats["name"])
	fmt.Printf("Count: %v\n", stats["count"])
	fmt.Printf("Dimension: %v\n", stats["dimension"])
	fmt.Printf("Metric: %v\n", stats["metric"])
	fmt.Println()

	// Delete a vector
	fmt.Println("=== Delete Vector ===")
	fmt.Printf("Count before delete: %d\n", coll.Count())
	if err := db.Delete("documents", "doc-42"); err != nil {
		panic(err)
	}
	fmt.Printf("Count after delete: %d\n\n", coll.Count())

	// List collections
	fmt.Println("=== List Collections ===")
	collections := db.ListCollections()
	for _, name := range collections {
		fmt.Printf("  - %s\n", name)
	}
	fmt.Println()

	fmt.Println("Example complete!")
}
