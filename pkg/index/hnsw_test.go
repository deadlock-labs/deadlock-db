package index

import (
	"fmt"
	"math/rand"
	"testing"

	"github.com/deadlock-labs/deadlock-db/pkg/core"
	"github.com/deadlock-labs/deadlock-db/pkg/distance"
)

func TestHNSWInsertAndSearch(t *testing.T) {
	config := DefaultHNSWConfig(128)
	idx := NewHNSW(config)

	// Insert vectors
	numVectors := 100
	vectors := generateRandomVectors(numVectors, 128)

	for i, v := range vectors {
		vec := core.NewVector(fmt.Sprintf("vec-%d", i), v)
		if err := idx.Insert(vec); err != nil {
			t.Fatalf("failed to insert vector %d: %v", i, err)
		}
	}

	if idx.Size() != numVectors {
		t.Errorf("expected size %d, got %d", numVectors, idx.Size())
	}

	// Search
	query := vectors[0]
	results := idx.Search(query, 10, 50)

	if len(results) == 0 {
		t.Error("expected at least one result")
	}

	// First result should be the query vector itself
	if results[0].ID != "vec-0" {
		t.Errorf("expected first result to be vec-0, got %s", results[0].ID)
	}
}

func TestHNSWDelete(t *testing.T) {
	config := DefaultHNSWConfig(64)
	idx := NewHNSW(config)

	// Insert vectors
	for i := 0; i < 50; i++ {
		v := make([]float32, 64)
		for j := range v {
			v[j] = rand.Float32()
		}
		vec := core.NewVector(fmt.Sprintf("vec-%d", i), v)
		if err := idx.Insert(vec); err != nil {
			t.Fatalf("insert failed: %v", err)
		}
	}

	// Delete a vector
	if err := idx.Delete("vec-10"); err != nil {
		t.Fatalf("delete failed: %v", err)
	}

	if idx.Size() != 49 {
		t.Errorf("expected size 49 after delete, got %d", idx.Size())
	}

	// Verify deletion
	if _, found := idx.GetVector("vec-10"); found {
		t.Error("deleted vector should not be found")
	}
}

func TestHNSWUpdate(t *testing.T) {
	config := DefaultHNSWConfig(32)
	idx := NewHNSW(config)

	// Insert a vector
	original := []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32}
	vec := core.NewVector("test-vec", original)
	if err := idx.Insert(vec); err != nil {
		t.Fatalf("insert failed: %v", err)
	}

	// Update with new values
	updated := make([]float32, 32)
	for i := range updated {
		updated[i] = float32(i * 2)
	}
	vec2 := core.NewVector("test-vec", updated)
	if err := idx.Insert(vec2); err != nil {
		t.Fatalf("update failed: %v", err)
	}

	// Size should remain the same
	if idx.Size() != 1 {
		t.Errorf("expected size 1 after update, got %d", idx.Size())
	}
}

func TestHNSWDifferentMetrics(t *testing.T) {
	metrics := []distance.Metric{
		distance.Cosine,
		distance.Euclidean,
		distance.DotProduct,
	}

	for _, metric := range metrics {
		t.Run(metric.String(), func(t *testing.T) {
			config := DefaultHNSWConfig(64)
			config.Metric = metric
			idx := NewHNSW(config)

			// Insert vectors
			for i := 0; i < 20; i++ {
				v := make([]float32, 64)
				for j := range v {
					v[j] = rand.Float32()
				}
				vec := core.NewVector(fmt.Sprintf("vec-%d", i), v)
				if err := idx.Insert(vec); err != nil {
					t.Fatalf("insert failed: %v", err)
				}
			}

			// Search should work
			query := make([]float32, 64)
			for i := range query {
				query[i] = rand.Float32()
			}
			results := idx.Search(query, 5, 20)
			if len(results) != 5 {
				t.Errorf("expected 5 results, got %d", len(results))
			}
		})
	}
}

func TestHNSWStats(t *testing.T) {
	config := DefaultHNSWConfig(32)
	idx := NewHNSW(config)

	for i := 0; i < 10; i++ {
		v := make([]float32, 32)
		for j := range v {
			v[j] = rand.Float32()
		}
		vec := core.NewVector(fmt.Sprintf("vec-%d", i), v)
		idx.Insert(vec)
	}

	stats := idx.Stats()
	if stats["size"].(uint64) != 10 {
		t.Errorf("expected size 10 in stats")
	}
	if stats["m"].(int) != config.M {
		t.Errorf("expected m=%d in stats", config.M)
	}
}

func generateRandomVectors(n, dim int) [][]float32 {
	vectors := make([][]float32, n)
	for i := 0; i < n; i++ {
		vectors[i] = make([]float32, dim)
		for j := 0; j < dim; j++ {
			vectors[i][j] = rand.Float32()
		}
	}
	return vectors
}

func BenchmarkHNSWInsert(b *testing.B) {
	config := DefaultHNSWConfig(128)
	idx := NewHNSW(config)

	vectors := generateRandomVectors(b.N, 128)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		vec := core.NewVector(fmt.Sprintf("vec-%d", i), vectors[i])
		idx.Insert(vec)
	}
}

func BenchmarkHNSWSearch(b *testing.B) {
	config := DefaultHNSWConfig(128)
	idx := NewHNSW(config)

	// Insert 10000 vectors
	vectors := generateRandomVectors(10000, 128)
	for i, v := range vectors {
		vec := core.NewVector(fmt.Sprintf("vec-%d", i), v)
		idx.Insert(vec)
	}

	query := vectors[0]

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		idx.Search(query, 10, 100)
	}
}
