package collection

import (
	"fmt"
	"math/rand"
	"testing"

	"github.com/deadlock-labs/deadlock-db/pkg/core"
	"github.com/deadlock-labs/deadlock-db/pkg/distance"
	"github.com/deadlock-labs/deadlock-db/pkg/filter"
)

func TestCollectionCreate(t *testing.T) {
	cfg := DefaultConfig("test-collection", 128)
	coll, err := New(cfg)
	if err != nil {
		t.Fatalf("failed to create collection: %v", err)
	}
	defer coll.Close()

	if coll.Name() != "test-collection" {
		t.Errorf("expected name 'test-collection', got '%s'", coll.Name())
	}
	if coll.Dimension() != 128 {
		t.Errorf("expected dimension 128, got %d", coll.Dimension())
	}
	if coll.Count() != 0 {
		t.Errorf("expected count 0, got %d", coll.Count())
	}
}

func TestCollectionInsertAndGet(t *testing.T) {
	cfg := DefaultConfig("test", 64)
	coll, err := New(cfg)
	if err != nil {
		t.Fatalf("failed to create collection: %v", err)
	}
	defer coll.Close()

	// Insert a vector
	values := make([]float32, 64)
	for i := range values {
		values[i] = float32(i)
	}
	vec := core.NewVectorWithMetadata("vec-1", values, map[string]any{"category": "test"})

	if err := coll.Insert(vec); err != nil {
		t.Fatalf("failed to insert: %v", err)
	}

	if coll.Count() != 1 {
		t.Errorf("expected count 1, got %d", coll.Count())
	}

	// Get the vector
	retrieved, err := coll.Get("vec-1")
	if err != nil {
		t.Fatalf("failed to get: %v", err)
	}

	if retrieved.ID != "vec-1" {
		t.Errorf("expected ID 'vec-1', got '%s'", retrieved.ID)
	}
	if len(retrieved.Values) != 64 {
		t.Errorf("expected 64 values, got %d", len(retrieved.Values))
	}
}

func TestCollectionDelete(t *testing.T) {
	cfg := DefaultConfig("test", 32)
	coll, err := New(cfg)
	if err != nil {
		t.Fatalf("failed to create collection: %v", err)
	}
	defer coll.Close()

	// Insert vectors
	for i := 0; i < 10; i++ {
		values := make([]float32, 32)
		for j := range values {
			values[j] = rand.Float32()
		}
		vec := core.NewVector(fmt.Sprintf("vec-%d", i), values)
		coll.Insert(vec)
	}

	if coll.Count() != 10 {
		t.Errorf("expected count 10, got %d", coll.Count())
	}

	// Delete a vector
	if err := coll.Delete("vec-5"); err != nil {
		t.Fatalf("failed to delete: %v", err)
	}

	if coll.Count() != 9 {
		t.Errorf("expected count 9, got %d", coll.Count())
	}

	// Verify it's gone
	_, err = coll.Get("vec-5")
	if err == nil {
		t.Error("expected error when getting deleted vector")
	}
}

func TestCollectionSearch(t *testing.T) {
	cfg := DefaultConfig("test", 64)
	cfg.Metric = distance.Cosine
	coll, err := New(cfg)
	if err != nil {
		t.Fatalf("failed to create collection: %v", err)
	}
	defer coll.Close()

	// Insert vectors
	numVectors := 100
	for i := 0; i < numVectors; i++ {
		values := make([]float32, 64)
		for j := range values {
			values[j] = rand.Float32()
		}
		vec := core.NewVectorWithMetadata(
			fmt.Sprintf("vec-%d", i),
			values,
			map[string]any{"index": i},
		)
		coll.Insert(vec)
	}

	// Search
	query := make([]float32, 64)
	for i := range query {
		query[i] = rand.Float32()
	}

	results, err := coll.Search(query, DefaultSearchOptions(10))
	if err != nil {
		t.Fatalf("search failed: %v", err)
	}

	if len(results) != 10 {
		t.Errorf("expected 10 results, got %d", len(results))
	}
}

func TestCollectionSearchWithFilter(t *testing.T) {
	cfg := DefaultConfig("test", 32)
	coll, err := New(cfg)
	if err != nil {
		t.Fatalf("failed to create collection: %v", err)
	}
	defer coll.Close()

	// Insert vectors with different categories
	categories := []string{"tech", "science", "arts"}
	for i := 0; i < 90; i++ {
		values := make([]float32, 32)
		for j := range values {
			values[j] = rand.Float32()
		}
		vec := core.NewVectorWithMetadata(
			fmt.Sprintf("vec-%d", i),
			values,
			map[string]any{"category": categories[i%3], "index": i},
		)
		coll.Insert(vec)
	}

	// Search with filter
	query := make([]float32, 32)
	for i := range query {
		query[i] = rand.Float32()
	}

	f := filter.NewFilter("category", "tech")
	results, err := coll.SearchWithFilter(query, 10, f)
	if err != nil {
		t.Fatalf("filtered search failed: %v", err)
	}

	// All results should have category "tech"
	for _, r := range results {
		if r.Metadata["category"] != "tech" {
			t.Errorf("result %s has wrong category: %v", r.ID, r.Metadata["category"])
		}
	}
}

func TestCollectionUpdate(t *testing.T) {
	cfg := DefaultConfig("test", 16)
	coll, err := New(cfg)
	if err != nil {
		t.Fatalf("failed to create collection: %v", err)
	}
	defer coll.Close()

	// Insert initial vector
	values := make([]float32, 16)
	for i := range values {
		values[i] = 1.0
	}
	vec := core.NewVectorWithMetadata("vec-1", values, map[string]any{"version": 1})
	coll.Insert(vec)

	// Update values
	newValues := make([]float32, 16)
	for i := range newValues {
		newValues[i] = 2.0
	}
	err = coll.Update("vec-1", newValues, map[string]any{"version": 2})
	if err != nil {
		t.Fatalf("update failed: %v", err)
	}

	// Verify update
	updated, err := coll.Get("vec-1")
	if err != nil {
		t.Fatalf("get failed: %v", err)
	}

	if updated.Values[0] != 2.0 {
		t.Errorf("expected values[0]=2.0, got %f", updated.Values[0])
	}
}

func TestCollectionStats(t *testing.T) {
	cfg := DefaultConfig("test-stats", 32)
	coll, err := New(cfg)
	if err != nil {
		t.Fatalf("failed to create collection: %v", err)
	}
	defer coll.Close()

	// Insert some vectors
	for i := 0; i < 50; i++ {
		values := make([]float32, 32)
		for j := range values {
			values[j] = rand.Float32()
		}
		coll.Insert(core.NewVector(fmt.Sprintf("vec-%d", i), values))
	}

	stats := coll.Stats()
	
	if stats["name"] != "test-stats" {
		t.Errorf("expected name 'test-stats', got %v", stats["name"])
	}
	if stats["dimension"] != 32 {
		t.Errorf("expected dimension 32, got %v", stats["dimension"])
	}
	if stats["count"].(uint64) != 50 {
		t.Errorf("expected count 50, got %v", stats["count"])
	}
}

func TestCollectionDimensionMismatch(t *testing.T) {
	cfg := DefaultConfig("test", 64)
	coll, err := New(cfg)
	if err != nil {
		t.Fatalf("failed to create collection: %v", err)
	}
	defer coll.Close()

	// Try to insert vector with wrong dimension
	values := make([]float32, 32) // Wrong dimension
	vec := core.NewVector("vec-1", values)

	err = coll.Insert(vec)
	if err == nil {
		t.Error("expected error for dimension mismatch")
	}
}

func BenchmarkCollectionInsert(b *testing.B) {
	cfg := DefaultConfig("bench", 128)
	coll, _ := New(cfg)
	defer coll.Close()

	vectors := make([]*core.Vector, b.N)
	for i := 0; i < b.N; i++ {
		values := make([]float32, 128)
		for j := range values {
			values[j] = rand.Float32()
		}
		vectors[i] = core.NewVector(fmt.Sprintf("vec-%d", i), values)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		coll.Insert(vectors[i])
	}
}

func BenchmarkCollectionSearch(b *testing.B) {
	cfg := DefaultConfig("bench", 128)
	coll, _ := New(cfg)
	defer coll.Close()

	// Insert 10000 vectors
	for i := 0; i < 10000; i++ {
		values := make([]float32, 128)
		for j := range values {
			values[j] = rand.Float32()
		}
		coll.Insert(core.NewVector(fmt.Sprintf("vec-%d", i), values))
	}

	query := make([]float32, 128)
	for i := range query {
		query[i] = rand.Float32()
	}
	opts := DefaultSearchOptions(10)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		coll.Search(query, opts)
	}
}
