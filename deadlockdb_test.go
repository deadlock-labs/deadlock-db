package deadlockdb

import (
	"fmt"
	"math/rand"
	"os"
	"path/filepath"
	"testing"

	"github.com/deadlock-labs/deadlock-db/pkg/core"
	"github.com/deadlock-labs/deadlock-db/pkg/distance"
	"github.com/deadlock-labs/deadlock-db/pkg/filter"
)

func TestDBOpen(t *testing.T) {
	db, err := Open(nil)
	if err != nil {
		t.Fatalf("failed to open db: %v", err)
	}
	defer db.Close()

	if len(db.ListCollections()) != 0 {
		t.Error("expected no collections initially")
	}
}

func TestDBCreateCollection(t *testing.T) {
	db, _ := Open(nil)
	defer db.Close()

	coll, err := db.CreateCollection("test-collection", 128)
	if err != nil {
		t.Fatalf("failed to create collection: %v", err)
	}

	if coll.Name() != "test-collection" {
		t.Errorf("expected name 'test-collection', got '%s'", coll.Name())
	}

	// Verify it's listed
	collections := db.ListCollections()
	if len(collections) != 1 {
		t.Errorf("expected 1 collection, got %d", len(collections))
	}
}

func TestDBCreateCollectionDuplicate(t *testing.T) {
	db, _ := Open(nil)
	defer db.Close()

	_, err := db.CreateCollection("test", 64)
	if err != nil {
		t.Fatalf("first create failed: %v", err)
	}

	_, err = db.CreateCollection("test", 64)
	if err == nil {
		t.Error("expected error when creating duplicate collection")
	}
}

func TestDBGetCollection(t *testing.T) {
	db, _ := Open(nil)
	defer db.Close()

	db.CreateCollection("my-collection", 256)

	coll, err := db.GetCollection("my-collection")
	if err != nil {
		t.Fatalf("get failed: %v", err)
	}

	if coll.Dimension() != 256 {
		t.Errorf("expected dimension 256, got %d", coll.Dimension())
	}
}

func TestDBGetCollectionNotFound(t *testing.T) {
	db, _ := Open(nil)
	defer db.Close()

	_, err := db.GetCollection("nonexistent")
	if err == nil {
		t.Error("expected error for nonexistent collection")
	}
}

func TestDBDeleteCollection(t *testing.T) {
	db, _ := Open(nil)
	defer db.Close()

	db.CreateCollection("to-delete", 64)
	
	if err := db.DeleteCollection("to-delete"); err != nil {
		t.Fatalf("delete failed: %v", err)
	}

	_, err := db.GetCollection("to-delete")
	if err == nil {
		t.Error("expected error after deletion")
	}
}

func TestDBInsertAndSearch(t *testing.T) {
	db, _ := Open(nil)
	defer db.Close()

	db.CreateCollection("vectors", 64, WithMetric(distance.Cosine))

	// Insert vectors
	for i := 0; i < 100; i++ {
		values := make([]float32, 64)
		for j := range values {
			values[j] = rand.Float32()
		}
		vec := core.NewVector(fmt.Sprintf("vec-%d", i), values)
		if err := db.Insert("vectors", vec); err != nil {
			t.Fatalf("insert failed: %v", err)
		}
	}

	// Search
	query := make([]float32, 64)
	for i := range query {
		query[i] = rand.Float32()
	}

	results, err := db.Search("vectors", query, 10)
	if err != nil {
		t.Fatalf("search failed: %v", err)
	}

	if len(results) != 10 {
		t.Errorf("expected 10 results, got %d", len(results))
	}
}

func TestDBSearchWithFilter(t *testing.T) {
	db, _ := Open(nil)
	defer db.Close()

	db.CreateCollection("filtered", 32)

	// Insert vectors with categories
	for i := 0; i < 50; i++ {
		values := make([]float32, 32)
		for j := range values {
			values[j] = rand.Float32()
		}
		category := "a"
		if i%2 == 0 {
			category = "b"
		}
		vec := core.NewVectorWithMetadata(fmt.Sprintf("vec-%d", i), values, map[string]any{"category": category})
		db.Insert("filtered", vec)
	}

	// Search with filter
	query := make([]float32, 32)
	for i := range query {
		query[i] = rand.Float32()
	}

	f := filter.NewFilter("category", "a")
	results, err := db.SearchWithFilter("filtered", query, 5, f)
	if err != nil {
		t.Fatalf("filtered search failed: %v", err)
	}

	// All results should have category "a"
	for _, r := range results {
		if r.Metadata["category"] != "a" {
			t.Errorf("result has wrong category: %v", r.Metadata)
		}
	}
}

func TestDBCollectionOptions(t *testing.T) {
	db, _ := Open(nil)
	defer db.Close()

	coll, err := db.CreateCollection(
		"custom",
		256,
		WithMetric(distance.Euclidean),
		WithHNSWParams(32, 400),
	)
	if err != nil {
		t.Fatalf("create failed: %v", err)
	}

	cfg := coll.Config()
	if cfg.Metric != distance.Euclidean {
		t.Errorf("expected Euclidean metric")
	}
	if cfg.M != 32 {
		t.Errorf("expected M=32, got %d", cfg.M)
	}
	if cfg.EfConstruction != 400 {
		t.Errorf("expected EfConstruction=400, got %d", cfg.EfConstruction)
	}
}

func TestDBPersistence(t *testing.T) {
	// Create temp directory
	tmpDir := filepath.Join(os.TempDir(), fmt.Sprintf("deadlock-test-%d", rand.Int()))
	defer os.RemoveAll(tmpDir)

	// Create DB with persistence
	db1, err := Open(&Options{DataDir: tmpDir})
	if err != nil {
		t.Fatalf("failed to open db: %v", err)
	}

	// Create collection and insert data
	db1.CreateCollection("persistent", 32)
	for i := 0; i < 10; i++ {
		values := make([]float32, 32)
		for j := range values {
			values[j] = float32(i * j)
		}
		db1.Insert("persistent", core.NewVector(fmt.Sprintf("vec-%d", i), values))
	}
	db1.Close()

	// Reopen and verify data persisted
	db2, err := Open(&Options{DataDir: tmpDir})
	if err != nil {
		t.Fatalf("failed to reopen db: %v", err)
	}
	defer db2.Close()

	coll, err := db2.GetCollection("persistent")
	if err != nil {
		t.Fatalf("collection not found: %v", err)
	}

	if coll.Count() != 10 {
		t.Errorf("expected 10 vectors, got %d", coll.Count())
	}

	// Verify a specific vector
	vec, err := db2.Get("persistent", "vec-5")
	if err != nil {
		t.Fatalf("get failed: %v", err)
	}
	if vec.Values[1] != 5 { // 5*1 = 5
		t.Errorf("expected values[1]=5, got %f", vec.Values[1])
	}
}

func TestVersion(t *testing.T) {
	v := Version()
	if v == "" {
		t.Error("version should not be empty")
	}
}

func BenchmarkDBInsert(b *testing.B) {
	db, _ := Open(nil)
	defer db.Close()

	db.CreateCollection("bench", 128)

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
		db.Insert("bench", vectors[i])
	}
}

func BenchmarkDBSearch(b *testing.B) {
	db, _ := Open(nil)
	defer db.Close()

	db.CreateCollection("bench", 128)

	// Insert 10000 vectors
	for i := 0; i < 10000; i++ {
		values := make([]float32, 128)
		for j := range values {
			values[j] = rand.Float32()
		}
		db.Insert("bench", core.NewVector(fmt.Sprintf("vec-%d", i), values))
	}

	query := make([]float32, 128)
	for i := range query {
		query[i] = rand.Float32()
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		db.Search("bench", query, 10)
	}
}
