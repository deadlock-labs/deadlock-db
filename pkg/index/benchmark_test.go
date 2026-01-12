package index

import (
	"fmt"
	"math/rand"
	"sort"
	"testing"

	"github.com/deadlock-labs/deadlock-db/pkg/core"
	"github.com/deadlock-labs/deadlock-db/pkg/distance"
)

// BruteForceSearch performs exact k-NN search for ground truth comparison.
func BruteForceSearch(query []float32, vectors [][]float32, ids []string, k int, calc distance.Calculator) []core.SearchResult {
	type result struct {
		id   string
		dist float32
	}
	results := make([]result, len(vectors))
	for i, v := range vectors {
		results[i] = result{ids[i], calc.Distance(query, v)}
	}
	sort.Slice(results, func(i, j int) bool {
		return results[i].dist < results[j].dist
	})

	out := make([]core.SearchResult, 0, k)
	for i := 0; i < k && i < len(results); i++ {
		out = append(out, core.SearchResult{ID: results[i].id, Score: results[i].dist})
	}
	return out
}

func TestHNSWRecallComparison(t *testing.T) {
	// Test parameters
	dim := 128
	numVectors := 5000
	numQueries := 100
	k := 10
	ef := 100

	// Generate dataset
	rand.Seed(42)
	vectors := make([][]float32, numVectors)
	ids := make([]string, numVectors)
	for i := 0; i < numVectors; i++ {
		vectors[i] = make([]float32, dim)
		for j := 0; j < dim; j++ {
			vectors[i][j] = rand.Float32()*2 - 1
		}
		ids[i] = fmt.Sprintf("vec-%d", i)
	}

	// Build HNSW index with heuristic
	configHeuristic := DefaultHNSWConfig(dim)
	configHeuristic.UseHeuristic = true
	idxHeuristic := NewHNSW(configHeuristic)

	// Build HNSW index without heuristic
	configSimple := DefaultHNSWConfig(dim)
	configSimple.UseHeuristic = false
	idxSimple := NewHNSW(configSimple)

	// Insert into both indexes
	for i := 0; i < numVectors; i++ {
		vec := core.NewVector(ids[i], vectors[i])
		idxHeuristic.Insert(vec)
		idxSimple.Insert(vec)
	}

	// Calculate recall for both configurations
	calc := distance.NewCalculator(distance.Cosine)
	var totalRecallHeuristic, totalRecallSimple float64

	for q := 0; q < numQueries; q++ {
		query := make([]float32, dim)
		for j := 0; j < dim; j++ {
			query[j] = rand.Float32()*2 - 1
		}

		// Ground truth using brute force
		groundTruth := BruteForceSearch(query, vectors, ids, k, calc)
		gtIDs := make(map[string]bool)
		for _, r := range groundTruth {
			gtIDs[r.ID] = true
		}

		// HNSW with heuristic
		resultsHeuristic := idxHeuristic.Search(query, k, ef)
		hitsHeuristic := 0
		for _, r := range resultsHeuristic {
			if gtIDs[r.ID] {
				hitsHeuristic++
			}
		}
		totalRecallHeuristic += float64(hitsHeuristic) / float64(k)

		// HNSW without heuristic
		resultsSimple := idxSimple.Search(query, k, ef)
		hitsSimple := 0
		for _, r := range resultsSimple {
			if gtIDs[r.ID] {
				hitsSimple++
			}
		}
		totalRecallSimple += float64(hitsSimple) / float64(k)
	}

	avgRecallHeuristic := totalRecallHeuristic / float64(numQueries)
	avgRecallSimple := totalRecallSimple / float64(numQueries)

	t.Logf("HNSW with RNG Heuristic - Avg Recall@%d: %.4f", k, avgRecallHeuristic)
	t.Logf("HNSW Simple Selection  - Avg Recall@%d: %.4f", k, avgRecallSimple)

	// Both should have reasonable recall
	if avgRecallHeuristic < 0.7 {
		t.Errorf("Heuristic recall too low: %.4f (expected > 0.7)", avgRecallHeuristic)
	}
	if avgRecallSimple < 0.6 {
		t.Errorf("Simple recall too low: %.4f (expected > 0.6)", avgRecallSimple)
	}
}

func TestHNSWAdaptiveSearch(t *testing.T) {
	dim := 64
	numVectors := 1000
	k := 10

	// Generate dataset
	rand.Seed(123)
	config := DefaultHNSWConfig(dim)
	idx := NewHNSW(config)

	for i := 0; i < numVectors; i++ {
		v := make([]float32, dim)
		for j := 0; j < dim; j++ {
			v[j] = rand.Float32()*2 - 1
		}
		vec := core.NewVector(fmt.Sprintf("vec-%d", i), v)
		idx.Insert(vec)
	}

	// Test adaptive search
	query := make([]float32, dim)
	for j := 0; j < dim; j++ {
		query[j] = rand.Float32()*2 - 1
	}

	results := idx.SearchAdaptive(query, k, 0.9)
	if len(results) != k {
		t.Errorf("Expected %d results, got %d", k, len(results))
	}

	// Results should be sorted by score
	for i := 1; i < len(results); i++ {
		if results[i].Score < results[i-1].Score {
			t.Error("Results not sorted by score")
		}
	}
}

func TestHNSWBatchInsert(t *testing.T) {
	dim := 32
	numVectors := 500

	config := DefaultHNSWConfig(dim)
	idx := NewHNSW(config)

	// Create vectors
	vectors := make([]*core.Vector, numVectors)
	for i := 0; i < numVectors; i++ {
		v := make([]float32, dim)
		for j := 0; j < dim; j++ {
			v[j] = rand.Float32()
		}
		vectors[i] = core.NewVector(fmt.Sprintf("vec-%d", i), v)
	}

	// Batch insert
	err := idx.InsertBatch(vectors)
	if err != nil {
		t.Fatalf("Batch insert failed: %v", err)
	}

	if idx.Size() != numVectors {
		t.Errorf("Expected size %d, got %d", numVectors, idx.Size())
	}

	// Verify search still works
	query := vectors[0].Values
	results := idx.Search(query, 10, 50)
	if len(results) == 0 {
		t.Error("Search returned no results after batch insert")
	}
	if results[0].ID != "vec-0" {
		t.Errorf("Expected first result to be vec-0, got %s", results[0].ID)
	}
}

func TestHNSWSearchWithFilterFunc(t *testing.T) {
	dim := 32
	numVectors := 200

	config := DefaultHNSWConfig(dim)
	idx := NewHNSW(config)

	// Insert vectors with IDs that encode a "category"
	for i := 0; i < numVectors; i++ {
		v := make([]float32, dim)
		for j := 0; j < dim; j++ {
			v[j] = rand.Float32()
		}
		// Even IDs are category A, odd are category B
		vec := core.NewVector(fmt.Sprintf("vec-%d", i), v)
		idx.Insert(vec)
	}

	query := make([]float32, dim)
	for j := 0; j < dim; j++ {
		query[j] = rand.Float32()
	}

	// Search with filter for only even IDs (category A)
	filterFunc := func(id string) bool {
		var num int
		fmt.Sscanf(id, "vec-%d", &num)
		return num%2 == 0
	}

	results := idx.SearchWithFilter(query, 10, 100, filterFunc)
	for _, r := range results {
		var num int
		fmt.Sscanf(r.ID, "vec-%d", &num)
		if num%2 != 0 {
			t.Errorf("Filter not applied correctly, got odd ID: %s", r.ID)
		}
	}
}

func BenchmarkHNSWInsertWithHeuristic(b *testing.B) {
	dim := 128
	config := DefaultHNSWConfig(dim)
	config.UseHeuristic = true
	idx := NewHNSW(config)

	vectors := generateRandomVectors(b.N, dim)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		vec := core.NewVector(fmt.Sprintf("vec-%d", i), vectors[i])
		idx.Insert(vec)
	}
}

func BenchmarkHNSWInsertSimple(b *testing.B) {
	dim := 128
	config := DefaultHNSWConfig(dim)
	config.UseHeuristic = false
	idx := NewHNSW(config)

	vectors := generateRandomVectors(b.N, dim)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		vec := core.NewVector(fmt.Sprintf("vec-%d", i), vectors[i])
		idx.Insert(vec)
	}
}

func BenchmarkHNSWSearchWithHeuristic(b *testing.B) {
	dim := 128
	numVectors := 10000
	config := DefaultHNSWConfig(dim)
	config.UseHeuristic = true
	idx := NewHNSW(config)

	vectors := generateRandomVectors(numVectors, dim)
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

func BenchmarkHNSWSearchSimple(b *testing.B) {
	dim := 128
	numVectors := 10000
	config := DefaultHNSWConfig(dim)
	config.UseHeuristic = false
	idx := NewHNSW(config)

	vectors := generateRandomVectors(numVectors, dim)
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

func BenchmarkHNSWBatchInsert(b *testing.B) {
	dim := 128
	batchSize := 1000

	for i := 0; i < b.N; i++ {
		config := DefaultHNSWConfig(dim)
		idx := NewHNSW(config)

		vectors := make([]*core.Vector, batchSize)
		for j := 0; j < batchSize; j++ {
			v := make([]float32, dim)
			for k := 0; k < dim; k++ {
				v[k] = rand.Float32()
			}
			vectors[j] = core.NewVector(fmt.Sprintf("vec-%d", j), v)
		}

		idx.InsertBatch(vectors)
	}
}
