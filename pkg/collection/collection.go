// Package collection provides the main collection type for managing vectors.
package collection

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"sync"
	"sync/atomic"
	"time"

	"github.com/deadlock-labs/deadlock-db/pkg/core"
	"github.com/deadlock-labs/deadlock-db/pkg/distance"
	"github.com/deadlock-labs/deadlock-db/pkg/filter"
	"github.com/deadlock-labs/deadlock-db/pkg/index"
	"github.com/deadlock-labs/deadlock-db/pkg/quantization"
	"github.com/deadlock-labs/deadlock-db/pkg/storage"
)

// Config holds the configuration for a collection.
type Config struct {
	Name           string          `json:"name"`
	Dimension      int             `json:"dimension"`
	Metric         distance.Metric `json:"metric"`
	IndexType      string          `json:"index_type"`
	StorageType    string          `json:"storage_type"`
	DataDir        string          `json:"data_dir,omitempty"`
	M              int             `json:"m,omitempty"`
	EfConstruction int             `json:"ef_construction,omitempty"`
	Quantization   string          `json:"quantization,omitempty"`
}

// DefaultConfig returns a default configuration.
func DefaultConfig(name string, dimension int) *Config {
	return &Config{
		Name:           name,
		Dimension:      dimension,
		Metric:         distance.Cosine,
		IndexType:      "hnsw",
		StorageType:    "memory",
		M:              16,
		EfConstruction: 200,
	}
}

// Collection represents a collection of vectors.
type Collection struct {
	config    *Config
	index     *index.HNSW
	storage   storage.Engine
	metadata  map[string]map[string]any // vectorID -> metadata
	quantizer quantization.Quantizer
	count     uint64
	mu        sync.RWMutex
	created   time.Time
	updated   time.Time
}

// New creates a new collection with the given configuration.
func New(config *Config) (*Collection, error) {
	if config == nil {
		return nil, fmt.Errorf("config is required")
	}
	if config.Name == "" {
		return nil, fmt.Errorf("collection name is required")
	}
	if config.Dimension <= 0 {
		return nil, fmt.Errorf("dimension must be positive")
	}

	// Create HNSW index
	hnswConfig := index.DefaultHNSWConfig(config.Dimension)
	hnswConfig.Metric = config.Metric
	if config.M > 0 {
		hnswConfig.M = config.M
	}
	if config.EfConstruction > 0 {
		hnswConfig.EfConstruction = config.EfConstruction
	}
	idx := index.NewHNSW(hnswConfig)

	// Create storage engine
	var store storage.Engine
	var err error
	switch config.StorageType {
	case "disk":
		if config.DataDir == "" {
			return nil, fmt.Errorf("data_dir is required for disk storage")
		}
		dataPath := filepath.Join(config.DataDir, config.Name)
		store, err = storage.NewDiskEngine(dataPath)
		if err != nil {
			return nil, fmt.Errorf("failed to create disk storage: %w", err)
		}
	default:
		store = storage.NewMemoryEngine()
	}

	// Create quantizer if configured
	var q quantization.Quantizer
	switch config.Quantization {
	case "scalar", "sq8":
		q = quantization.NewScalarQuantizer(config.Dimension)
	case "pq", "product":
		numSubvecs := config.Dimension / 8
		if numSubvecs < 1 {
			numSubvecs = 1
		}
		q = quantization.NewProductQuantizer(config.Dimension, numSubvecs)
	case "binary":
		q = quantization.NewBinaryQuantizer(config.Dimension)
	}

	c := &Collection{
		config:    config,
		index:     idx,
		storage:   store,
		metadata:  make(map[string]map[string]any),
		quantizer: q,
		created:   time.Now(),
		updated:   time.Now(),
	}

	// Load existing data from storage
	if err := c.loadFromStorage(); err != nil {
		return nil, fmt.Errorf("failed to load from storage: %w", err)
	}

	return c, nil
}

// loadFromStorage loads vectors from the storage engine into the index.
func (c *Collection) loadFromStorage() error {
	return c.storage.Iterate(func(v *core.Vector) error {
		if err := c.index.Insert(v); err != nil {
			return err
		}
		if v.Metadata != nil {
			c.metadata[v.ID] = v.Metadata
		}
		atomic.AddUint64(&c.count, 1)
		return nil
	})
}

// Name returns the collection name.
func (c *Collection) Name() string {
	return c.config.Name
}

// Dimension returns the vector dimension.
func (c *Collection) Dimension() int {
	return c.config.Dimension
}

// Count returns the number of vectors.
func (c *Collection) Count() int {
	return int(atomic.LoadUint64(&c.count))
}

// Config returns the collection configuration.
func (c *Collection) Config() *Config {
	return c.config
}

// Insert adds a vector to the collection.
func (c *Collection) Insert(v *core.Vector) error {
	if v == nil {
		return fmt.Errorf("vector is nil")
	}
	if len(v.Values) != c.config.Dimension {
		return fmt.Errorf("dimension mismatch: expected %d, got %d", c.config.Dimension, len(v.Values))
	}

	c.mu.Lock()
	defer c.mu.Unlock()

	// Check if exists (for update)
	_, exists := c.index.GetVector(v.ID)

	// Store to disk
	if err := c.storage.Put(v); err != nil {
		return fmt.Errorf("failed to store vector: %w", err)
	}

	// Insert into index
	if err := c.index.Insert(v); err != nil {
		return fmt.Errorf("failed to index vector: %w", err)
	}

	// Store metadata
	if v.Metadata != nil {
		c.metadata[v.ID] = v.Metadata
	}

	if !exists {
		atomic.AddUint64(&c.count, 1)
	}
	c.updated = time.Now()

	return nil
}

// InsertBatch inserts multiple vectors efficiently.
// It validates all vectors first, then delegates to the HNSW batch inserter
// for better throughput on large batches.
func (c *Collection) InsertBatch(vectors []*core.Vector) error {
	if len(vectors) == 0 {
		return nil
	}

	// Validate all vectors first
	for _, v := range vectors {
		if v == nil {
			return fmt.Errorf("vector is nil")
		}
		if len(v.Values) != c.config.Dimension {
			return fmt.Errorf("dimension mismatch: expected %d, got %d", c.config.Dimension, len(v.Values))
		}
	}

	c.mu.Lock()
	defer c.mu.Unlock()

	// Store to storage and collect metadata
	for _, v := range vectors {
		if err := c.storage.Put(v); err != nil {
			return fmt.Errorf("failed to store vector: %w", err)
		}
		if v.Metadata != nil {
			c.metadata[v.ID] = v.Metadata
		}
	}

	// Batch insert into HNSW index
	if err := c.index.InsertBatch(vectors); err != nil {
		return fmt.Errorf("failed to index batch: %w", err)
	}

	atomic.AddUint64(&c.count, uint64(len(vectors)))
	c.updated = time.Now()

	return nil
}

// Get retrieves a vector by ID.
func (c *Collection) Get(id string) (*core.Vector, error) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	v, err := c.storage.Get(id)
	if err != nil {
		return nil, err
	}

	// Add metadata if available
	if meta, ok := c.metadata[id]; ok {
		v.Metadata = meta
	}

	return v, nil
}

// Delete removes a vector by ID.
func (c *Collection) Delete(id string) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	// Delete from storage
	if err := c.storage.Delete(id); err != nil {
		return err
	}

	// Delete from index
	if err := c.index.Delete(id); err != nil {
		return err
	}

	// Delete metadata
	delete(c.metadata, id)

	atomic.AddUint64(&c.count, ^uint64(0)) // Decrement
	c.updated = time.Now()

	return nil
}

// Update updates a vector's values and/or metadata.
func (c *Collection) Update(id string, values []float32, metadata map[string]any) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	// Get existing vector
	existing, err := c.storage.Get(id)
	if err != nil {
		return err
	}

	// Update values if provided
	if values != nil {
		if len(values) != c.config.Dimension {
			return fmt.Errorf("dimension mismatch")
		}
		existing.Values = values
	}

	// Update metadata if provided
	if metadata != nil {
		existing.Metadata = metadata
		c.metadata[id] = metadata
	}

	// Increment version on update
	existing.IncrementVersion()

	// Store updated vector
	if err := c.storage.Put(existing); err != nil {
		return err
	}

	// Update index if values changed
	if values != nil {
		if err := c.index.Insert(existing); err != nil {
			return err
		}
	}

	c.updated = time.Now()
	return nil
}

// SearchOptions configures a search operation.
type SearchOptions struct {
	K        int            // Number of results to return
	Ef       int            // Search expansion factor (higher = more accurate but slower)
	Filter   *filter.Filter // Metadata filter
	Include  []string       // Fields to include in results
	Rerank   bool           // Whether to re-rank results using exact distance
}

// DefaultSearchOptions returns default search options.
func DefaultSearchOptions(k int) *SearchOptions {
	return &SearchOptions{
		K:      k,
		Ef:     k * 10,
		Rerank: false,
	}
}

// Search finds the k nearest neighbors to the query vector.
func (c *Collection) Search(query []float32, opts *SearchOptions) ([]core.SearchResult, error) {
	if len(query) != c.config.Dimension {
		return nil, fmt.Errorf("dimension mismatch: expected %d, got %d", c.config.Dimension, len(query))
	}

	if opts == nil {
		opts = DefaultSearchOptions(10)
	}

	c.mu.RLock()
	defer c.mu.RUnlock()

	// Search the index
	ef := opts.Ef
	if ef < opts.K {
		ef = opts.K * 10
	}

	// If we have a filter, we need to search for more candidates
	searchK := opts.K
	if opts.Filter != nil {
		searchK = opts.K * 10 // Get more candidates for filtering
		if searchK > c.Count() {
			searchK = c.Count()
		}
	}

	results := c.index.Search(query, searchK, ef)

	// Apply metadata filter if provided
	if opts.Filter != nil {
		filtered := make([]core.SearchResult, 0, opts.K)
		for _, r := range results {
			meta := c.metadata[r.ID]
			if opts.Filter.Match(meta) {
				r.Metadata = meta
				filtered = append(filtered, r)
				if len(filtered) >= opts.K {
					break
				}
			}
		}
		results = filtered
	} else {
		// Add metadata to results
		for i := range results {
			if i >= opts.K {
				break
			}
			results[i].Metadata = c.metadata[results[i].ID]
		}
		if len(results) > opts.K {
			results = results[:opts.K]
		}
	}

	// Re-rank using exact distance if requested
	if opts.Rerank && len(results) > 0 {
		calc := distance.NewCalculator(c.config.Metric)
		for i := range results {
			v, err := c.storage.Get(results[i].ID)
			if err == nil {
				results[i].Score = calc.Distance(query, v.Values)
			}
		}
		// Sort by score
		sort.Slice(results, func(i, j int) bool {
			return results[i].Score < results[j].Score
		})
	}

	return results, nil
}

// SearchWithFilter performs a hybrid search with metadata filtering.
func (c *Collection) SearchWithFilter(query []float32, k int, f *filter.Filter) ([]core.SearchResult, error) {
	opts := DefaultSearchOptions(k)
	opts.Filter = f
	return c.Search(query, opts)
}

// Sync forces data to disk.
func (c *Collection) Sync() error {
	return c.storage.Sync()
}

// Close closes the collection.
func (c *Collection) Close() error {
	return c.storage.Close()
}

// Stats returns statistics about the collection.
func (c *Collection) Stats() map[string]any {
	c.mu.RLock()
	defer c.mu.RUnlock()

	indexStats := c.index.Stats()

	return map[string]any{
		"name":       c.config.Name,
		"dimension":  c.config.Dimension,
		"count":      c.count,
		"metric":     c.config.Metric.String(),
		"index_type": c.config.IndexType,
		"created":    c.created,
		"updated":    c.updated,
		"index":      indexStats,
	}
}

// GraphData returns the HNSW graph structure for visualization.
func (c *Collection) GraphData(maxNodes int) map[string]any {
	c.mu.RLock()
	defer c.mu.RUnlock()

	return c.index.GraphData(maxNodes)
}

// Export exports the collection to a file.
func (c *Collection) Export(path string) error {
	c.mu.RLock()
	defer c.mu.RUnlock()

	file, err := os.Create(path)
	if err != nil {
		return err
	}
	defer file.Close()

	encoder := json.NewEncoder(file)

	// Export config first
	if err := encoder.Encode(c.config); err != nil {
		return err
	}

	// Export vectors
	return c.storage.Iterate(func(v *core.Vector) error {
		return encoder.Encode(v)
	})
}

// Import imports vectors from a file into the collection.
func (c *Collection) Import(path string) error {
	file, err := os.Open(path)
	if err != nil {
		return err
	}
	defer file.Close()

	decoder := json.NewDecoder(file)

	// Skip config line
	var cfg Config
	if err := decoder.Decode(&cfg); err != nil {
		return err
	}

	// Import vectors
	for {
		var v core.Vector
		if err := decoder.Decode(&v); err != nil {
			break
		}
		if err := c.Insert(&v); err != nil {
			return err
		}
	}

	return nil
}
