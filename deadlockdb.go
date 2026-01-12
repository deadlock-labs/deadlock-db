// Package deadlockdb provides the main database interface for the deadlock-db vector database.
package deadlockdb

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sync"

	"github.com/deadlock-labs/deadlock-db/pkg/collection"
	"github.com/deadlock-labs/deadlock-db/pkg/core"
	"github.com/deadlock-labs/deadlock-db/pkg/distance"
	"github.com/deadlock-labs/deadlock-db/pkg/filter"
)

// DB represents the deadlock-db vector database.
type DB struct {
	dataDir     string
	collections map[string]*collection.Collection
	mu          sync.RWMutex
}

// Options configures the database.
type Options struct {
	DataDir string // Directory for persistent storage (empty for in-memory only)
}

// DefaultOptions returns the default database options.
func DefaultOptions() *Options {
	return &Options{
		DataDir: "",
	}
}

// Open opens or creates a database.
func Open(opts *Options) (*DB, error) {
	if opts == nil {
		opts = DefaultOptions()
	}

	db := &DB{
		dataDir:     opts.DataDir,
		collections: make(map[string]*collection.Collection),
	}

	// Load existing collections if using disk storage
	if opts.DataDir != "" {
		if err := os.MkdirAll(opts.DataDir, 0755); err != nil {
			return nil, fmt.Errorf("failed to create data directory: %w", err)
		}
		if err := db.loadCollections(); err != nil {
			return nil, err
		}
	}

	return db, nil
}

// loadCollections loads existing collections from disk.
func (db *DB) loadCollections() error {
	if db.dataDir == "" {
		return nil
	}

	entries, err := os.ReadDir(db.dataDir)
	if err != nil {
		return nil
	}

	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}

		configPath := filepath.Join(db.dataDir, entry.Name(), "config.json")
		if _, err := os.Stat(configPath); os.IsNotExist(err) {
			continue
		}

		data, err := os.ReadFile(configPath)
		if err != nil {
			continue
		}

		var cfg collection.Config
		if err := json.Unmarshal(data, &cfg); err != nil {
			continue
		}

		cfg.DataDir = db.dataDir
		cfg.StorageType = "disk"

		coll, err := collection.New(&cfg)
		if err != nil {
			continue
		}

		db.collections[cfg.Name] = coll
	}

	return nil
}

// CreateCollection creates a new collection.
func (db *DB) CreateCollection(name string, dimension int, opts ...CollectionOption) (*collection.Collection, error) {
	db.mu.Lock()
	defer db.mu.Unlock()

	if _, exists := db.collections[name]; exists {
		return nil, fmt.Errorf("collection %s already exists", name)
	}

	cfg := collection.DefaultConfig(name, dimension)
	for _, opt := range opts {
		opt(cfg)
	}

	// Use disk storage if dataDir is set
	if db.dataDir != "" {
		cfg.DataDir = db.dataDir
		cfg.StorageType = "disk"
	}

	coll, err := collection.New(cfg)
	if err != nil {
		return nil, err
	}

	// Save config for disk storage
	if db.dataDir != "" {
		if err := db.saveCollectionConfig(cfg); err != nil {
			return nil, err
		}
	}

	db.collections[name] = coll
	return coll, nil
}

// saveCollectionConfig saves the collection configuration to disk.
func (db *DB) saveCollectionConfig(cfg *collection.Config) error {
	collDir := filepath.Join(db.dataDir, cfg.Name)
	if err := os.MkdirAll(collDir, 0755); err != nil {
		return err
	}

	configPath := filepath.Join(collDir, "config.json")
	data, err := json.MarshalIndent(cfg, "", "  ")
	if err != nil {
		return err
	}

	return os.WriteFile(configPath, data, 0644)
}

// GetCollection retrieves a collection by name.
func (db *DB) GetCollection(name string) (*collection.Collection, error) {
	db.mu.RLock()
	defer db.mu.RUnlock()

	coll, exists := db.collections[name]
	if !exists {
		return nil, fmt.Errorf("collection %s not found", name)
	}
	return coll, nil
}

// DeleteCollection deletes a collection.
func (db *DB) DeleteCollection(name string) error {
	db.mu.Lock()
	defer db.mu.Unlock()

	coll, exists := db.collections[name]
	if !exists {
		return fmt.Errorf("collection %s not found", name)
	}

	if err := coll.Close(); err != nil {
		return err
	}

	delete(db.collections, name)

	// Remove from disk if using disk storage
	if db.dataDir != "" {
		collDir := filepath.Join(db.dataDir, name)
		if err := os.RemoveAll(collDir); err != nil {
			return err
		}
	}

	return nil
}

// ListCollections returns the names of all collections.
func (db *DB) ListCollections() []string {
	db.mu.RLock()
	defer db.mu.RUnlock()

	names := make([]string, 0, len(db.collections))
	for name := range db.collections {
		names = append(names, name)
	}
	return names
}

// Insert inserts a vector into a collection.
func (db *DB) Insert(collectionName string, v *core.Vector) error {
	coll, err := db.GetCollection(collectionName)
	if err != nil {
		return err
	}
	return coll.Insert(v)
}

// InsertBatch inserts multiple vectors into a collection.
func (db *DB) InsertBatch(collectionName string, vectors []*core.Vector) error {
	coll, err := db.GetCollection(collectionName)
	if err != nil {
		return err
	}
	return coll.InsertBatch(vectors)
}

// Get retrieves a vector from a collection.
func (db *DB) Get(collectionName string, id string) (*core.Vector, error) {
	coll, err := db.GetCollection(collectionName)
	if err != nil {
		return nil, err
	}
	return coll.Get(id)
}

// Delete removes a vector from a collection.
func (db *DB) Delete(collectionName string, id string) error {
	coll, err := db.GetCollection(collectionName)
	if err != nil {
		return err
	}
	return coll.Delete(id)
}

// Search performs a similarity search in a collection.
func (db *DB) Search(collectionName string, query []float32, k int) ([]core.SearchResult, error) {
	coll, err := db.GetCollection(collectionName)
	if err != nil {
		return nil, err
	}
	return coll.Search(query, collection.DefaultSearchOptions(k))
}

// SearchWithFilter performs a filtered similarity search.
func (db *DB) SearchWithFilter(collectionName string, query []float32, k int, f *filter.Filter) ([]core.SearchResult, error) {
	coll, err := db.GetCollection(collectionName)
	if err != nil {
		return nil, err
	}
	return coll.SearchWithFilter(query, k, f)
}

// Close closes the database and all collections.
func (db *DB) Close() error {
	db.mu.Lock()
	defer db.mu.Unlock()

	for _, coll := range db.collections {
		if err := coll.Close(); err != nil {
			return err
		}
	}

	db.collections = make(map[string]*collection.Collection)
	return nil
}

// CollectionOption is a function that configures a collection.
type CollectionOption func(*collection.Config)

// WithMetric sets the distance metric for the collection.
func WithMetric(metric distance.Metric) CollectionOption {
	return func(cfg *collection.Config) {
		cfg.Metric = metric
	}
}

// WithHNSWParams sets the HNSW index parameters.
func WithHNSWParams(m, efConstruction int) CollectionOption {
	return func(cfg *collection.Config) {
		cfg.M = m
		cfg.EfConstruction = efConstruction
	}
}

// WithQuantization sets the quantization type.
func WithQuantization(quantType string) CollectionOption {
	return func(cfg *collection.Config) {
		cfg.Quantization = quantType
	}
}

// Version returns the deadlock-db version.
func Version() string {
	return "0.1.0"
}
