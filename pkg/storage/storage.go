// Package storage provides persistent storage for the vector database.
package storage

import (
	"bufio"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/deadlock-labs/deadlock-db/pkg/core"
)

// Engine is the interface for storage backends.
type Engine interface {
	// Put stores a vector.
	Put(v *core.Vector) error
	// Get retrieves a vector by ID.
	Get(id string) (*core.Vector, error)
	// Delete removes a vector by ID.
	Delete(id string) error
	// Iterate iterates over all vectors.
	Iterate(fn func(*core.Vector) error) error
	// Close closes the storage engine.
	Close() error
	// Sync forces data to disk.
	Sync() error
}

// MemoryEngine provides in-memory storage.
type MemoryEngine struct {
	data map[string]*core.Vector
	mu   sync.RWMutex
}

// NewMemoryEngine creates a new in-memory storage engine.
func NewMemoryEngine() *MemoryEngine {
	return &MemoryEngine{
		data: make(map[string]*core.Vector),
	}
}

// Put stores a vector.
func (m *MemoryEngine) Put(v *core.Vector) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.data[v.ID] = v.Clone()
	return nil
}

// Get retrieves a vector by ID.
func (m *MemoryEngine) Get(id string) (*core.Vector, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if v, ok := m.data[id]; ok {
		return v.Clone(), nil
	}
	return nil, fmt.Errorf("vector not found: %s", id)
}

// Delete removes a vector by ID.
func (m *MemoryEngine) Delete(id string) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	delete(m.data, id)
	return nil
}

// Iterate iterates over all vectors.
func (m *MemoryEngine) Iterate(fn func(*core.Vector) error) error {
	m.mu.RLock()
	defer m.mu.RUnlock()
	for _, v := range m.data {
		if err := fn(v); err != nil {
			return err
		}
	}
	return nil
}

// Close closes the storage engine.
func (m *MemoryEngine) Close() error {
	return nil
}

// Sync forces data to disk (no-op for memory engine).
func (m *MemoryEngine) Sync() error {
	return nil
}

// WALOperation represents a write-ahead log operation.
type WALOperation uint8

const (
	WALPut WALOperation = iota
	WALDelete
)

// WALEntry represents a single entry in the write-ahead log.
type WALEntry struct {
	Op        WALOperation `json:"op"`
	VectorID  string       `json:"id"`
	Values    []float32    `json:"values,omitempty"`
	Metadata  map[string]any `json:"metadata,omitempty"`
	Timestamp int64        `json:"ts"`
}

// WAL implements a write-ahead log for durability.
type WAL struct {
	file       *os.File
	writer     *bufio.Writer
	mu         sync.Mutex
	path       string
	syncPeriod time.Duration
	done       chan struct{}
}

// NewWAL creates a new write-ahead log.
func NewWAL(path string) (*WAL, error) {
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return nil, fmt.Errorf("failed to create WAL directory: %w", err)
	}

	file, err := os.OpenFile(path, os.O_APPEND|os.O_CREATE|os.O_RDWR, 0644)
	if err != nil {
		return nil, fmt.Errorf("failed to open WAL file: %w", err)
	}

	wal := &WAL{
		file:       file,
		writer:     bufio.NewWriter(file),
		path:       path,
		syncPeriod: time.Second,
		done:       make(chan struct{}),
	}

	// Start background sync
	go wal.backgroundSync()

	return wal, nil
}

// Append adds an entry to the WAL.
func (w *WAL) Append(entry *WALEntry) error {
	w.mu.Lock()
	defer w.mu.Unlock()

	entry.Timestamp = time.Now().UnixNano()

	data, err := json.Marshal(entry)
	if err != nil {
		return fmt.Errorf("failed to marshal WAL entry: %w", err)
	}

	// Write length prefix
	lenBuf := make([]byte, 4)
	binary.LittleEndian.PutUint32(lenBuf, uint32(len(data)))
	if _, err := w.writer.Write(lenBuf); err != nil {
		return fmt.Errorf("failed to write WAL entry length: %w", err)
	}

	// Write data
	if _, err := w.writer.Write(data); err != nil {
		return fmt.Errorf("failed to write WAL entry: %w", err)
	}

	return nil
}

// Sync flushes the WAL to disk.
func (w *WAL) Sync() error {
	w.mu.Lock()
	defer w.mu.Unlock()

	if err := w.writer.Flush(); err != nil {
		return fmt.Errorf("failed to flush WAL: %w", err)
	}
	if err := w.file.Sync(); err != nil {
		return fmt.Errorf("failed to sync WAL: %w", err)
	}
	return nil
}

// backgroundSync periodically syncs the WAL.
func (w *WAL) backgroundSync() {
	ticker := time.NewTicker(w.syncPeriod)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			_ = w.Sync()
		case <-w.done:
			return
		}
	}
}

// Recover reads all entries from the WAL.
func (w *WAL) Recover() ([]*WALEntry, error) {
	w.mu.Lock()
	defer w.mu.Unlock()

	// Seek to beginning
	if _, err := w.file.Seek(0, 0); err != nil {
		return nil, fmt.Errorf("failed to seek WAL: %w", err)
	}

	var entries []*WALEntry
	reader := bufio.NewReader(w.file)

	for {
		// Read length prefix
		lenBuf := make([]byte, 4)
		if _, err := io.ReadFull(reader, lenBuf); err != nil {
			if err == io.EOF {
				break
			}
			return entries, nil // Return what we have on partial read
		}

		length := binary.LittleEndian.Uint32(lenBuf)
		if length > 100*1024*1024 { // 100MB sanity check
			break
		}

		// Read data
		data := make([]byte, length)
		if _, err := io.ReadFull(reader, data); err != nil {
			return entries, nil // Return what we have on partial read
		}

		var entry WALEntry
		if err := json.Unmarshal(data, &entry); err != nil {
			continue // Skip corrupted entries
		}

		entries = append(entries, &entry)
	}

	return entries, nil
}

// Truncate removes all entries from the WAL.
func (w *WAL) Truncate() error {
	w.mu.Lock()
	defer w.mu.Unlock()

	if err := w.writer.Flush(); err != nil {
		return err
	}
	if err := w.file.Truncate(0); err != nil {
		return err
	}
	if _, err := w.file.Seek(0, 0); err != nil {
		return err
	}
	w.writer.Reset(w.file)
	return nil
}

// Close closes the WAL.
func (w *WAL) Close() error {
	close(w.done)
	if err := w.Sync(); err != nil {
		return err
	}
	return w.file.Close()
}

// DiskEngine provides persistent disk storage.
type DiskEngine struct {
	dataDir string
	wal     *WAL
	memory  *MemoryEngine
	mu      sync.RWMutex
}

// NewDiskEngine creates a new disk storage engine.
func NewDiskEngine(dataDir string) (*DiskEngine, error) {
	if err := os.MkdirAll(dataDir, 0755); err != nil {
		return nil, fmt.Errorf("failed to create data directory: %w", err)
	}

	walPath := filepath.Join(dataDir, "wal.log")
	wal, err := NewWAL(walPath)
	if err != nil {
		return nil, err
	}

	engine := &DiskEngine{
		dataDir: dataDir,
		wal:     wal,
		memory:  NewMemoryEngine(),
	}

	// Recover from WAL
	if err := engine.recover(); err != nil {
		return nil, err
	}

	return engine, nil
}

// recover replays the WAL to restore state.
func (d *DiskEngine) recover() error {
	entries, err := d.wal.Recover()
	if err != nil {
		return err
	}

	for _, entry := range entries {
		switch entry.Op {
		case WALPut:
			v := &core.Vector{
				ID:       entry.VectorID,
				Values:   entry.Values,
				Metadata: entry.Metadata,
			}
			_ = d.memory.Put(v)
		case WALDelete:
			_ = d.memory.Delete(entry.VectorID)
		}
	}

	return nil
}

// Put stores a vector.
func (d *DiskEngine) Put(v *core.Vector) error {
	d.mu.Lock()
	defer d.mu.Unlock()

	// Write to WAL first
	entry := &WALEntry{
		Op:       WALPut,
		VectorID: v.ID,
		Values:   v.Values,
		Metadata: v.Metadata,
	}
	if err := d.wal.Append(entry); err != nil {
		return err
	}

	// Then update memory
	return d.memory.Put(v)
}

// Get retrieves a vector by ID.
func (d *DiskEngine) Get(id string) (*core.Vector, error) {
	d.mu.RLock()
	defer d.mu.RUnlock()
	return d.memory.Get(id)
}

// Delete removes a vector by ID.
func (d *DiskEngine) Delete(id string) error {
	d.mu.Lock()
	defer d.mu.Unlock()

	// Write to WAL first
	entry := &WALEntry{
		Op:       WALDelete,
		VectorID: id,
	}
	if err := d.wal.Append(entry); err != nil {
		return err
	}

	// Then update memory
	return d.memory.Delete(id)
}

// Iterate iterates over all vectors.
func (d *DiskEngine) Iterate(fn func(*core.Vector) error) error {
	d.mu.RLock()
	defer d.mu.RUnlock()
	return d.memory.Iterate(fn)
}

// Sync forces data to disk.
func (d *DiskEngine) Sync() error {
	return d.wal.Sync()
}

// Close closes the storage engine.
func (d *DiskEngine) Close() error {
	return d.wal.Close()
}

// Compact compacts the storage by rewriting all data.
func (d *DiskEngine) Compact() error {
	d.mu.Lock()
	defer d.mu.Unlock()

	// Create new WAL
	newWalPath := filepath.Join(d.dataDir, "wal.new")
	newWal, err := NewWAL(newWalPath)
	if err != nil {
		return err
	}

	// Write all current data to new WAL
	err = d.memory.Iterate(func(v *core.Vector) error {
		entry := &WALEntry{
			Op:       WALPut,
			VectorID: v.ID,
			Values:   v.Values,
			Metadata: v.Metadata,
		}
		return newWal.Append(entry)
	})
	if err != nil {
		newWal.Close()
		os.Remove(newWalPath)
		return err
	}

	// Sync and close new WAL
	if err := newWal.Sync(); err != nil {
		newWal.Close()
		os.Remove(newWalPath)
		return err
	}
	if err := newWal.Close(); err != nil {
		os.Remove(newWalPath)
		return err
	}

	// Close old WAL
	if err := d.wal.Close(); err != nil {
		return err
	}

	// Replace old WAL with new
	oldWalPath := filepath.Join(d.dataDir, "wal.log")
	if err := os.Rename(newWalPath, oldWalPath); err != nil {
		return err
	}

	// Reopen WAL
	d.wal, err = NewWAL(oldWalPath)
	return err
}
