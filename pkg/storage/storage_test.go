package storage

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"github.com/deadlock-labs/deadlock-db/pkg/core"
)

func TestMemoryEngine(t *testing.T) {
	engine := NewMemoryEngine()

	// Test Put and Get
	vec := core.NewVector("test-1", []float32{1, 2, 3, 4})
	if err := engine.Put(vec); err != nil {
		t.Fatalf("put failed: %v", err)
	}

	retrieved, err := engine.Get("test-1")
	if err != nil {
		t.Fatalf("get failed: %v", err)
	}

	if retrieved.ID != "test-1" {
		t.Errorf("expected ID 'test-1', got '%s'", retrieved.ID)
	}

	// Test Delete
	if err := engine.Delete("test-1"); err != nil {
		t.Fatalf("delete failed: %v", err)
	}

	_, err = engine.Get("test-1")
	if err == nil {
		t.Error("expected error after delete")
	}
}

func TestMemoryEngineIterate(t *testing.T) {
	engine := NewMemoryEngine()

	// Insert multiple vectors
	for i := 0; i < 10; i++ {
		vec := core.NewVector(string(rune('a'+i)), []float32{float32(i)})
		engine.Put(vec)
	}

	// Count iterations
	count := 0
	err := engine.Iterate(func(_ *core.Vector) error {
		count++
		return nil
	})
	if err != nil {
		t.Fatalf("iterate failed: %v", err)
	}

	if count != 10 {
		t.Errorf("expected 10 iterations, got %d", count)
	}
}

func TestWAL(t *testing.T) {
	tmpDir := t.TempDir()
	walPath := filepath.Join(tmpDir, "test.wal")

	// Create WAL and write entries
	wal, err := NewWAL(walPath)
	if err != nil {
		t.Fatalf("failed to create WAL: %v", err)
	}

	entries := []*WALEntry{
		{Op: WALPut, VectorID: "vec-1", Values: []float32{1, 2, 3}},
		{Op: WALPut, VectorID: "vec-2", Values: []float32{4, 5, 6}},
		{Op: WALDelete, VectorID: "vec-1"},
	}

	for _, entry := range entries {
		if err := wal.Append(entry); err != nil {
			t.Fatalf("append failed: %v", err)
		}
	}

	if err := wal.Sync(); err != nil {
		t.Fatalf("sync failed: %v", err)
	}
	wal.Close()

	// Reopen and recover
	wal2, err := NewWAL(walPath)
	if err != nil {
		t.Fatalf("failed to reopen WAL: %v", err)
	}
	defer wal2.Close()

	recovered, err := wal2.Recover()
	if err != nil {
		t.Fatalf("recover failed: %v", err)
	}

	if len(recovered) != 3 {
		t.Errorf("expected 3 entries, got %d", len(recovered))
	}

	// Verify entries
	if recovered[0].VectorID != "vec-1" {
		t.Errorf("first entry ID mismatch")
	}
	if recovered[2].Op != WALDelete {
		t.Errorf("third entry should be delete")
	}
}

func TestWALTruncate(t *testing.T) {
	tmpDir := t.TempDir()
	walPath := filepath.Join(tmpDir, "truncate.wal")

	wal, _ := NewWAL(walPath)

	// Write some entries
	for i := 0; i < 5; i++ {
		wal.Append(&WALEntry{Op: WALPut, VectorID: string(rune('a' + i))})
	}
	wal.Sync()

	// Truncate
	if err := wal.Truncate(); err != nil {
		t.Fatalf("truncate failed: %v", err)
	}

	// Verify empty
	recovered, _ := wal.Recover()
	if len(recovered) != 0 {
		t.Errorf("expected 0 entries after truncate, got %d", len(recovered))
	}

	wal.Close()
}

func TestDiskEngine(t *testing.T) {
	tmpDir := t.TempDir()

	engine, err := NewDiskEngine(tmpDir)
	if err != nil {
		t.Fatalf("failed to create disk engine: %v", err)
	}

	// Test Put and Get
	vec := core.NewVectorWithMetadata("test-vec", []float32{1, 2, 3, 4}, map[string]any{"key": "value"})
	if err := engine.Put(vec); err != nil {
		t.Fatalf("put failed: %v", err)
	}

	retrieved, err := engine.Get("test-vec")
	if err != nil {
		t.Fatalf("get failed: %v", err)
	}

	if retrieved.ID != "test-vec" {
		t.Errorf("ID mismatch")
	}

	engine.Close()

	// Reopen and verify persistence
	engine2, err := NewDiskEngine(tmpDir)
	if err != nil {
		t.Fatalf("failed to reopen: %v", err)
	}
	defer engine2.Close()

	retrieved2, err := engine2.Get("test-vec")
	if err != nil {
		t.Fatalf("get after reopen failed: %v", err)
	}

	if retrieved2.ID != "test-vec" {
		t.Errorf("ID mismatch after reopen")
	}
}

func TestDiskEngineDelete(t *testing.T) {
	tmpDir := t.TempDir()

	engine, _ := NewDiskEngine(tmpDir)

	// Insert and delete
	vec := core.NewVector("to-delete", []float32{1, 2, 3})
	engine.Put(vec)
	engine.Delete("to-delete")
	engine.Close()

	// Reopen and verify deletion persisted
	engine2, _ := NewDiskEngine(tmpDir)
	defer engine2.Close()

	_, err := engine2.Get("to-delete")
	if err == nil {
		t.Error("deleted vector should not be found after reopen")
	}
}

func TestDiskEngineCompact(t *testing.T) {
	tmpDir := t.TempDir()

	engine, _ := NewDiskEngine(tmpDir)

	// Insert many vectors then delete some
	for i := 0; i < 100; i++ {
		vec := core.NewVector(string(rune(i)), []float32{float32(i)})
		engine.Put(vec)
	}

	for i := 0; i < 50; i++ {
		engine.Delete(string(rune(i)))
	}

	// Compact
	if err := engine.Compact(); err != nil {
		t.Fatalf("compact failed: %v", err)
	}

	// Verify remaining vectors
	count := 0
	engine.Iterate(func(_ *core.Vector) error {
		count++
		return nil
	})

	if count != 50 {
		t.Errorf("expected 50 vectors after compact, got %d", count)
	}

	engine.Close()
}

func BenchmarkMemoryEnginePut(b *testing.B) {
	engine := NewMemoryEngine()

	vectors := make([]*core.Vector, b.N)
	for i := 0; i < b.N; i++ {
		vectors[i] = core.NewVector(string(rune(i)), []float32{float32(i)})
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		engine.Put(vectors[i])
	}
}

func BenchmarkDiskEnginePut(b *testing.B) {
	tmpDir, _ := os.MkdirTemp("", "bench-disk-*")
	defer os.RemoveAll(tmpDir)

	engine, _ := NewDiskEngine(tmpDir)
	defer engine.Close()

	vectors := make([]*core.Vector, b.N)
	for i := 0; i < b.N; i++ {
		vectors[i] = core.NewVector(string(rune(i)), []float32{float32(i)})
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		engine.Put(vectors[i])
	}
}

func TestMMapEngine(t *testing.T) {
	tmpDir := t.TempDir()

	engine, err := NewMMapEngine(tmpDir)
	if err != nil {
		t.Fatalf("failed to create mmap engine: %v", err)
	}

	// Test Put and Get
	vec := core.NewVectorWithMetadata("mmap-vec-1", []float32{1, 2, 3, 4}, map[string]any{"key": "value"})
	if err := engine.Put(vec); err != nil {
		t.Fatalf("put failed: %v", err)
	}

	retrieved, err := engine.Get("mmap-vec-1")
	if err != nil {
		t.Fatalf("get failed: %v", err)
	}

	if retrieved.ID != "mmap-vec-1" {
		t.Errorf("ID mismatch: expected 'mmap-vec-1', got '%s'", retrieved.ID)
	}

	// Test persistence
	engine.Close()

	// Reopen and verify
	engine2, err := NewMMapEngine(tmpDir)
	if err != nil {
		t.Fatalf("failed to reopen mmap engine: %v", err)
	}
	defer engine2.Close()

	retrieved2, err := engine2.Get("mmap-vec-1")
	if err != nil {
		t.Fatalf("get after reopen failed: %v", err)
	}

	if retrieved2.ID != "mmap-vec-1" {
		t.Errorf("ID mismatch after reopen")
	}
}

func TestMMapEngineIterate(t *testing.T) {
	tmpDir := t.TempDir()

	engine, err := NewMMapEngine(tmpDir)
	if err != nil {
		t.Fatalf("failed to create mmap engine: %v", err)
	}

	// Insert multiple vectors
	for i := 0; i < 10; i++ {
		vec := core.NewVector(fmt.Sprintf("mmap-iterate-%d", i), []float32{float32(i), float32(i * 2)})
		engine.Put(vec)
	}

	// Count iterations
	count := 0
	err = engine.Iterate(func(_ *core.Vector) error {
		count++
		return nil
	})
	if err != nil {
		t.Fatalf("iterate failed: %v", err)
	}

	if count != 10 {
		t.Errorf("expected 10 iterations, got %d", count)
	}

	engine.Close()
}

func TestMMapEngineDelete(t *testing.T) {
	tmpDir := t.TempDir()

	engine, err := NewMMapEngine(tmpDir)
	if err != nil {
		t.Fatalf("failed to create mmap engine: %v", err)
	}

	// Insert and delete
	vec := core.NewVector("to-delete-mmap", []float32{1, 2, 3})
	engine.Put(vec)

	if engine.Size() != 1 {
		t.Errorf("expected size 1, got %d", engine.Size())
	}

	engine.Delete("to-delete-mmap")

	if engine.Size() != 0 {
		t.Errorf("expected size 0 after delete, got %d", engine.Size())
	}

	// Verify deletion
	_, err = engine.Get("to-delete-mmap")
	if err == nil {
		t.Error("deleted vector should not be found")
	}

	engine.Close()
}

func BenchmarkMMapEnginePut(b *testing.B) {
	tmpDir, _ := os.MkdirTemp("", "bench-mmap-*")
	defer os.RemoveAll(tmpDir)

	engine, err := NewMMapEngine(tmpDir)
	if err != nil {
		b.Fatalf("failed to create mmap engine: %v", err)
	}
	defer engine.Close()

	vectors := make([]*core.Vector, b.N)
	for i := 0; i < b.N; i++ {
		vectors[i] = core.NewVector(fmt.Sprintf("bench-mmap-%d", i), []float32{float32(i), float32(i * 2)})
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		engine.Put(vectors[i])
	}
}
