package core

import (
	"testing"
)

func TestNewVector(t *testing.T) {
	id := "test-vector"
	values := []float32{1.0, 2.0, 3.0, 4.0}

	v := NewVector(id, values)

	if v.ID != id {
		t.Errorf("expected ID %s, got %s", id, v.ID)
	}
	if len(v.Values) != len(values) {
		t.Errorf("expected %d values, got %d", len(values), len(v.Values))
	}
	for i, val := range values {
		if v.Values[i] != val {
			t.Errorf("expected value[%d] = %f, got %f", i, val, v.Values[i])
		}
	}
}

func TestVectorDimension(t *testing.T) {
	v := NewVector("test", []float32{1, 2, 3, 4, 5})
	if v.Dimension() != 5 {
		t.Errorf("expected dimension 5, got %d", v.Dimension())
	}
}

func TestVectorClone(t *testing.T) {
	v := NewVectorWithMetadata("test", []float32{1, 2, 3}, map[string]any{"key": "value"})
	clone := v.Clone()

	if clone.ID != v.ID {
		t.Errorf("clone ID mismatch")
	}
	
	// Modify original and verify clone is independent
	v.Values[0] = 100
	if clone.Values[0] == 100 {
		t.Error("clone values should be independent")
	}
}

func TestVectorNormalize(t *testing.T) {
	v := NewVector("test", []float32{3, 4})
	v.Normalize()

	// Expected: [0.6, 0.8] (magnitude should be 1)
	magnitude := v.Magnitude()
	if magnitude < 0.999 || magnitude > 1.001 {
		t.Errorf("expected magnitude 1, got %f", magnitude)
	}
}

func TestVectorSerializeDeserialize(t *testing.T) {
	original := NewVector("test-id", []float32{1.5, 2.5, 3.5, 4.5})

	data, err := original.Serialize()
	if err != nil {
		t.Fatalf("serialize error: %v", err)
	}

	restored, err := Deserialize(data)
	if err != nil {
		t.Fatalf("deserialize error: %v", err)
	}

	if restored.ID != original.ID {
		t.Errorf("ID mismatch: expected %s, got %s", original.ID, restored.ID)
	}
	if len(restored.Values) != len(original.Values) {
		t.Errorf("values length mismatch")
	}
	for i := range original.Values {
		if restored.Values[i] != original.Values[i] {
			t.Errorf("value mismatch at %d", i)
		}
	}
}

func TestVectorPool(t *testing.T) {
	pool := NewVectorPool(128)

	v1 := pool.Get()
	if len(v1) != 128 {
		t.Errorf("expected 128 elements, got %d", len(v1))
	}

	// Modify and return
	v1[0] = 1.0
	pool.Put(v1)

	// Get again - should be zeroed
	v2 := pool.Get()
	if v2[0] != 0 {
		t.Error("pooled vector should be zeroed")
	}
}

func BenchmarkVectorNormalize(b *testing.B) {
	v := NewVector("bench", make([]float32, 1024))
	for i := range v.Values {
		v.Values[i] = float32(i)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		v.Normalize()
	}
}

func BenchmarkVectorSerialize(b *testing.B) {
	v := NewVector("bench", make([]float32, 1024))
	for i := range v.Values {
		v.Values[i] = float32(i)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		v.Serialize()
	}
}
