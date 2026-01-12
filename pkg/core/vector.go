// Package core provides the fundamental data structures for deadlock-db.
package core

import (
	"encoding/binary"
	"fmt"
	"math"
	"sync"
	"time"
)

// Vector represents a dense vector with an ID and optional metadata.
type Vector struct {
	ID        string         `json:"id"`
	Values    []float32      `json:"values"`
	Metadata  map[string]any `json:"metadata,omitempty"`
	Sparse    *SparseVector  `json:"sparse,omitempty"`
	Version   int64          `json:"version,omitempty"`   // Auto-incrementing version number
	Timestamp int64          `json:"timestamp,omitempty"` // Unix nanoseconds when created/updated
}

// SparseVector represents a sparse vector using indices and values.
type SparseVector struct {
	Indices []uint32  `json:"indices"`
	Values  []float32 `json:"values"`
}

// NewVector creates a new vector with the given ID and values.
func NewVector(id string, values []float32) *Vector {
	return &Vector{
		ID:        id,
		Values:    values,
		Metadata:  make(map[string]any),
		Version:   1,
		Timestamp: time.Now().UnixNano(),
	}
}

// NewVectorWithMetadata creates a new vector with metadata.
func NewVectorWithMetadata(id string, values []float32, metadata map[string]any) *Vector {
	return &Vector{
		ID:        id,
		Values:    values,
		Metadata:  metadata,
		Version:   1,
		Timestamp: time.Now().UnixNano(),
	}
}

// Dimension returns the dimensionality of the vector.
func (v *Vector) Dimension() int {
	return len(v.Values)
}

// Clone creates a deep copy of the vector.
func (v *Vector) Clone() *Vector {
	clone := &Vector{
		ID:        v.ID,
		Values:    make([]float32, len(v.Values)),
		Metadata:  make(map[string]any, len(v.Metadata)),
		Version:   v.Version,
		Timestamp: v.Timestamp,
	}
	copy(clone.Values, v.Values)
	for k, val := range v.Metadata {
		clone.Metadata[k] = val
	}
	if v.Sparse != nil {
		clone.Sparse = &SparseVector{
			Indices: make([]uint32, len(v.Sparse.Indices)),
			Values:  make([]float32, len(v.Sparse.Values)),
		}
		copy(clone.Sparse.Indices, v.Sparse.Indices)
		copy(clone.Sparse.Values, v.Sparse.Values)
	}
	return clone
}

// IncrementVersion increments the version and updates the timestamp.
// Call this when updating a vector.
func (v *Vector) IncrementVersion() {
	v.Version++
	v.Timestamp = time.Now().UnixNano()
}

// Normalize normalizes the vector to unit length (L2 norm).
func (v *Vector) Normalize() {
	norm := float32(0)
	for _, val := range v.Values {
		norm += val * val
	}
	norm = float32(math.Sqrt(float64(norm)))
	if norm > 0 {
		for i := range v.Values {
			v.Values[i] /= norm
		}
	}
}

// Magnitude returns the L2 norm of the vector.
func (v *Vector) Magnitude() float32 {
	sum := float32(0)
	for _, val := range v.Values {
		sum += val * val
	}
	return float32(math.Sqrt(float64(sum)))
}

// Serialize converts the vector to bytes for storage.
func (v *Vector) Serialize() ([]byte, error) {
	// Calculate buffer size
	idBytes := []byte(v.ID)
	size := 4 + len(idBytes) + // ID length + ID
		4 + len(v.Values)*4 // values length + values
	
	buf := make([]byte, size)
	offset := 0
	
	// Write ID
	binary.LittleEndian.PutUint32(buf[offset:], uint32(len(idBytes)))
	offset += 4
	copy(buf[offset:], idBytes)
	offset += len(idBytes)
	
	// Write values
	binary.LittleEndian.PutUint32(buf[offset:], uint32(len(v.Values)))
	offset += 4
	for _, val := range v.Values {
		binary.LittleEndian.PutUint32(buf[offset:], math.Float32bits(val))
		offset += 4
	}
	
	return buf, nil
}

// Deserialize reconstructs a vector from bytes.
func Deserialize(data []byte) (*Vector, error) {
	if len(data) < 8 {
		return nil, fmt.Errorf("data too short for vector deserialization")
	}
	
	offset := 0
	
	// Read ID
	idLen := binary.LittleEndian.Uint32(data[offset:])
	offset += 4
	if offset+int(idLen) > len(data) {
		return nil, fmt.Errorf("invalid ID length in vector data")
	}
	id := string(data[offset : offset+int(idLen)])
	offset += int(idLen)
	
	// Read values
	if offset+4 > len(data) {
		return nil, fmt.Errorf("data too short for values length")
	}
	valuesLen := binary.LittleEndian.Uint32(data[offset:])
	offset += 4
	
	if offset+int(valuesLen)*4 > len(data) {
		return nil, fmt.Errorf("data too short for values")
	}
	
	values := make([]float32, valuesLen)
	for i := range values {
		values[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[offset:]))
		offset += 4
	}
	
	return NewVector(id, values), nil
}

// VectorPool provides a pool of reusable vector slices to reduce allocations.
type VectorPool struct {
	pool sync.Pool
	dim  int
}

// NewVectorPool creates a new vector pool for the given dimension.
func NewVectorPool(dimension int) *VectorPool {
	return &VectorPool{
		dim: dimension,
		pool: sync.Pool{
			New: func() any {
				return make([]float32, dimension)
			},
		},
	}
}

// Get retrieves a vector slice from the pool.
func (vp *VectorPool) Get() []float32 {
	return vp.pool.Get().([]float32)
}

// Put returns a vector slice to the pool.
func (vp *VectorPool) Put(v []float32) {
	if len(v) == vp.dim {
		// Zero out the slice before returning
		for i := range v {
			v[i] = 0
		}
		vp.pool.Put(v)
	}
}

// SearchResult represents a single search result.
type SearchResult struct {
	ID       string         `json:"id"`
	Score    float32        `json:"score"`
	Vector   *Vector        `json:"vector,omitempty"`
	Metadata map[string]any `json:"metadata,omitempty"`
}

// SearchResults is a slice of search results.
type SearchResults []SearchResult

// Len returns the number of results.
func (sr SearchResults) Len() int { return len(sr) }

// Less compares two results by score (higher is better for similarity).
func (sr SearchResults) Less(i, j int) bool { return sr[i].Score > sr[j].Score }

// Swap swaps two results.
func (sr SearchResults) Swap(i, j int) { sr[i], sr[j] = sr[j], sr[i] }
