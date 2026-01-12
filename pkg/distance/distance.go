// Package distance provides various distance metrics for vector similarity.
package distance

import (
	"math"
)

// Metric represents the type of distance metric to use.
type Metric int

const (
	// Cosine distance (1 - cosine similarity)
	Cosine Metric = iota
	// Euclidean (L2) distance
	Euclidean
	// DotProduct (inner product, higher is more similar)
	DotProduct
	// Manhattan (L1) distance
	Manhattan
	// Hamming distance for binary vectors
	Hamming
)

// String returns the string representation of the metric.
func (m Metric) String() string {
	switch m {
	case Cosine:
		return "cosine"
	case Euclidean:
		return "euclidean"
	case DotProduct:
		return "dotproduct"
	case Manhattan:
		return "manhattan"
	case Hamming:
		return "hamming"
	default:
		return "unknown"
	}
}

// ParseMetric parses a string into a Metric.
func ParseMetric(s string) Metric {
	switch s {
	case "cosine":
		return Cosine
	case "euclidean", "l2":
		return Euclidean
	case "dotproduct", "dot", "ip":
		return DotProduct
	case "manhattan", "l1":
		return Manhattan
	case "hamming":
		return Hamming
	default:
		return Cosine
	}
}

// Calculator is the interface for distance calculations.
type Calculator interface {
	// Distance calculates the distance between two vectors.
	// Lower values mean more similar for distance metrics.
	Distance(a, b []float32) float32
	// Metric returns the metric type.
	Metric() Metric
}

// NewCalculator creates a new distance calculator for the given metric.
func NewCalculator(m Metric) Calculator {
	switch m {
	case Cosine:
		return &CosineCalculator{}
	case Euclidean:
		return &EuclideanCalculator{}
	case DotProduct:
		return &DotProductCalculator{}
	case Manhattan:
		return &ManhattanCalculator{}
	case Hamming:
		return &HammingCalculator{}
	default:
		return &CosineCalculator{}
	}
}

// CosineCalculator calculates cosine distance.
type CosineCalculator struct{}

// Distance calculates cosine distance (1 - cosine similarity).
func (c *CosineCalculator) Distance(a, b []float32) float32 {
	return 1.0 - CosineSimilarity(a, b)
}

// Metric returns Cosine.
func (c *CosineCalculator) Metric() Metric { return Cosine }

// EuclideanCalculator calculates Euclidean (L2) distance.
type EuclideanCalculator struct{}

// Distance calculates the Euclidean distance.
func (e *EuclideanCalculator) Distance(a, b []float32) float32 {
	return EuclideanDistance(a, b)
}

// Metric returns Euclidean.
func (e *EuclideanCalculator) Metric() Metric { return Euclidean }

// DotProductCalculator calculates dot product (negative for distance).
type DotProductCalculator struct{}

// Distance returns negative dot product (so lower is more similar).
func (d *DotProductCalculator) Distance(a, b []float32) float32 {
	return -DotProductValue(a, b)
}

// Metric returns DotProduct.
func (d *DotProductCalculator) Metric() Metric { return DotProduct }

// ManhattanCalculator calculates Manhattan (L1) distance.
type ManhattanCalculator struct{}

// Distance calculates the Manhattan distance.
func (m *ManhattanCalculator) Distance(a, b []float32) float32 {
	return ManhattanDistance(a, b)
}

// Metric returns Manhattan.
func (m *ManhattanCalculator) Metric() Metric { return Manhattan }

// HammingCalculator calculates Hamming distance.
type HammingCalculator struct{}

// Distance calculates the Hamming distance.
func (h *HammingCalculator) Distance(a, b []float32) float32 {
	return HammingDistance(a, b)
}

// Metric returns Hamming.
func (h *HammingCalculator) Metric() Metric { return Hamming }

// CosineSimilarity computes the cosine similarity between two vectors.
// Returns a value between -1 and 1, where 1 means identical direction.
func CosineSimilarity(a, b []float32) float32 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}
	
	var dotProduct, normA, normB float32
	
	// Process 4 elements at a time for better CPU utilization
	n := len(a)
	i := 0
	for ; i <= n-4; i += 4 {
		dotProduct += a[i]*b[i] + a[i+1]*b[i+1] + a[i+2]*b[i+2] + a[i+3]*b[i+3]
		normA += a[i]*a[i] + a[i+1]*a[i+1] + a[i+2]*a[i+2] + a[i+3]*a[i+3]
		normB += b[i]*b[i] + b[i+1]*b[i+1] + b[i+2]*b[i+2] + b[i+3]*b[i+3]
	}
	// Handle remaining elements
	for ; i < n; i++ {
		dotProduct += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}
	
	if normA == 0 || normB == 0 {
		return 0
	}
	
	return dotProduct / (float32(math.Sqrt(float64(normA))) * float32(math.Sqrt(float64(normB))))
}

// EuclideanDistance computes the Euclidean (L2) distance between two vectors.
func EuclideanDistance(a, b []float32) float32 {
	if len(a) != len(b) || len(a) == 0 {
		return float32(math.MaxFloat32)
	}
	
	var sum float32
	
	// Process 4 elements at a time
	n := len(a)
	i := 0
	for ; i <= n-4; i += 4 {
		d0 := a[i] - b[i]
		d1 := a[i+1] - b[i+1]
		d2 := a[i+2] - b[i+2]
		d3 := a[i+3] - b[i+3]
		sum += d0*d0 + d1*d1 + d2*d2 + d3*d3
	}
	// Handle remaining elements
	for ; i < n; i++ {
		d := a[i] - b[i]
		sum += d * d
	}
	
	return float32(math.Sqrt(float64(sum)))
}

// EuclideanDistanceSquared returns the squared Euclidean distance (faster, no sqrt).
func EuclideanDistanceSquared(a, b []float32) float32 {
	if len(a) != len(b) || len(a) == 0 {
		return float32(math.MaxFloat32)
	}
	
	var sum float32
	
	n := len(a)
	i := 0
	for ; i <= n-4; i += 4 {
		d0 := a[i] - b[i]
		d1 := a[i+1] - b[i+1]
		d2 := a[i+2] - b[i+2]
		d3 := a[i+3] - b[i+3]
		sum += d0*d0 + d1*d1 + d2*d2 + d3*d3
	}
	for ; i < n; i++ {
		d := a[i] - b[i]
		sum += d * d
	}
	
	return sum
}

// DotProductValue computes the dot product between two vectors.
func DotProductValue(a, b []float32) float32 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}
	
	var sum float32
	
	// Process 4 elements at a time
	n := len(a)
	i := 0
	for ; i <= n-4; i += 4 {
		sum += a[i]*b[i] + a[i+1]*b[i+1] + a[i+2]*b[i+2] + a[i+3]*b[i+3]
	}
	// Handle remaining elements
	for ; i < n; i++ {
		sum += a[i] * b[i]
	}
	
	return sum
}

// ManhattanDistance computes the Manhattan (L1) distance between two vectors.
func ManhattanDistance(a, b []float32) float32 {
	if len(a) != len(b) || len(a) == 0 {
		return float32(math.MaxFloat32)
	}
	
	var sum float32
	
	n := len(a)
	i := 0
	for ; i <= n-4; i += 4 {
		sum += float32(math.Abs(float64(a[i]-b[i]))) +
			float32(math.Abs(float64(a[i+1]-b[i+1]))) +
			float32(math.Abs(float64(a[i+2]-b[i+2]))) +
			float32(math.Abs(float64(a[i+3]-b[i+3])))
	}
	for ; i < n; i++ {
		sum += float32(math.Abs(float64(a[i] - b[i])))
	}
	
	return sum
}

// HammingDistance computes the Hamming distance (count of different elements).
// For continuous vectors, it counts elements with different signs.
func HammingDistance(a, b []float32) float32 {
	if len(a) != len(b) || len(a) == 0 {
		return float32(len(a))
	}
	
	var count float32
	for i := range a {
		if (a[i] >= 0) != (b[i] >= 0) {
			count++
		}
	}
	
	return count
}

// NormalizeVector normalizes a vector to unit length in place.
func NormalizeVector(v []float32) {
	var norm float32
	for _, val := range v {
		norm += val * val
	}
	norm = float32(math.Sqrt(float64(norm)))
	if norm > 0 {
		for i := range v {
			v[i] /= norm
		}
	}
}

// BatchCosineSimilarity computes cosine similarity between a query and multiple vectors.
func BatchCosineSimilarity(query []float32, vectors [][]float32, results []float32) {
	for i, v := range vectors {
		results[i] = CosineSimilarity(query, v)
	}
}

// BatchEuclideanDistance computes Euclidean distance between a query and multiple vectors.
func BatchEuclideanDistance(query []float32, vectors [][]float32, results []float32) {
	for i, v := range vectors {
		results[i] = EuclideanDistance(query, v)
	}
}
