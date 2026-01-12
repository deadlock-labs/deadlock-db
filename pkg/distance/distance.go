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
// Uses 8-way loop unrolling for better CPU pipelining and cache utilization.
func CosineSimilarity(a, b []float32) float32 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}

	var dotProduct, normA, normB float32
	// Use multiple accumulators to reduce data dependencies and improve ILP
	var dot1, dot2, normA1, normA2, normB1, normB2 float32

	// Process 8 elements at a time for better CPU pipelining
	n := len(a)
	i := 0
	for ; i <= n-8; i += 8 {
		// First 4 elements
		dot1 += a[i]*b[i] + a[i+1]*b[i+1] + a[i+2]*b[i+2] + a[i+3]*b[i+3]
		normA1 += a[i]*a[i] + a[i+1]*a[i+1] + a[i+2]*a[i+2] + a[i+3]*a[i+3]
		normB1 += b[i]*b[i] + b[i+1]*b[i+1] + b[i+2]*b[i+2] + b[i+3]*b[i+3]
		// Second 4 elements (parallel computation path)
		dot2 += a[i+4]*b[i+4] + a[i+5]*b[i+5] + a[i+6]*b[i+6] + a[i+7]*b[i+7]
		normA2 += a[i+4]*a[i+4] + a[i+5]*a[i+5] + a[i+6]*a[i+6] + a[i+7]*a[i+7]
		normB2 += b[i+4]*b[i+4] + b[i+5]*b[i+5] + b[i+6]*b[i+6] + b[i+7]*b[i+7]
	}
	// Combine accumulators
	dotProduct = dot1 + dot2
	normA = normA1 + normA2
	normB = normB1 + normB2

	// Handle remaining elements (up to 7)
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
// Uses 8-way loop unrolling with dual accumulators for better ILP.
func EuclideanDistance(a, b []float32) float32 {
	if len(a) != len(b) || len(a) == 0 {
		return float32(math.MaxFloat32)
	}

	var sum1, sum2 float32

	// Process 8 elements at a time
	n := len(a)
	i := 0
	for ; i <= n-8; i += 8 {
		d0 := a[i] - b[i]
		d1 := a[i+1] - b[i+1]
		d2 := a[i+2] - b[i+2]
		d3 := a[i+3] - b[i+3]
		d4 := a[i+4] - b[i+4]
		d5 := a[i+5] - b[i+5]
		d6 := a[i+6] - b[i+6]
		d7 := a[i+7] - b[i+7]
		sum1 += d0*d0 + d1*d1 + d2*d2 + d3*d3
		sum2 += d4*d4 + d5*d5 + d6*d6 + d7*d7
	}
	// Handle remaining elements
	sum := sum1 + sum2
	for ; i < n; i++ {
		d := a[i] - b[i]
		sum += d * d
	}

	return float32(math.Sqrt(float64(sum)))
}

// EuclideanDistanceSquared returns the squared Euclidean distance (faster, no sqrt).
// Uses 8-way loop unrolling for better CPU pipelining.
func EuclideanDistanceSquared(a, b []float32) float32 {
	if len(a) != len(b) || len(a) == 0 {
		return float32(math.MaxFloat32)
	}

	var sum1, sum2 float32

	n := len(a)
	i := 0
	for ; i <= n-8; i += 8 {
		d0 := a[i] - b[i]
		d1 := a[i+1] - b[i+1]
		d2 := a[i+2] - b[i+2]
		d3 := a[i+3] - b[i+3]
		d4 := a[i+4] - b[i+4]
		d5 := a[i+5] - b[i+5]
		d6 := a[i+6] - b[i+6]
		d7 := a[i+7] - b[i+7]
		sum1 += d0*d0 + d1*d1 + d2*d2 + d3*d3
		sum2 += d4*d4 + d5*d5 + d6*d6 + d7*d7
	}
	sum := sum1 + sum2
	for ; i < n; i++ {
		d := a[i] - b[i]
		sum += d * d
	}

	return sum
}

// DotProductValue computes the dot product between two vectors.
// Uses 8-way loop unrolling with dual accumulators for better ILP.
func DotProductValue(a, b []float32) float32 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}

	var sum1, sum2 float32

	// Process 8 elements at a time
	n := len(a)
	i := 0
	for ; i <= n-8; i += 8 {
		sum1 += a[i]*b[i] + a[i+1]*b[i+1] + a[i+2]*b[i+2] + a[i+3]*b[i+3]
		sum2 += a[i+4]*b[i+4] + a[i+5]*b[i+5] + a[i+6]*b[i+6] + a[i+7]*b[i+7]
	}
	// Handle remaining elements
	sum := sum1 + sum2
	for ; i < n; i++ {
		sum += a[i] * b[i]
	}

	return sum
}

// ManhattanDistance computes the Manhattan (L1) distance between two vectors.
// Uses abs() without math.Abs for better performance.
func ManhattanDistance(a, b []float32) float32 {
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
		// Branchless abs: if d < 0, flip sign
		if d0 < 0 {
			d0 = -d0
		}
		if d1 < 0 {
			d1 = -d1
		}
		if d2 < 0 {
			d2 = -d2
		}
		if d3 < 0 {
			d3 = -d3
		}
		sum += d0 + d1 + d2 + d3
	}
	for ; i < n; i++ {
		d := a[i] - b[i]
		if d < 0 {
			d = -d
		}
		sum += d
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
// Processes vectors in a cache-friendly manner.
func BatchCosineSimilarity(query []float32, vectors [][]float32, results []float32) {
	// Process in small batches for better cache utilization
	const batchSize = 4
	n := len(vectors)
	i := 0
	for ; i <= n-batchSize; i += batchSize {
		results[i] = CosineSimilarity(query, vectors[i])
		results[i+1] = CosineSimilarity(query, vectors[i+1])
		results[i+2] = CosineSimilarity(query, vectors[i+2])
		results[i+3] = CosineSimilarity(query, vectors[i+3])
	}
	for ; i < n; i++ {
		results[i] = CosineSimilarity(query, vectors[i])
	}
}

// BatchEuclideanDistance computes Euclidean distance between a query and multiple vectors.
// Processes vectors in a cache-friendly manner.
func BatchEuclideanDistance(query []float32, vectors [][]float32, results []float32) {
	const batchSize = 4
	n := len(vectors)
	i := 0
	for ; i <= n-batchSize; i += batchSize {
		results[i] = EuclideanDistance(query, vectors[i])
		results[i+1] = EuclideanDistance(query, vectors[i+1])
		results[i+2] = EuclideanDistance(query, vectors[i+2])
		results[i+3] = EuclideanDistance(query, vectors[i+3])
	}
	for ; i < n; i++ {
		results[i] = EuclideanDistance(query, vectors[i])
	}
}

// PrefetchVector is a hint to prefetch the next vector into cache.
// This is a no-op on architectures without prefetch but helps on x86/ARM.
func PrefetchVector(v []float32) {
	// Go's runtime doesn't expose prefetch instructions directly,
	// but accessing the first element helps with cache line loading.
	if len(v) > 0 {
		_ = v[0]
	}
}
