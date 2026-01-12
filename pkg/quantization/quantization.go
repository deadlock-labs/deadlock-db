// Package quantization provides vector quantization for memory-efficient storage.
package quantization

import (
	"math"
	"sync"
)

// Quantizer is the interface for vector quantization.
type Quantizer interface {
	// Encode quantizes a vector.
	Encode(vector []float32) []byte
	// Decode reconstructs a vector from quantized form.
	Decode(encoded []byte) []float32
	// DistanceSquared computes approximate squared distance without full decode.
	DistanceSquared(query []float32, encoded []byte) float32
	// Dimension returns the expected vector dimension.
	Dimension() int
}

// ScalarQuantizer implements scalar quantization (SQ8).
// Each float32 is quantized to a single uint8 byte.
type ScalarQuantizer struct {
	dim    int
	minVal []float32
	maxVal []float32
	scale  []float32
	mu     sync.RWMutex
	trained bool
}

// NewScalarQuantizer creates a new scalar quantizer for the given dimension.
func NewScalarQuantizer(dimension int) *ScalarQuantizer {
	return &ScalarQuantizer{
		dim:    dimension,
		minVal: make([]float32, dimension),
		maxVal: make([]float32, dimension),
		scale:  make([]float32, dimension),
	}
}

// Train trains the quantizer on a set of training vectors.
func (sq *ScalarQuantizer) Train(vectors [][]float32) {
	if len(vectors) == 0 {
		return
	}

	sq.mu.Lock()
	defer sq.mu.Unlock()

	// Initialize with first vector
	for i := 0; i < sq.dim; i++ {
		sq.minVal[i] = vectors[0][i]
		sq.maxVal[i] = vectors[0][i]
	}

	// Find min/max for each dimension
	for _, v := range vectors {
		for i := 0; i < sq.dim && i < len(v); i++ {
			if v[i] < sq.minVal[i] {
				sq.minVal[i] = v[i]
			}
			if v[i] > sq.maxVal[i] {
				sq.maxVal[i] = v[i]
			}
		}
	}

	// Calculate scale factors
	for i := 0; i < sq.dim; i++ {
		rangeVal := sq.maxVal[i] - sq.minVal[i]
		if rangeVal > 0 {
			sq.scale[i] = 255.0 / rangeVal
		} else {
			sq.scale[i] = 1.0
		}
	}

	sq.trained = true
}

// Encode quantizes a vector to bytes.
func (sq *ScalarQuantizer) Encode(vector []float32) []byte {
	sq.mu.RLock()
	defer sq.mu.RUnlock()

	if !sq.trained {
		// Use default range [-1, 1] if not trained
		encoded := make([]byte, sq.dim)
		for i := 0; i < sq.dim && i < len(vector); i++ {
			// Clamp to [-1, 1] range
			val := vector[i]
			if val < -1 {
				val = -1
			}
			if val > 1 {
				val = 1
			}
			// Map [-1, 1] to [0, 255]
			encoded[i] = byte((val + 1) * 127.5)
		}
		return encoded
	}

	encoded := make([]byte, sq.dim)
	for i := 0; i < sq.dim && i < len(vector); i++ {
		// Clamp to trained range
		val := vector[i]
		if val < sq.minVal[i] {
			val = sq.minVal[i]
		}
		if val > sq.maxVal[i] {
			val = sq.maxVal[i]
		}
		// Scale to [0, 255]
		encoded[i] = byte((val - sq.minVal[i]) * sq.scale[i])
	}
	return encoded
}

// Decode reconstructs a vector from quantized form.
func (sq *ScalarQuantizer) Decode(encoded []byte) []float32 {
	sq.mu.RLock()
	defer sq.mu.RUnlock()

	vector := make([]float32, sq.dim)

	if !sq.trained {
		for i := 0; i < sq.dim && i < len(encoded); i++ {
			// Map [0, 255] back to [-1, 1]
			vector[i] = float32(encoded[i])/127.5 - 1
		}
		return vector
	}

	for i := 0; i < sq.dim && i < len(encoded); i++ {
		vector[i] = float32(encoded[i])/sq.scale[i] + sq.minVal[i]
	}
	return vector
}

// DistanceSquared computes approximate squared Euclidean distance.
func (sq *ScalarQuantizer) DistanceSquared(query []float32, encoded []byte) float32 {
	sq.mu.RLock()
	defer sq.mu.RUnlock()

	var sum float32
	for i := 0; i < sq.dim && i < len(query) && i < len(encoded); i++ {
		var decoded float32
		if !sq.trained {
			decoded = float32(encoded[i])/127.5 - 1
		} else {
			decoded = float32(encoded[i])/sq.scale[i] + sq.minVal[i]
		}
		diff := query[i] - decoded
		sum += diff * diff
	}
	return sum
}

// Dimension returns the expected vector dimension.
func (sq *ScalarQuantizer) Dimension() int {
	return sq.dim
}

// ProductQuantizer implements product quantization (PQ).
// Divides vector into subvectors and quantizes each independently.
type ProductQuantizer struct {
	dim        int
	numSubvecs int
	subvecDim  int
	numCodes   int  // Number of centroids per subvector (typically 256)
	centroids  [][]float32 // [numSubvecs * numCodes][subvecDim]
	mu         sync.RWMutex
	trained    bool
}

// NewProductQuantizer creates a new product quantizer.
// numSubvecs should divide dimension evenly.
func NewProductQuantizer(dimension, numSubvecs int) *ProductQuantizer {
	subvecDim := dimension / numSubvecs
	if dimension%numSubvecs != 0 {
		subvecDim++
	}
	return &ProductQuantizer{
		dim:        dimension,
		numSubvecs: numSubvecs,
		subvecDim:  subvecDim,
		numCodes:   256,
		centroids:  nil,
	}
}

// Train trains the product quantizer using k-means on training vectors.
func (pq *ProductQuantizer) Train(vectors [][]float32) {
	if len(vectors) == 0 {
		return
	}

	pq.mu.Lock()
	defer pq.mu.Unlock()

	// Initialize centroids for each subvector
	pq.centroids = make([][]float32, pq.numSubvecs*pq.numCodes)
	for i := range pq.centroids {
		pq.centroids[i] = make([]float32, pq.subvecDim)
	}

	// Train each subvector independently using k-means
	for m := 0; m < pq.numSubvecs; m++ {
		// Extract subvectors
		subvecs := make([][]float32, len(vectors))
		for i, v := range vectors {
			start := m * pq.subvecDim
			end := start + pq.subvecDim
			if end > len(v) {
				end = len(v)
			}
			subvecs[i] = v[start:end]
		}

		// Run k-means
		centroids := kmeans(subvecs, pq.numCodes, 10)
		for k := 0; k < pq.numCodes; k++ {
			copy(pq.centroids[m*pq.numCodes+k], centroids[k])
		}
	}

	pq.trained = true
}

// kmeans performs k-means clustering.
func kmeans(vectors [][]float32, k, iterations int) [][]float32 {
	if len(vectors) == 0 || k == 0 {
		return nil
	}

	dim := len(vectors[0])
	centroids := make([][]float32, k)

	// Initialize centroids with random vectors
	for i := 0; i < k; i++ {
		centroids[i] = make([]float32, dim)
		if i < len(vectors) {
			copy(centroids[i], vectors[i%len(vectors)])
		}
	}

	assignments := make([]int, len(vectors))

	for iter := 0; iter < iterations; iter++ {
		// Assign each vector to nearest centroid
		for i, v := range vectors {
			minDist := float32(math.MaxFloat32)
			minIdx := 0
			for j, c := range centroids {
				dist := squaredDistance(v, c)
				if dist < minDist {
					minDist = dist
					minIdx = j
				}
			}
			assignments[i] = minIdx
		}

		// Update centroids
		counts := make([]int, k)
		newCentroids := make([][]float32, k)
		for i := range newCentroids {
			newCentroids[i] = make([]float32, dim)
		}

		for i, v := range vectors {
			c := assignments[i]
			counts[c]++
			for j := range v {
				newCentroids[c][j] += v[j]
			}
		}

		for i := range centroids {
			if counts[i] > 0 {
				for j := range centroids[i] {
					centroids[i][j] = newCentroids[i][j] / float32(counts[i])
				}
			}
		}
	}

	return centroids
}

// squaredDistance computes squared Euclidean distance.
func squaredDistance(a, b []float32) float32 {
	var sum float32
	for i := range a {
		if i < len(b) {
			diff := a[i] - b[i]
			sum += diff * diff
		}
	}
	return sum
}

// Encode quantizes a vector using product quantization.
func (pq *ProductQuantizer) Encode(vector []float32) []byte {
	pq.mu.RLock()
	defer pq.mu.RUnlock()

	if !pq.trained {
		// Return zeros if not trained
		return make([]byte, pq.numSubvecs)
	}

	encoded := make([]byte, pq.numSubvecs)
	for m := 0; m < pq.numSubvecs; m++ {
		// Extract subvector
		start := m * pq.subvecDim
		end := start + pq.subvecDim
		if end > len(vector) {
			end = len(vector)
		}
		subvec := vector[start:end]

		// Find nearest centroid
		minDist := float32(math.MaxFloat32)
		minIdx := 0
		for k := 0; k < pq.numCodes; k++ {
			dist := squaredDistance(subvec, pq.centroids[m*pq.numCodes+k])
			if dist < minDist {
				minDist = dist
				minIdx = k
			}
		}
		encoded[m] = byte(minIdx)
	}
	return encoded
}

// Decode reconstructs a vector from product-quantized form.
func (pq *ProductQuantizer) Decode(encoded []byte) []float32 {
	pq.mu.RLock()
	defer pq.mu.RUnlock()

	vector := make([]float32, pq.dim)
	if !pq.trained {
		return vector
	}

	for m := 0; m < pq.numSubvecs && m < len(encoded); m++ {
		code := int(encoded[m])
		centroid := pq.centroids[m*pq.numCodes+code]
		start := m * pq.subvecDim
		for i := 0; i < pq.subvecDim && start+i < pq.dim; i++ {
			if i < len(centroid) {
				vector[start+i] = centroid[i]
			}
		}
	}
	return vector
}

// DistanceSquared computes approximate squared distance using lookup tables.
func (pq *ProductQuantizer) DistanceSquared(query []float32, encoded []byte) float32 {
	pq.mu.RLock()
	defer pq.mu.RUnlock()

	if !pq.trained {
		return float32(math.MaxFloat32)
	}

	var sum float32
	for m := 0; m < pq.numSubvecs && m < len(encoded); m++ {
		code := int(encoded[m])
		centroid := pq.centroids[m*pq.numCodes+code]
		
		start := m * pq.subvecDim
		end := start + pq.subvecDim
		if end > len(query) {
			end = len(query)
		}
		
		for i := start; i < end; i++ {
			if i-start < len(centroid) {
				diff := query[i] - centroid[i-start]
				sum += diff * diff
			}
		}
	}
	return sum
}

// Dimension returns the expected vector dimension.
func (pq *ProductQuantizer) Dimension() int {
	return pq.dim
}

// BinaryQuantizer quantizes vectors to binary (1 bit per dimension).
type BinaryQuantizer struct {
	dim int
}

// NewBinaryQuantizer creates a new binary quantizer.
func NewBinaryQuantizer(dimension int) *BinaryQuantizer {
	return &BinaryQuantizer{dim: dimension}
}

// Encode quantizes a vector to binary.
func (bq *BinaryQuantizer) Encode(vector []float32) []byte {
	numBytes := (bq.dim + 7) / 8
	encoded := make([]byte, numBytes)
	
	for i := 0; i < bq.dim && i < len(vector); i++ {
		if vector[i] > 0 {
			encoded[i/8] |= 1 << (i % 8)
		}
	}
	return encoded
}

// Decode reconstructs a vector from binary form.
func (bq *BinaryQuantizer) Decode(encoded []byte) []float32 {
	vector := make([]float32, bq.dim)
	
	for i := 0; i < bq.dim && i/8 < len(encoded); i++ {
		if (encoded[i/8] & (1 << (i % 8))) != 0 {
			vector[i] = 1
		} else {
			vector[i] = -1
		}
	}
	return vector
}

// DistanceSquared computes Hamming distance for binary vectors.
func (bq *BinaryQuantizer) DistanceSquared(query []float32, encoded []byte) float32 {
	// Convert query to binary and compute Hamming distance
	queryBinary := bq.Encode(query)
	
	var dist int
	for i := 0; i < len(queryBinary) && i < len(encoded); i++ {
		xor := queryBinary[i] ^ encoded[i]
		// Count set bits (Hamming distance)
		for xor != 0 {
			dist++
			xor &= xor - 1
		}
	}
	return float32(dist)
}

// Dimension returns the expected vector dimension.
func (bq *BinaryQuantizer) Dimension() int {
	return bq.dim
}
