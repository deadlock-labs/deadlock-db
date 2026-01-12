package quantization

import (
	"math"
	"testing"
)

func TestScalarQuantizer(t *testing.T) {
	dim := 128
	sq := NewScalarQuantizer(dim)

	// Test without training (default range)
	original := make([]float32, dim)
	for i := range original {
		original[i] = float32(i)/float32(dim)*2 - 1 // Range [-1, 1]
	}

	encoded := sq.Encode(original)
	decoded := sq.Decode(encoded)

	// Check dimensions
	if len(encoded) != dim {
		t.Errorf("expected encoded length %d, got %d", dim, len(encoded))
	}
	if len(decoded) != dim {
		t.Errorf("expected decoded length %d, got %d", dim, len(decoded))
	}

	// Check reconstruction error is reasonable
	var totalError float32
	for i := range original {
		diff := original[i] - decoded[i]
		totalError += diff * diff
	}
	avgError := totalError / float32(dim)
	if avgError > 0.01 { // Allow small reconstruction error
		t.Errorf("average reconstruction error too high: %f", avgError)
	}
}

func TestScalarQuantizerWithTraining(t *testing.T) {
	dim := 64
	sq := NewScalarQuantizer(dim)

	// Generate training data
	trainingData := make([][]float32, 100)
	for i := range trainingData {
		trainingData[i] = make([]float32, dim)
		for j := range trainingData[i] {
			trainingData[i][j] = float32(i+j) / 100.0 // Range [0, ~1.5]
		}
	}

	// Train
	sq.Train(trainingData)

	// Test encoding/decoding
	original := trainingData[50]
	encoded := sq.Encode(original)
	decoded := sq.Decode(encoded)

	// Check reconstruction
	var totalError float32
	for i := range original {
		diff := original[i] - decoded[i]
		totalError += diff * diff
	}
	avgError := totalError / float32(dim)
	if avgError > 0.01 {
		t.Errorf("average reconstruction error too high after training: %f", avgError)
	}
}

func TestScalarQuantizerDistanceSquared(t *testing.T) {
	dim := 32
	sq := NewScalarQuantizer(dim)

	// Create two vectors
	a := make([]float32, dim)
	b := make([]float32, dim)
	for i := range a {
		a[i] = float32(i) / float32(dim)
		b[i] = float32(i+1) / float32(dim)
	}

	// Encode b
	encodedB := sq.Encode(b)

	// Compute distance
	dist := sq.DistanceSquared(a, encodedB)

	// Compare with actual distance
	var actualDist float32
	decodedB := sq.Decode(encodedB)
	for i := range a {
		diff := a[i] - decodedB[i]
		actualDist += diff * diff
	}

	// They should be very close
	if math.Abs(float64(dist-actualDist)) > 0.1 {
		t.Errorf("distance mismatch: approximate=%f, actual=%f", dist, actualDist)
	}
}

func TestProductQuantizer(t *testing.T) {
	dim := 64
	numSubvecs := 8
	pq := NewProductQuantizer(dim, numSubvecs)

	// Generate training data
	trainingData := make([][]float32, 200)
	for i := range trainingData {
		trainingData[i] = make([]float32, dim)
		for j := range trainingData[i] {
			trainingData[i][j] = float32(i%10)*0.1 + float32(j%10)*0.01
		}
	}

	// Train
	pq.Train(trainingData)

	// Test encoding/decoding
	original := trainingData[100]
	encoded := pq.Encode(original)
	decoded := pq.Decode(encoded)

	// Check dimensions
	if len(encoded) != numSubvecs {
		t.Errorf("expected encoded length %d, got %d", numSubvecs, len(encoded))
	}
	if len(decoded) != dim {
		t.Errorf("expected decoded length %d, got %d", dim, len(decoded))
	}

	// PQ has higher reconstruction error than SQ, but should still be reasonable
	var totalError float32
	for i := range original {
		diff := original[i] - decoded[i]
		totalError += diff * diff
	}
	avgError := totalError / float32(dim)
	// PQ allows more error due to clustering
	if avgError > 0.5 {
		t.Errorf("average reconstruction error too high: %f", avgError)
	}
}

func TestBinaryQuantizer(t *testing.T) {
	dim := 128
	bq := NewBinaryQuantizer(dim)

	original := make([]float32, dim)
	for i := range original {
		if i%2 == 0 {
			original[i] = 1.0
		} else {
			original[i] = -1.0
		}
	}

	encoded := bq.Encode(original)
	decoded := bq.Decode(encoded)

	// Check encoding size
	expectedBytes := (dim + 7) / 8
	if len(encoded) != expectedBytes {
		t.Errorf("expected %d bytes, got %d", expectedBytes, len(encoded))
	}

	// Binary quantizer should perfectly reconstruct sign pattern
	for i := range original {
		if (original[i] > 0) != (decoded[i] > 0) {
			t.Errorf("sign mismatch at index %d", i)
		}
	}
}

func TestBinaryQuantizerDistance(t *testing.T) {
	dim := 64
	bq := NewBinaryQuantizer(dim)

	// Two identical vectors (in sign)
	a := make([]float32, dim)
	b := make([]float32, dim)
	for i := range a {
		a[i] = 1.0
		b[i] = 0.5 // Same sign
	}

	encodedB := bq.Encode(b)
	dist := bq.DistanceSquared(a, encodedB)
	if dist != 0 {
		t.Errorf("expected distance 0 for same-sign vectors, got %f", dist)
	}

	// Different signs
	for i := range b {
		b[i] = -1.0
	}
	encodedB = bq.Encode(b)
	dist = bq.DistanceSquared(a, encodedB)
	if dist != float32(dim) {
		t.Errorf("expected distance %d for opposite vectors, got %f", dim, dist)
	}
}

func BenchmarkScalarQuantizerEncode(b *testing.B) {
	dim := 1024
	sq := NewScalarQuantizer(dim)
	vector := make([]float32, dim)
	for i := range vector {
		vector[i] = float32(i) / float32(dim)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sq.Encode(vector)
	}
}

func BenchmarkScalarQuantizerDecode(b *testing.B) {
	dim := 1024
	sq := NewScalarQuantizer(dim)
	vector := make([]float32, dim)
	for i := range vector {
		vector[i] = float32(i) / float32(dim)
	}
	encoded := sq.Encode(vector)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sq.Decode(encoded)
	}
}
