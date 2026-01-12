package distance

import (
	"math"
	"testing"
)

func TestCosineSimilarity(t *testing.T) {
	tests := []struct {
		name     string
		a        []float32
		b        []float32
		expected float32
	}{
		{
			name:     "identical vectors",
			a:        []float32{1, 0, 0},
			b:        []float32{1, 0, 0},
			expected: 1.0,
		},
		{
			name:     "orthogonal vectors",
			a:        []float32{1, 0, 0},
			b:        []float32{0, 1, 0},
			expected: 0.0,
		},
		{
			name:     "opposite vectors",
			a:        []float32{1, 0, 0},
			b:        []float32{-1, 0, 0},
			expected: -1.0,
		},
		{
			name:     "similar vectors",
			a:        []float32{1, 2, 3},
			b:        []float32{2, 4, 6},
			expected: 1.0, // Same direction, different magnitude
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := CosineSimilarity(tt.a, tt.b)
			if math.Abs(float64(result-tt.expected)) > 0.0001 {
				t.Errorf("CosineSimilarity(%v, %v) = %f, expected %f", tt.a, tt.b, result, tt.expected)
			}
		})
	}
}

func TestEuclideanDistance(t *testing.T) {
	tests := []struct {
		name     string
		a        []float32
		b        []float32
		expected float32
	}{
		{
			name:     "same point",
			a:        []float32{0, 0, 0},
			b:        []float32{0, 0, 0},
			expected: 0.0,
		},
		{
			name:     "unit distance",
			a:        []float32{0, 0, 0},
			b:        []float32{1, 0, 0},
			expected: 1.0,
		},
		{
			name:     "3-4-5 triangle",
			a:        []float32{0, 0},
			b:        []float32{3, 4},
			expected: 5.0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := EuclideanDistance(tt.a, tt.b)
			if math.Abs(float64(result-tt.expected)) > 0.0001 {
				t.Errorf("EuclideanDistance(%v, %v) = %f, expected %f", tt.a, tt.b, result, tt.expected)
			}
		})
	}
}

func TestDotProductValue(t *testing.T) {
	tests := []struct {
		name     string
		a        []float32
		b        []float32
		expected float32
	}{
		{
			name:     "orthogonal",
			a:        []float32{1, 0},
			b:        []float32{0, 1},
			expected: 0.0,
		},
		{
			name:     "parallel",
			a:        []float32{1, 2, 3},
			b:        []float32{1, 2, 3},
			expected: 14.0,
		},
		{
			name:     "simple",
			a:        []float32{1, 2},
			b:        []float32{3, 4},
			expected: 11.0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := DotProductValue(tt.a, tt.b)
			if math.Abs(float64(result-tt.expected)) > 0.0001 {
				t.Errorf("DotProductValue(%v, %v) = %f, expected %f", tt.a, tt.b, result, tt.expected)
			}
		})
	}
}

func TestManhattanDistance(t *testing.T) {
	tests := []struct {
		name     string
		a        []float32
		b        []float32
		expected float32
	}{
		{
			name:     "same point",
			a:        []float32{0, 0},
			b:        []float32{0, 0},
			expected: 0.0,
		},
		{
			name:     "simple",
			a:        []float32{0, 0},
			b:        []float32{3, 4},
			expected: 7.0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := ManhattanDistance(tt.a, tt.b)
			if math.Abs(float64(result-tt.expected)) > 0.0001 {
				t.Errorf("ManhattanDistance(%v, %v) = %f, expected %f", tt.a, tt.b, result, tt.expected)
			}
		})
	}
}

func TestCalculator(t *testing.T) {
	metrics := []Metric{Cosine, Euclidean, DotProduct, Manhattan}

	for _, m := range metrics {
		calc := NewCalculator(m)
		if calc.Metric() != m {
			t.Errorf("Calculator metric mismatch: expected %v, got %v", m, calc.Metric())
		}
	}
}

func BenchmarkCosineSimilarity(b *testing.B) {
	dim := 1024
	a := make([]float32, dim)
	bv := make([]float32, dim)
	for i := 0; i < dim; i++ {
		a[i] = float32(i)
		bv[i] = float32(dim - i)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		CosineSimilarity(a, bv)
	}
}

func BenchmarkEuclideanDistance(b *testing.B) {
	dim := 1024
	a := make([]float32, dim)
	bv := make([]float32, dim)
	for i := 0; i < dim; i++ {
		a[i] = float32(i)
		bv[i] = float32(dim - i)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		EuclideanDistance(a, bv)
	}
}

func BenchmarkDotProduct(b *testing.B) {
	dim := 1024
	a := make([]float32, dim)
	bv := make([]float32, dim)
	for i := 0; i < dim; i++ {
		a[i] = float32(i)
		bv[i] = float32(dim - i)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		DotProductValue(a, bv)
	}
}
