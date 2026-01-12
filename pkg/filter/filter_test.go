package filter

import (
	"testing"
)

func TestFilterEquality(t *testing.T) {
	f := NewFilter("category", "tech")

	tests := []struct {
		name     string
		metadata map[string]any
		expected bool
	}{
		{
			name:     "match",
			metadata: map[string]any{"category": "tech"},
			expected: true,
		},
		{
			name:     "no match",
			metadata: map[string]any{"category": "science"},
			expected: false,
		},
		{
			name:     "missing field",
			metadata: map[string]any{"other": "value"},
			expected: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if f.Match(tt.metadata) != tt.expected {
				t.Errorf("expected %v, got %v", tt.expected, f.Match(tt.metadata))
			}
		})
	}
}

func TestFilterComparison(t *testing.T) {
	tests := []struct {
		name     string
		filter   *Filter
		metadata map[string]any
		expected bool
	}{
		{
			name:     "greater than - true",
			filter:   NewComparisonFilter("age", OpGt, 20),
			metadata: map[string]any{"age": 25},
			expected: true,
		},
		{
			name:     "greater than - false",
			filter:   NewComparisonFilter("age", OpGt, 20),
			metadata: map[string]any{"age": 15},
			expected: false,
		},
		{
			name:     "less than",
			filter:   NewComparisonFilter("price", OpLt, 100),
			metadata: map[string]any{"price": 50},
			expected: true,
		},
		{
			name:     "greater than or equal",
			filter:   NewComparisonFilter("score", OpGte, 90),
			metadata: map[string]any{"score": 90},
			expected: true,
		},
		{
			name:     "not equal",
			filter:   NewComparisonFilter("status", OpNe, "deleted"),
			metadata: map[string]any{"status": "active"},
			expected: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.filter.Match(tt.metadata) != tt.expected {
				t.Errorf("expected %v", tt.expected)
			}
		})
	}
}

func TestFilterIn(t *testing.T) {
	f := NewComparisonFilter("category", OpIn, []string{"tech", "science", "health"})

	tests := []struct {
		metadata map[string]any
		expected bool
	}{
		{map[string]any{"category": "tech"}, true},
		{map[string]any{"category": "arts"}, false},
		{map[string]any{"category": "science"}, true},
	}

	for i, tt := range tests {
		if f.Match(tt.metadata) != tt.expected {
			t.Errorf("test %d: expected %v", i, tt.expected)
		}
	}
}

func TestFilterAnd(t *testing.T) {
	f := And(
		NewFilter("category", "tech"),
		NewComparisonFilter("year", OpGte, 2020),
	)

	tests := []struct {
		name     string
		metadata map[string]any
		expected bool
	}{
		{
			name:     "both conditions met",
			metadata: map[string]any{"category": "tech", "year": 2023},
			expected: true,
		},
		{
			name:     "first condition not met",
			metadata: map[string]any{"category": "science", "year": 2023},
			expected: false,
		},
		{
			name:     "second condition not met",
			metadata: map[string]any{"category": "tech", "year": 2018},
			expected: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if f.Match(tt.metadata) != tt.expected {
				t.Errorf("expected %v", tt.expected)
			}
		})
	}
}

func TestFilterOr(t *testing.T) {
	f := Or(
		NewFilter("category", "tech"),
		NewFilter("category", "science"),
	)

	tests := []struct {
		metadata map[string]any
		expected bool
	}{
		{map[string]any{"category": "tech"}, true},
		{map[string]any{"category": "science"}, true},
		{map[string]any{"category": "arts"}, false},
	}

	for i, tt := range tests {
		if f.Match(tt.metadata) != tt.expected {
			t.Errorf("test %d: expected %v", i, tt.expected)
		}
	}
}

func TestFilterNot(t *testing.T) {
	f := Not(NewFilter("status", "deleted"))

	tests := []struct {
		metadata map[string]any
		expected bool
	}{
		{map[string]any{"status": "active"}, true},
		{map[string]any{"status": "deleted"}, false},
	}

	for i, tt := range tests {
		if f.Match(tt.metadata) != tt.expected {
			t.Errorf("test %d: expected %v", i, tt.expected)
		}
	}
}

func TestFilterContains(t *testing.T) {
	f := NewComparisonFilter("title", OpContains, "go")

	tests := []struct {
		metadata map[string]any
		expected bool
	}{
		{map[string]any{"title": "Learning Go Programming"}, true},
		{map[string]any{"title": "Python Tutorial"}, false},
		{map[string]any{"title": "GOLANG basics"}, true}, // Case insensitive
	}

	for i, tt := range tests {
		if f.Match(tt.metadata) != tt.expected {
			t.Errorf("test %d: expected %v", i, tt.expected)
		}
	}
}

func TestParseFilter(t *testing.T) {
	tests := []struct {
		name     string
		input    map[string]any
		metadata map[string]any
		expected bool
	}{
		{
			name:     "simple equality",
			input:    map[string]any{"category": "tech"},
			metadata: map[string]any{"category": "tech"},
			expected: true,
		},
		{
			name:     "operator",
			input:    map[string]any{"score": map[string]any{"$gt": 80}},
			metadata: map[string]any{"score": 90},
			expected: true,
		},
		{
			name: "and",
			input: map[string]any{
				"$and": []any{
					map[string]any{"category": "tech"},
					map[string]any{"year": map[string]any{"$gte": 2020}},
				},
			},
			metadata: map[string]any{"category": "tech", "year": 2023},
			expected: true,
		},
		{
			name: "or",
			input: map[string]any{
				"$or": []any{
					map[string]any{"category": "tech"},
					map[string]any{"category": "science"},
				},
			},
			metadata: map[string]any{"category": "science"},
			expected: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			f, err := ParseFilter(tt.input)
			if err != nil {
				t.Fatalf("parse error: %v", err)
			}
			if f.Match(tt.metadata) != tt.expected {
				t.Errorf("expected %v", tt.expected)
			}
		})
	}
}

func TestNilFilter(t *testing.T) {
	var f *Filter
	if !f.Match(map[string]any{"any": "thing"}) {
		t.Error("nil filter should match everything")
	}
}
