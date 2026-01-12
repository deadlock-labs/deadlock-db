// Package filter provides metadata filtering capabilities for hybrid search.
package filter

import (
	"fmt"
	"strings"
)

// Operator represents a comparison operator.
type Operator string

const (
	// Comparison operators
	OpEq  Operator = "$eq"
	OpNe  Operator = "$ne"
	OpGt  Operator = "$gt"
	OpGte Operator = "$gte"
	OpLt  Operator = "$lt"
	OpLte Operator = "$lte"
	OpIn  Operator = "$in"
	OpNin Operator = "$nin"

	// Logical operators
	OpAnd Operator = "$and"
	OpOr  Operator = "$or"
	OpNot Operator = "$not"

	// String operators
	OpContains   Operator = "$contains"
	OpStartsWith Operator = "$startswith"
	OpEndsWith   Operator = "$endswith"

	// Array operators
	OpAll Operator = "$all"
)

// Filter represents a filter condition.
type Filter struct {
	Field    string
	Operator Operator
	Value    any
	Children []*Filter // For logical operators
}

// NewFilter creates a simple equality filter.
func NewFilter(field string, value any) *Filter {
	return &Filter{
		Field:    field,
		Operator: OpEq,
		Value:    value,
	}
}

// NewComparisonFilter creates a comparison filter.
func NewComparisonFilter(field string, op Operator, value any) *Filter {
	return &Filter{
		Field:    field,
		Operator: op,
		Value:    value,
	}
}

// And creates an AND filter combining multiple filters.
func And(filters ...*Filter) *Filter {
	return &Filter{
		Operator: OpAnd,
		Children: filters,
	}
}

// Or creates an OR filter combining multiple filters.
func Or(filters ...*Filter) *Filter {
	return &Filter{
		Operator: OpOr,
		Children: filters,
	}
}

// Not creates a NOT filter negating a filter.
func Not(f *Filter) *Filter {
	return &Filter{
		Operator: OpNot,
		Children: []*Filter{f},
	}
}

// Match checks if the given metadata matches the filter.
func (f *Filter) Match(metadata map[string]any) bool {
	if f == nil {
		return true
	}

	switch f.Operator {
	case OpAnd:
		for _, child := range f.Children {
			if !child.Match(metadata) {
				return false
			}
		}
		return true

	case OpOr:
		for _, child := range f.Children {
			if child.Match(metadata) {
				return true
			}
		}
		return false

	case OpNot:
		if len(f.Children) > 0 {
			return !f.Children[0].Match(metadata)
		}
		return true

	default:
		return f.matchField(metadata)
	}
}

// matchField matches a single field against the filter.
func (f *Filter) matchField(metadata map[string]any) bool {
	value, exists := metadata[f.Field]

	switch f.Operator {
	case OpEq:
		if !exists {
			return f.Value == nil
		}
		return compareEqual(value, f.Value)

	case OpNe:
		if !exists {
			return f.Value != nil
		}
		return !compareEqual(value, f.Value)

	case OpGt:
		if !exists {
			return false
		}
		return compareNumeric(value, f.Value) > 0

	case OpGte:
		if !exists {
			return false
		}
		return compareNumeric(value, f.Value) >= 0

	case OpLt:
		if !exists {
			return false
		}
		return compareNumeric(value, f.Value) < 0

	case OpLte:
		if !exists {
			return false
		}
		return compareNumeric(value, f.Value) <= 0

	case OpIn:
		if !exists {
			return false
		}
		return matchIn(value, f.Value)

	case OpNin:
		if !exists {
			return true
		}
		return !matchIn(value, f.Value)

	case OpContains:
		if !exists {
			return false
		}
		return matchContains(value, f.Value)

	case OpStartsWith:
		if !exists {
			return false
		}
		return matchStartsWith(value, f.Value)

	case OpEndsWith:
		if !exists {
			return false
		}
		return matchEndsWith(value, f.Value)

	case OpAll:
		if !exists {
			return false
		}
		return matchAll(value, f.Value)

	default:
		return false
	}
}

// compareEqual compares two values for equality.
func compareEqual(a, b any) bool {
	// Handle type conversions
	switch av := a.(type) {
	case int:
		return compareNumericEqual(float64(av), b)
	case int32:
		return compareNumericEqual(float64(av), b)
	case int64:
		return compareNumericEqual(float64(av), b)
	case float32:
		return compareNumericEqual(float64(av), b)
	case float64:
		return compareNumericEqual(av, b)
	case string:
		if bv, ok := b.(string); ok {
			return av == bv
		}
	case bool:
		if bv, ok := b.(bool); ok {
			return av == bv
		}
	}
	return fmt.Sprintf("%v", a) == fmt.Sprintf("%v", b)
}

// compareNumericEqual compares a float64 with another value.
func compareNumericEqual(av float64, b any) bool {
	switch bv := b.(type) {
	case int:
		return av == float64(bv)
	case int32:
		return av == float64(bv)
	case int64:
		return av == float64(bv)
	case float32:
		return av == float64(bv)
	case float64:
		return av == bv
	}
	return false
}

// compareNumeric compares two values numerically.
// Returns -1 if a < b, 0 if a == b, 1 if a > b.
func compareNumeric(a, b any) int {
	af := toFloat64(a)
	bf := toFloat64(b)

	if af < bf {
		return -1
	}
	if af > bf {
		return 1
	}
	return 0
}

// toFloat64 converts a value to float64.
func toFloat64(v any) float64 {
	switch vv := v.(type) {
	case int:
		return float64(vv)
	case int32:
		return float64(vv)
	case int64:
		return float64(vv)
	case float32:
		return float64(vv)
	case float64:
		return vv
	}
	return 0
}

// matchIn checks if a value is in a list.
func matchIn(value any, list any) bool {
	switch l := list.(type) {
	case []any:
		for _, item := range l {
			if compareEqual(value, item) {
				return true
			}
		}
	case []string:
		valueStr := fmt.Sprintf("%v", value)
		for _, item := range l {
			if valueStr == item {
				return true
			}
		}
	case []int:
		for _, item := range l {
			if compareEqual(value, item) {
				return true
			}
		}
	case []float64:
		for _, item := range l {
			if compareEqual(value, item) {
				return true
			}
		}
	}
	return false
}

// matchContains checks if a string contains a substring.
func matchContains(value any, substr any) bool {
	vs := fmt.Sprintf("%v", value)
	ss := fmt.Sprintf("%v", substr)
	return strings.Contains(strings.ToLower(vs), strings.ToLower(ss))
}

// matchStartsWith checks if a string starts with a prefix.
func matchStartsWith(value any, prefix any) bool {
	vs := fmt.Sprintf("%v", value)
	ps := fmt.Sprintf("%v", prefix)
	return strings.HasPrefix(strings.ToLower(vs), strings.ToLower(ps))
}

// matchEndsWith checks if a string ends with a suffix.
func matchEndsWith(value any, suffix any) bool {
	vs := fmt.Sprintf("%v", value)
	ss := fmt.Sprintf("%v", suffix)
	return strings.HasSuffix(strings.ToLower(vs), strings.ToLower(ss))
}

// matchAll checks if all elements in the filter are present in the value.
func matchAll(value any, required any) bool {
	valueSlice, ok := toSlice(value)
	if !ok {
		return false
	}
	requiredSlice, ok := toSlice(required)
	if !ok {
		return false
	}

	for _, r := range requiredSlice {
		found := false
		for _, v := range valueSlice {
			if compareEqual(v, r) {
				found = true
				break
			}
		}
		if !found {
			return false
		}
	}
	return true
}

// toSlice converts a value to a slice of any.
func toSlice(v any) ([]any, bool) {
	switch vv := v.(type) {
	case []any:
		return vv, true
	case []string:
		result := make([]any, len(vv))
		for i, s := range vv {
			result[i] = s
		}
		return result, true
	case []int:
		result := make([]any, len(vv))
		for i, n := range vv {
			result[i] = n
		}
		return result, true
	case []float64:
		result := make([]any, len(vv))
		for i, n := range vv {
			result[i] = n
		}
		return result, true
	}
	return nil, false
}

// ParseFilter parses a filter from a map representation.
func ParseFilter(m map[string]any) (*Filter, error) {
	if m == nil {
		return nil, nil
	}

	// Check for logical operators
	if and, ok := m["$and"]; ok {
		children, err := parseFilterList(and)
		if err != nil {
			return nil, err
		}
		return And(children...), nil
	}

	if or, ok := m["$or"]; ok {
		children, err := parseFilterList(or)
		if err != nil {
			return nil, err
		}
		return Or(children...), nil
	}

	if not, ok := m["$not"]; ok {
		if notMap, ok := not.(map[string]any); ok {
			child, err := ParseFilter(notMap)
			if err != nil {
				return nil, err
			}
			return Not(child), nil
		}
		return nil, fmt.Errorf("$not value must be an object")
	}

	// Single field comparison
	if len(m) == 1 {
		for field, value := range m {
			return parseFieldFilter(field, value)
		}
	}

	// Multiple fields = implicit AND
	children := make([]*Filter, 0, len(m))
	for field, value := range m {
		f, err := parseFieldFilter(field, value)
		if err != nil {
			return nil, err
		}
		children = append(children, f)
	}
	return And(children...), nil
}

// parseFilterList parses a list of filters.
func parseFilterList(v any) ([]*Filter, error) {
	list, ok := v.([]any)
	if !ok {
		return nil, fmt.Errorf("filter list must be an array")
	}

	filters := make([]*Filter, 0, len(list))
	for _, item := range list {
		if itemMap, ok := item.(map[string]any); ok {
			f, err := ParseFilter(itemMap)
			if err != nil {
				return nil, err
			}
			filters = append(filters, f)
		} else {
			return nil, fmt.Errorf("filter list item must be an object")
		}
	}
	return filters, nil
}

// parseFieldFilter parses a filter for a single field.
func parseFieldFilter(field string, value any) (*Filter, error) {
	// Check if value is an operator object
	if valueMap, ok := value.(map[string]any); ok {
		for op, opValue := range valueMap {
			return &Filter{
				Field:    field,
				Operator: Operator(op),
				Value:    opValue,
			}, nil
		}
	}

	// Simple equality
	return NewFilter(field, value), nil
}
