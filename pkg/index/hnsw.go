// Package index provides indexing algorithms for fast approximate nearest neighbor search.
package index

import (
	"container/heap"
	"math"
	"math/rand"
	"sync"
	"sync/atomic"

	"github.com/deadlock-labs/deadlock-db/pkg/core"
	"github.com/deadlock-labs/deadlock-db/pkg/distance"
)

// HNSWConfig contains configuration for the HNSW index.
type HNSWConfig struct {
	// M is the maximum number of connections per node per layer.
	M int
	// EfConstruction is the size of the dynamic candidate list during construction.
	EfConstruction int
	// Ml is the level multiplier (controls layer distribution).
	Ml float64
	// MaxLevel is the maximum level in the graph.
	MaxLevel int
	// Metric is the distance metric to use.
	Metric distance.Metric
	// Dimension is the vector dimension.
	Dimension int
	// UseHeuristic enables the improved RNG-based neighbor selection heuristic.
	// This improves recall at the cost of slightly slower insertion.
	UseHeuristic bool
}

// DefaultHNSWConfig returns the default HNSW configuration.
func DefaultHNSWConfig(dimension int) *HNSWConfig {
	return &HNSWConfig{
		M:              16,
		EfConstruction: 200,
		Ml:             1.0 / math.Log(16),
		MaxLevel:       16,
		Metric:         distance.Cosine,
		Dimension:      dimension,
		UseHeuristic:   true, // Enable improved neighbor selection by default
	}
}

// HNSWNode represents a node in the HNSW graph.
type HNSWNode struct {
	ID          uint64
	VectorID    string
	Vector      []float32
	Connections [][]uint64 // Connections per layer
	mu          sync.RWMutex
}

// NewHNSWNode creates a new HNSW node.
func NewHNSWNode(id uint64, vectorID string, vector []float32, maxLevel int) *HNSWNode {
	return &HNSWNode{
		ID:          id,
		VectorID:    vectorID,
		Vector:      vector,
		Connections: make([][]uint64, maxLevel+1),
	}
}

// HNSW implements the Hierarchical Navigable Small World graph for ANN search.
type HNSW struct {
	config     *HNSWConfig
	nodes      map[uint64]*HNSWNode
	vectorToID map[string]uint64
	entryPoint uint64
	maxLayer   int
	size       uint64
	nextID     uint64
	calc       distance.Calculator
	mu         sync.RWMutex
}

// NewHNSW creates a new HNSW index with the given configuration.
func NewHNSW(config *HNSWConfig) *HNSW {
	if config == nil {
		config = DefaultHNSWConfig(128)
	}
	return &HNSW{
		config:     config,
		nodes:      make(map[uint64]*HNSWNode),
		vectorToID: make(map[string]uint64),
		entryPoint: 0,
		maxLayer:   -1,
		calc:       distance.NewCalculator(config.Metric),
	}
}

// Size returns the number of vectors in the index.
func (h *HNSW) Size() int {
	return int(atomic.LoadUint64(&h.size))
}

// randomLevel generates a random level for a new node.
func (h *HNSW) randomLevel() int {
	level := 0
	for level < h.config.MaxLevel && rand.Float64() < h.config.Ml {
		level++
	}
	return level
}

// Insert adds a vector to the index.
func (h *HNSW) Insert(v *core.Vector) error {
	h.mu.Lock()
	defer h.mu.Unlock()

	// Check if vector already exists
	if _, exists := h.vectorToID[v.ID]; exists {
		return h.update(v)
	}

	nodeID := atomic.AddUint64(&h.nextID, 1) - 1
	level := h.randomLevel()
	node := NewHNSWNode(nodeID, v.ID, v.Values, level)

	// If this is the first node
	if h.size == 0 {
		h.nodes[nodeID] = node
		h.vectorToID[v.ID] = nodeID
		h.entryPoint = nodeID
		h.maxLayer = level
		atomic.AddUint64(&h.size, 1)
		return nil
	}

	// Find entry point for each layer
	ep := h.entryPoint
	epNode := h.nodes[ep]

	// Start from top layer and go down
	for lc := h.maxLayer; lc > level; lc-- {
		changed := true
		for changed {
			changed = false
			if epNode.Connections == nil || lc >= len(epNode.Connections) {
				break
			}
			for _, neighborID := range epNode.Connections[lc] {
				neighbor := h.nodes[neighborID]
				if neighbor == nil {
					continue
				}
				if h.calc.Distance(v.Values, neighbor.Vector) < h.calc.Distance(v.Values, epNode.Vector) {
					ep = neighborID
					epNode = neighbor
					changed = true
				}
			}
		}
	}

	// Insert at each layer from level down to 0
	for lc := min(level, h.maxLayer); lc >= 0; lc-- {
		neighbors := h.searchLayer(v.Values, ep, h.config.EfConstruction, lc)
		selectedNeighbors := h.selectNeighbors(v.Values, neighbors, h.config.M, lc)

		// Connect the new node to selected neighbors
		node.Connections[lc] = make([]uint64, len(selectedNeighbors))
		for i, n := range selectedNeighbors {
			node.Connections[lc][i] = n.nodeID
		}

		// Connect neighbors back to the new node
		for _, n := range selectedNeighbors {
			neighbor := h.nodes[n.nodeID]
			if neighbor == nil {
				continue
			}
			neighbor.mu.Lock()
			if lc < len(neighbor.Connections) {
				neighbor.Connections[lc] = append(neighbor.Connections[lc], nodeID)
				// Prune if too many connections
				if len(neighbor.Connections[lc]) > h.config.M*2 {
					neighbor.Connections[lc] = h.pruneConnections(neighbor, lc, h.config.M*2)
				}
			}
			neighbor.mu.Unlock()
		}

		if len(neighbors) > 0 {
			ep = neighbors[0].nodeID
		}
	}

	h.nodes[nodeID] = node
	h.vectorToID[v.ID] = nodeID

	if level > h.maxLayer {
		h.maxLayer = level
		h.entryPoint = nodeID
	}

	atomic.AddUint64(&h.size, 1)
	return nil
}

// update replaces an existing vector in the index.
func (h *HNSW) update(v *core.Vector) error {
	nodeID, exists := h.vectorToID[v.ID]
	if !exists {
		return nil
	}
	node := h.nodes[nodeID]
	if node != nil {
		node.Vector = v.Values
	}
	return nil
}

// Delete removes a vector from the index.
func (h *HNSW) Delete(vectorID string) error {
	h.mu.Lock()
	defer h.mu.Unlock()

	nodeID, exists := h.vectorToID[vectorID]
	if !exists {
		return nil
	}

	node := h.nodes[nodeID]
	if node == nil {
		return nil
	}

	// Remove connections to this node from all neighbors
	for lc := 0; lc < len(node.Connections); lc++ {
		for _, neighborID := range node.Connections[lc] {
			neighbor := h.nodes[neighborID]
			if neighbor == nil {
				continue
			}
			neighbor.mu.Lock()
			if lc < len(neighbor.Connections) {
				newConnections := make([]uint64, 0, len(neighbor.Connections[lc]))
				for _, id := range neighbor.Connections[lc] {
					if id != nodeID {
						newConnections = append(newConnections, id)
					}
				}
				neighbor.Connections[lc] = newConnections
			}
			neighbor.mu.Unlock()
		}
	}

	delete(h.nodes, nodeID)
	delete(h.vectorToID, vectorID)
	atomic.AddUint64(&h.size, ^uint64(0)) // Decrement

	// Update entry point if necessary
	if h.entryPoint == nodeID && len(h.nodes) > 0 {
		for id := range h.nodes {
			h.entryPoint = id
			break
		}
	}

	return nil
}

// candidate represents a candidate node during search.
type candidate struct {
	nodeID   uint64
	distance float32
}

// candidateHeap implements a min-heap for candidates.
type candidateHeap []candidate

func (h candidateHeap) Len() int           { return len(h) }
func (h candidateHeap) Less(i, j int) bool { return h[i].distance < h[j].distance }
func (h candidateHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }

func (h *candidateHeap) Push(x any) {
	*h = append(*h, x.(candidate))
}

func (h *candidateHeap) Pop() any {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[0 : n-1]
	return x
}

// maxCandidateHeap implements a max-heap for candidates (for ef results).
type maxCandidateHeap []candidate

func (h maxCandidateHeap) Len() int           { return len(h) }
func (h maxCandidateHeap) Less(i, j int) bool { return h[i].distance > h[j].distance }
func (h maxCandidateHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }

func (h *maxCandidateHeap) Push(x any) {
	*h = append(*h, x.(candidate))
}

func (h *maxCandidateHeap) Pop() any {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[0 : n-1]
	return x
}

// searchLayer performs a greedy search at a single layer.
func (h *HNSW) searchLayer(query []float32, ep uint64, ef int, layer int) []candidate {
	epNode := h.nodes[ep]
	if epNode == nil {
		return nil
	}

	visited := make(map[uint64]bool)
	visited[ep] = true

	candidates := &candidateHeap{}
	results := &maxCandidateHeap{}

	d := h.calc.Distance(query, epNode.Vector)
	heap.Push(candidates, candidate{ep, d})
	heap.Push(results, candidate{ep, d})

	for candidates.Len() > 0 {
		c := heap.Pop(candidates).(candidate)

		// Check if we've found enough results
		if results.Len() > 0 && c.distance > (*results)[0].distance {
			break
		}

		cNode := h.nodes[c.nodeID]
		if cNode == nil || layer >= len(cNode.Connections) {
			continue
		}

		for _, neighborID := range cNode.Connections[layer] {
			if visited[neighborID] {
				continue
			}
			visited[neighborID] = true

			neighbor := h.nodes[neighborID]
			if neighbor == nil {
				continue
			}

			dist := h.calc.Distance(query, neighbor.Vector)

			if results.Len() < ef || dist < (*results)[0].distance {
				heap.Push(candidates, candidate{neighborID, dist})
				heap.Push(results, candidate{neighborID, dist})
				if results.Len() > ef {
					heap.Pop(results)
				}
			}
		}
	}

	// Convert results to slice
	result := make([]candidate, results.Len())
	for i := results.Len() - 1; i >= 0; i-- {
		result[i] = heap.Pop(results).(candidate)
	}

	return result
}

// selectNeighbors selects the best neighbors for a node.
// Uses RNG-based heuristic when enabled for better graph quality.
func (h *HNSW) selectNeighbors(query []float32, candidates []candidate, m int, _ int) []candidate {
	if len(candidates) <= m {
		return candidates
	}

	if !h.config.UseHeuristic {
		// Simple selection: just take top M by distance
		selected := make([]candidate, 0, m)
		for _, c := range candidates {
			if len(selected) >= m {
				break
			}
			selected = append(selected, c)
		}
		return selected
	}

	// RNG-based heuristic: select neighbors that are close to query
	// but not too close to each other, promoting graph diversity
	return h.selectNeighborsHeuristic(query, candidates, m)
}

// selectNeighborsHeuristic implements the RNG-based neighbor selection.
// This produces a more navigable graph by avoiding clusters of similar neighbors.
func (h *HNSW) selectNeighborsHeuristic(query []float32, candidates []candidate, m int) []candidate {
	if len(candidates) == 0 {
		return nil
	}

	selected := make([]candidate, 0, m)
	// Working set starts with all candidates
	working := make([]candidate, len(candidates))
	copy(working, candidates)

	for len(selected) < m && len(working) > 0 {
		// Find the closest candidate to query
		minIdx := 0
		for i := 1; i < len(working); i++ {
			if working[i].distance < working[minIdx].distance {
				minIdx = i
			}
		}
		closest := working[minIdx]

		// Remove from working set
		working[minIdx] = working[len(working)-1]
		working = working[:len(working)-1]

		// Check if this candidate is a good addition (RNG condition)
		// A candidate is good if it's not dominated by existing selected neighbors
		good := true
		closestNode := h.nodes[closest.nodeID]
		if closestNode != nil {
			for _, sel := range selected {
				selNode := h.nodes[sel.nodeID]
				if selNode == nil {
					continue
				}
				// If distance(closest, sel) < distance(query, closest),
				// then 'sel' dominates 'closest' for this query
				distToSel := h.calc.Distance(closestNode.Vector, selNode.Vector)
				if distToSel < closest.distance {
					good = false
					break
				}
			}
		}

		if good {
			selected = append(selected, closest)
		}
	}

	// If we couldn't fill M neighbors with the heuristic, fill with remaining best
	if len(selected) < m && len(candidates) > len(selected) {
		for _, c := range candidates {
			if len(selected) >= m {
				break
			}
			found := false
			for _, s := range selected {
				if s.nodeID == c.nodeID {
					found = true
					break
				}
			}
			if !found {
				selected = append(selected, c)
			}
		}
	}

	return selected
}

// pruneConnections reduces the number of connections to maxConnections.
// Uses the same heuristic as selectNeighbors for consistency.
func (h *HNSW) pruneConnections(node *HNSWNode, layer int, maxConnections int) []uint64 {
	if layer >= len(node.Connections) || len(node.Connections[layer]) <= maxConnections {
		return node.Connections[layer]
	}

	// Calculate distances to all neighbors
	type neighborDist struct {
		id   uint64
		dist float32
	}
	neighbors := make([]neighborDist, 0, len(node.Connections[layer]))
	for _, id := range node.Connections[layer] {
		n := h.nodes[id]
		if n == nil {
			continue
		}
		neighbors = append(neighbors, neighborDist{
			id:   id,
			dist: h.calc.Distance(node.Vector, n.Vector),
		})
	}

	// Sort by distance
	for i := 0; i < len(neighbors)-1; i++ {
		for j := i + 1; j < len(neighbors); j++ {
			if neighbors[j].dist < neighbors[i].dist {
				neighbors[i], neighbors[j] = neighbors[j], neighbors[i]
			}
		}
	}

	if !h.config.UseHeuristic {
		// Simple: keep only the closest
		result := make([]uint64, 0, maxConnections)
		for i := 0; i < len(neighbors) && i < maxConnections; i++ {
			result = append(result, neighbors[i].id)
		}
		return result
	}

	// RNG heuristic: select diverse neighbors
	result := make([]uint64, 0, maxConnections)
	for _, neighbor := range neighbors {
		if len(result) >= maxConnections {
			break
		}
		good := true
		neighborNode := h.nodes[neighbor.id]
		if neighborNode != nil {
			for _, selID := range result {
				selNode := h.nodes[selID]
				if selNode == nil {
					continue
				}
				distToSel := h.calc.Distance(neighborNode.Vector, selNode.Vector)
				if distToSel < neighbor.dist {
					good = false
					break
				}
			}
		}
		if good {
			result = append(result, neighbor.id)
		}
	}

	// Fill remaining slots if needed
	if len(result) < maxConnections {
		for _, neighbor := range neighbors {
			if len(result) >= maxConnections {
				break
			}
			found := false
			for _, id := range result {
				if id == neighbor.id {
					found = true
					break
				}
			}
			if !found {
				result = append(result, neighbor.id)
			}
		}
	}

	return result
}

// Search finds the k nearest neighbors to the query vector.
func (h *HNSW) Search(query []float32, k int, ef int) []core.SearchResult {
	h.mu.RLock()
	defer h.mu.RUnlock()

	if h.size == 0 {
		return nil
	}

	if ef < k {
		ef = k
	}

	ep := h.entryPoint
	epNode := h.nodes[ep]
	if epNode == nil {
		return nil
	}

	// Traverse from top to layer 1
	for lc := h.maxLayer; lc > 0; lc-- {
		changed := true
		for changed {
			changed = false
			if epNode.Connections == nil || lc >= len(epNode.Connections) {
				break
			}
			for _, neighborID := range epNode.Connections[lc] {
				neighbor := h.nodes[neighborID]
				if neighbor == nil {
					continue
				}
				if h.calc.Distance(query, neighbor.Vector) < h.calc.Distance(query, epNode.Vector) {
					ep = neighborID
					epNode = neighbor
					changed = true
				}
			}
		}
	}

	// Search at layer 0 with ef
	candidates := h.searchLayer(query, ep, ef, 0)

	// Return top k results
	results := make([]core.SearchResult, 0, k)
	for i := 0; i < len(candidates) && i < k; i++ {
		node := h.nodes[candidates[i].nodeID]
		if node == nil {
			continue
		}
		score := candidates[i].distance
		// Convert distance to similarity score if needed
		if h.config.Metric == distance.DotProduct {
			score = -score // Negate back to positive
		}
		results = append(results, core.SearchResult{
			ID:    node.VectorID,
			Score: score,
		})
	}

	return results
}

// GetVector retrieves a vector by its ID.
func (h *HNSW) GetVector(vectorID string) (*core.Vector, bool) {
	h.mu.RLock()
	defer h.mu.RUnlock()

	nodeID, exists := h.vectorToID[vectorID]
	if !exists {
		return nil, false
	}

	node := h.nodes[nodeID]
	if node == nil {
		return nil, false
	}

	return core.NewVector(vectorID, node.Vector), true
}

// Stats returns statistics about the index.
func (h *HNSW) Stats() map[string]any {
	h.mu.RLock()
	defer h.mu.RUnlock()

	totalConnections := 0
	for _, node := range h.nodes {
		for _, conns := range node.Connections {
			totalConnections += len(conns)
		}
	}

	return map[string]any{
		"size":              h.size,
		"max_layer":         h.maxLayer,
		"entry_point":       h.entryPoint,
		"total_connections": totalConnections,
		"m":                 h.config.M,
		"ef_construction":   h.config.EfConstruction,
		"metric":            h.config.Metric.String(),
		"use_heuristic":     h.config.UseHeuristic,
	}
}

// GraphNode represents a node in the graph visualization.
type GraphNode struct {
	ID          uint64   `json:"id"`
	VectorID    string   `json:"vector_id"`
	MaxLayer    int      `json:"max_layer"`
	Connections []uint64 `json:"connections"`
}

// GraphData returns graph structure data for visualization.
// maxNodes limits the number of nodes returned (0 for all nodes).
func (h *HNSW) GraphData(maxNodes int) map[string]any {
	h.mu.RLock()
	defer h.mu.RUnlock()

	nodes := make([]GraphNode, 0)
	count := 0

	// Get entry point vector ID
	var entryPointVectorID string
	if epNode, ok := h.nodes[h.entryPoint]; ok {
		entryPointVectorID = epNode.VectorID
	}

	for id, node := range h.nodes {
		if maxNodes > 0 && count >= maxNodes {
			break
		}

		// Determine max layer for this node
		maxLayer := 0
		for layer, conns := range node.Connections {
			if len(conns) > 0 && layer > maxLayer {
				maxLayer = layer
			}
		}

		// Get layer 0 connections (base layer has most connections)
		var connections []uint64
		if len(node.Connections) > 0 {
			connections = node.Connections[0]
		}

		nodes = append(nodes, GraphNode{
			ID:          id,
			VectorID:    node.VectorID,
			MaxLayer:    maxLayer,
			Connections: connections,
		})
		count++
	}

	return map[string]any{
		"nodes":       nodes,
		"max_layer":   h.maxLayer,
		"entry_point": entryPointVectorID,
		"total_nodes": h.size,
	}
}

// SearchAdaptive performs a search with adaptive ef parameter.
// It starts with a small ef and increases if the initial results seem poor.
// This provides a good balance between speed and recall.
func (h *HNSW) SearchAdaptive(query []float32, k int, minRecall float32) []core.SearchResult {
	h.mu.RLock()
	defer h.mu.RUnlock()

	if h.size == 0 {
		return nil
	}

	// Start with ef = k * 2
	ef := k * 2
	maxEf := k * 20

	// Do initial search
	results := h.searchInternal(query, k, ef)
	if len(results) < k {
		return results
	}

	// Check if results are "good enough" by looking at score distribution
	// If the worst score is much worse than the best, we might need more exploration
	if len(results) >= 2 {
		bestScore := results[0].Score
		worstScore := results[len(results)-1].Score

		// Calculate score spread (normalized)
		var spread float32
		if bestScore != 0 {
			spread = (worstScore - bestScore) / bestScore
		}

		// If spread is high, increase ef and re-search
		for spread > 0.5 && ef < maxEf {
			ef = ef * 2
			if ef > maxEf {
				ef = maxEf
			}
			results = h.searchInternal(query, k, ef)

			if len(results) >= 2 {
				bestScore = results[0].Score
				worstScore = results[len(results)-1].Score
				if bestScore != 0 {
					spread = (worstScore - bestScore) / bestScore
				} else {
					break
				}
			} else {
				break
			}
		}
	}

	return results
}

// searchInternal performs search without acquiring locks (caller must hold read lock).
func (h *HNSW) searchInternal(query []float32, k int, ef int) []core.SearchResult {
	if ef < k {
		ef = k
	}

	ep := h.entryPoint
	epNode := h.nodes[ep]
	if epNode == nil {
		return nil
	}

	// Traverse from top to layer 1
	for lc := h.maxLayer; lc > 0; lc-- {
		changed := true
		for changed {
			changed = false
			if epNode.Connections == nil || lc >= len(epNode.Connections) {
				break
			}
			for _, neighborID := range epNode.Connections[lc] {
				neighbor := h.nodes[neighborID]
				if neighbor == nil {
					continue
				}
				if h.calc.Distance(query, neighbor.Vector) < h.calc.Distance(query, epNode.Vector) {
					ep = neighborID
					epNode = neighbor
					changed = true
				}
			}
		}
	}

	// Search at layer 0 with ef
	candidates := h.searchLayer(query, ep, ef, 0)

	// Return top k results
	results := make([]core.SearchResult, 0, k)
	for i := 0; i < len(candidates) && i < k; i++ {
		node := h.nodes[candidates[i].nodeID]
		if node == nil {
			continue
		}
		score := candidates[i].distance
		if h.config.Metric == distance.DotProduct {
			score = -score
		}
		results = append(results, core.SearchResult{
			ID:    node.VectorID,
			Score: score,
		})
	}

	return results
}

// InsertBatch inserts multiple vectors in a batch.
// This is more efficient than inserting one by one as it allows for
// better parallelization of the graph construction.
func (h *HNSW) InsertBatch(vectors []*core.Vector) error {
	if len(vectors) == 0 {
		return nil
	}

	// For small batches, use sequential insertion
	if len(vectors) < 100 {
		for _, v := range vectors {
			if err := h.Insert(v); err != nil {
				return err
			}
		}
		return nil
	}

	// For larger batches, insert first vector to initialize,
	// then process remaining in parallel chunks
	if err := h.Insert(vectors[0]); err != nil {
		return err
	}

	// Process remaining vectors in chunks
	remaining := vectors[1:]
	chunkSize := 50
	var wg sync.WaitGroup
	errChan := make(chan error, (len(remaining)/chunkSize)+1)

	for i := 0; i < len(remaining); i += chunkSize {
		end := i + chunkSize
		if end > len(remaining) {
			end = len(remaining)
		}
		chunk := remaining[i:end]

		wg.Add(1)
		go func(vecs []*core.Vector) {
			defer wg.Done()
			for _, v := range vecs {
				if err := h.Insert(v); err != nil {
					select {
					case errChan <- err:
					default:
					}
					return
				}
			}
		}(chunk)
	}

	wg.Wait()
	close(errChan)

	// Return first error if any
	for err := range errChan {
		if err != nil {
			return err
		}
	}

	return nil
}

// SearchWithFilter performs k-NN search with a filter function.
// The filter function returns true for vectors that should be included.
func (h *HNSW) SearchWithFilter(query []float32, k int, ef int, filter func(vectorID string) bool) []core.SearchResult {
	h.mu.RLock()
	defer h.mu.RUnlock()

	if h.size == 0 || filter == nil {
		return h.Search(query, k, ef)
	}

	// We need to search for more candidates since some will be filtered
	searchK := k * 10
	if searchK > int(h.size) {
		searchK = int(h.size)
	}

	if ef < searchK {
		ef = searchK
	}

	ep := h.entryPoint
	epNode := h.nodes[ep]
	if epNode == nil {
		return nil
	}

	// Traverse from top to layer 1
	for lc := h.maxLayer; lc > 0; lc-- {
		changed := true
		for changed {
			changed = false
			if epNode.Connections == nil || lc >= len(epNode.Connections) {
				break
			}
			for _, neighborID := range epNode.Connections[lc] {
				neighbor := h.nodes[neighborID]
				if neighbor == nil {
					continue
				}
				if h.calc.Distance(query, neighbor.Vector) < h.calc.Distance(query, epNode.Vector) {
					ep = neighborID
					epNode = neighbor
					changed = true
				}
			}
		}
	}

	// Search at layer 0 with ef
	candidates := h.searchLayer(query, ep, ef, 0)

	// Filter and return top k results
	results := make([]core.SearchResult, 0, k)
	for _, c := range candidates {
		node := h.nodes[c.nodeID]
		if node == nil {
			continue
		}
		if !filter(node.VectorID) {
			continue
		}
		score := c.distance
		if h.config.Metric == distance.DotProduct {
			score = -score
		}
		results = append(results, core.SearchResult{
			ID:    node.VectorID,
			Score: score,
		})
		if len(results) >= k {
			break
		}
	}

	return results
}

// ComputeRecall computes the recall against ground truth results.
// This is useful for benchmarking and quality assessment.
func ComputeRecall(predicted, groundTruth []core.SearchResult) float64 {
	if len(groundTruth) == 0 {
		return 0
	}

	gtSet := make(map[string]bool)
	for _, r := range groundTruth {
		gtSet[r.ID] = true
	}

	hits := 0
	for _, r := range predicted {
		if gtSet[r.ID] {
			hits++
		}
	}

	return float64(hits) / float64(len(groundTruth))
}
