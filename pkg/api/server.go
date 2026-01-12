// Package api provides HTTP and gRPC API interfaces for deadlock-db.
package api

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strings"

	deadlockdb "github.com/deadlock-labs/deadlock-db"
	"github.com/deadlock-labs/deadlock-db/pkg/core"
	"github.com/deadlock-labs/deadlock-db/pkg/distance"
	"github.com/deadlock-labs/deadlock-db/pkg/filter"
)

// Server provides the HTTP API for deadlock-db.
type Server struct {
	db   *deadlockdb.DB
	addr string
	mux  *http.ServeMux
}

// NewServer creates a new API server.
func NewServer(db *deadlockdb.DB, addr string) *Server {
	s := &Server{
		db:   db,
		addr: addr,
		mux:  http.NewServeMux(),
	}
	s.setupRoutes()
	return s
}

// setupRoutes configures the HTTP routes.
func (s *Server) setupRoutes() {
	s.mux.HandleFunc("/health", s.handleHealth)
	s.mux.HandleFunc("/version", s.handleVersion)
	s.mux.HandleFunc("/collections", s.handleCollections)
	s.mux.HandleFunc("/collections/", s.handleCollection)
}

// Start starts the HTTP server.
func (s *Server) Start() error {
	return http.ListenAndServe(s.addr, s.mux)
}

// handleHealth handles health check requests.
func (s *Server) handleHealth(w http.ResponseWriter, _ *http.Request) {
	s.writeJSON(w, http.StatusOK, map[string]string{"status": "healthy"})
}

// handleVersion handles version requests.
func (s *Server) handleVersion(w http.ResponseWriter, _ *http.Request) {
	s.writeJSON(w, http.StatusOK, map[string]string{"version": deadlockdb.Version()})
}

// handleCollections handles collection listing and creation.
func (s *Server) handleCollections(w http.ResponseWriter, r *http.Request) {
	switch r.Method {
	case http.MethodGet:
		s.listCollections(w, r)
	case http.MethodPost:
		s.createCollection(w, r)
	default:
		s.writeError(w, http.StatusMethodNotAllowed, "method not allowed")
	}
}

// handleCollection handles individual collection operations.
func (s *Server) handleCollection(w http.ResponseWriter, r *http.Request) {
	path := strings.TrimPrefix(r.URL.Path, "/collections/")
	parts := strings.Split(path, "/")
	
	if len(parts) == 0 || parts[0] == "" {
		s.writeError(w, http.StatusBadRequest, "collection name required")
		return
	}

	collName := parts[0]
	
	if len(parts) == 1 {
		// Collection operations
		switch r.Method {
		case http.MethodGet:
			s.getCollectionInfo(w, r, collName)
		case http.MethodDelete:
			s.deleteCollection(w, r, collName)
		default:
			s.writeError(w, http.StatusMethodNotAllowed, "method not allowed")
		}
		return
	}

	operation := parts[1]
	switch operation {
	case "vectors":
		s.handleVectors(w, r, collName)
	case "search":
		s.handleSearch(w, r, collName)
	default:
		s.writeError(w, http.StatusNotFound, "unknown operation")
	}
}

// listCollections lists all collections.
func (s *Server) listCollections(w http.ResponseWriter, _ *http.Request) {
	collections := s.db.ListCollections()
	s.writeJSON(w, http.StatusOK, map[string]any{
		"collections": collections,
	})
}

// CreateCollectionRequest is the request body for creating a collection.
type CreateCollectionRequest struct {
	Name           string `json:"name"`
	Dimension      int    `json:"dimension"`
	Metric         string `json:"metric,omitempty"`
	M              int    `json:"m,omitempty"`
	EfConstruction int    `json:"ef_construction,omitempty"`
	Quantization   string `json:"quantization,omitempty"`
}

// createCollection creates a new collection.
func (s *Server) createCollection(w http.ResponseWriter, r *http.Request) {
	var req CreateCollectionRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		s.writeError(w, http.StatusBadRequest, "invalid request body")
		return
	}

	if req.Name == "" {
		s.writeError(w, http.StatusBadRequest, "name is required")
		return
	}
	if req.Dimension <= 0 {
		s.writeError(w, http.StatusBadRequest, "dimension must be positive")
		return
	}

	opts := []deadlockdb.CollectionOption{}
	if req.Metric != "" {
		opts = append(opts, deadlockdb.WithMetric(distance.ParseMetric(req.Metric)))
	}
	if req.M > 0 && req.EfConstruction > 0 {
		opts = append(opts, deadlockdb.WithHNSWParams(req.M, req.EfConstruction))
	}
	if req.Quantization != "" {
		opts = append(opts, deadlockdb.WithQuantization(req.Quantization))
	}

	coll, err := s.db.CreateCollection(req.Name, req.Dimension, opts...)
	if err != nil {
		s.writeError(w, http.StatusConflict, err.Error())
		return
	}

	s.writeJSON(w, http.StatusCreated, map[string]any{
		"name":      coll.Name(),
		"dimension": coll.Dimension(),
	})
}

// getCollectionInfo returns information about a collection.
func (s *Server) getCollectionInfo(w http.ResponseWriter, _ *http.Request, collName string) {
	coll, err := s.db.GetCollection(collName)
	if err != nil {
		s.writeError(w, http.StatusNotFound, err.Error())
		return
	}

	s.writeJSON(w, http.StatusOK, coll.Stats())
}

// deleteCollection deletes a collection.
func (s *Server) deleteCollection(w http.ResponseWriter, _ *http.Request, collName string) {
	if err := s.db.DeleteCollection(collName); err != nil {
		s.writeError(w, http.StatusNotFound, err.Error())
		return
	}

	s.writeJSON(w, http.StatusOK, map[string]string{"status": "deleted"})
}

// handleVectors handles vector CRUD operations.
func (s *Server) handleVectors(w http.ResponseWriter, r *http.Request, collName string) {
	// Get vector ID from query params if present
	vectorID := r.URL.Query().Get("id")

	switch r.Method {
	case http.MethodPost:
		s.insertVectors(w, r, collName)
	case http.MethodGet:
		if vectorID == "" {
			s.writeError(w, http.StatusBadRequest, "vector id required")
			return
		}
		s.getVector(w, r, collName, vectorID)
	case http.MethodDelete:
		if vectorID == "" {
			s.writeError(w, http.StatusBadRequest, "vector id required")
			return
		}
		s.deleteVector(w, r, collName, vectorID)
	default:
		s.writeError(w, http.StatusMethodNotAllowed, "method not allowed")
	}
}

// InsertVectorsRequest is the request body for inserting vectors.
type InsertVectorsRequest struct {
	Vectors []struct {
		ID       string         `json:"id"`
		Values   []float32      `json:"values"`
		Metadata map[string]any `json:"metadata,omitempty"`
	} `json:"vectors"`
}

// insertVectors inserts vectors into a collection.
func (s *Server) insertVectors(w http.ResponseWriter, r *http.Request, collName string) {
	var req InsertVectorsRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		s.writeError(w, http.StatusBadRequest, "invalid request body")
		return
	}

	if len(req.Vectors) == 0 {
		s.writeError(w, http.StatusBadRequest, "no vectors provided")
		return
	}

	vectors := make([]*core.Vector, len(req.Vectors))
	for i, v := range req.Vectors {
		vectors[i] = core.NewVectorWithMetadata(v.ID, v.Values, v.Metadata)
	}

	if err := s.db.InsertBatch(collName, vectors); err != nil {
		s.writeError(w, http.StatusBadRequest, err.Error())
		return
	}

	s.writeJSON(w, http.StatusCreated, map[string]any{
		"inserted": len(vectors),
	})
}

// getVector retrieves a vector by ID.
func (s *Server) getVector(w http.ResponseWriter, _ *http.Request, collName, vectorID string) {
	v, err := s.db.Get(collName, vectorID)
	if err != nil {
		s.writeError(w, http.StatusNotFound, err.Error())
		return
	}

	s.writeJSON(w, http.StatusOK, v)
}

// deleteVector deletes a vector by ID.
func (s *Server) deleteVector(w http.ResponseWriter, _ *http.Request, collName, vectorID string) {
	if err := s.db.Delete(collName, vectorID); err != nil {
		s.writeError(w, http.StatusNotFound, err.Error())
		return
	}

	s.writeJSON(w, http.StatusOK, map[string]string{"status": "deleted"})
}

// SearchRequest is the request body for search.
type SearchRequest struct {
	Vector []float32      `json:"vector"`
	K      int            `json:"k"`
	Filter map[string]any `json:"filter,omitempty"`
	Ef     int            `json:"ef,omitempty"`
	Rerank bool           `json:"rerank,omitempty"`
}

// handleSearch handles search requests.
func (s *Server) handleSearch(w http.ResponseWriter, r *http.Request, collName string) {
	if r.Method != http.MethodPost {
		s.writeError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}

	var req SearchRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		s.writeError(w, http.StatusBadRequest, "invalid request body")
		return
	}

	if len(req.Vector) == 0 {
		s.writeError(w, http.StatusBadRequest, "vector is required")
		return
	}
	if req.K <= 0 {
		req.K = 10
	}

	var f *filter.Filter
	if req.Filter != nil {
		var err error
		f, err = filter.ParseFilter(req.Filter)
		if err != nil {
			s.writeError(w, http.StatusBadRequest, fmt.Sprintf("invalid filter: %v", err))
			return
		}
	}

	var results []core.SearchResult
	var err error
	if f != nil {
		results, err = s.db.SearchWithFilter(collName, req.Vector, req.K, f)
	} else {
		results, err = s.db.Search(collName, req.Vector, req.K)
	}

	if err != nil {
		s.writeError(w, http.StatusBadRequest, err.Error())
		return
	}

	s.writeJSON(w, http.StatusOK, map[string]any{
		"results": results,
	})
}

// writeJSON writes a JSON response.
func (s *Server) writeJSON(w http.ResponseWriter, status int, data any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(data)
}

// writeError writes an error response.
func (s *Server) writeError(w http.ResponseWriter, status int, message string) {
	s.writeJSON(w, status, map[string]string{"error": message})
}
