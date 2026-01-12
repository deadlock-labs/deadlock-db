package api

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	deadlockdb "github.com/deadlock-labs/deadlock-db"
)

func setupTestServer(t *testing.T) (*Server, func()) {
	db, err := deadlockdb.Open(nil)
	if err != nil {
		t.Fatalf("failed to open db: %v", err)
	}

	server := NewServer(db, ":8080")

	cleanup := func() {
		db.Close()
	}

	return server, cleanup
}

func TestHealthEndpoint(t *testing.T) {
	server, cleanup := setupTestServer(t)
	defer cleanup()

	req := httptest.NewRequest(http.MethodGet, "/health", nil)
	w := httptest.NewRecorder()

	server.mux.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("expected status 200, got %d", w.Code)
	}

	var resp map[string]string
	json.NewDecoder(w.Body).Decode(&resp)

	if resp["status"] != "healthy" {
		t.Errorf("expected status 'healthy', got '%s'", resp["status"])
	}
}

func TestVersionEndpoint(t *testing.T) {
	server, cleanup := setupTestServer(t)
	defer cleanup()

	req := httptest.NewRequest(http.MethodGet, "/version", nil)
	w := httptest.NewRecorder()

	server.mux.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("expected status 200, got %d", w.Code)
	}

	var resp map[string]string
	json.NewDecoder(w.Body).Decode(&resp)

	if resp["version"] == "" {
		t.Error("version should not be empty")
	}
}

func TestCreateCollection(t *testing.T) {
	server, cleanup := setupTestServer(t)
	defer cleanup()

	body := `{"name": "test-vectors", "dimension": 128, "metric": "cosine"}`
	req := httptest.NewRequest(http.MethodPost, "/collections", bytes.NewBufferString(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	server.mux.ServeHTTP(w, req)

	if w.Code != http.StatusCreated {
		t.Errorf("expected status 201, got %d: %s", w.Code, w.Body.String())
	}

	var resp map[string]any
	json.NewDecoder(w.Body).Decode(&resp)

	if resp["name"] != "test-vectors" {
		t.Errorf("expected name 'test-vectors', got '%v'", resp["name"])
	}
}

func TestListCollections(t *testing.T) {
	server, cleanup := setupTestServer(t)
	defer cleanup()

	// Create a collection first
	body := `{"name": "my-collection", "dimension": 64}`
	req := httptest.NewRequest(http.MethodPost, "/collections", bytes.NewBufferString(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	server.mux.ServeHTTP(w, req)

	// List collections
	req = httptest.NewRequest(http.MethodGet, "/collections", nil)
	w = httptest.NewRecorder()
	server.mux.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("expected status 200, got %d", w.Code)
	}

	var resp map[string]any
	json.NewDecoder(w.Body).Decode(&resp)

	collections := resp["collections"].([]any)
	if len(collections) != 1 {
		t.Errorf("expected 1 collection, got %d", len(collections))
	}
}

func TestInsertAndSearchVectors(t *testing.T) {
	server, cleanup := setupTestServer(t)
	defer cleanup()

	// Create collection
	createBody := `{"name": "search-test", "dimension": 4}`
	req := httptest.NewRequest(http.MethodPost, "/collections", bytes.NewBufferString(createBody))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	server.mux.ServeHTTP(w, req)

	// Insert vectors
	insertBody := `{
		"vectors": [
			{"id": "vec-1", "values": [1.0, 0.0, 0.0, 0.0], "metadata": {"category": "a"}},
			{"id": "vec-2", "values": [0.0, 1.0, 0.0, 0.0], "metadata": {"category": "b"}},
			{"id": "vec-3", "values": [0.0, 0.0, 1.0, 0.0], "metadata": {"category": "a"}}
		]
	}`
	req = httptest.NewRequest(http.MethodPost, "/collections/search-test/vectors", bytes.NewBufferString(insertBody))
	req.Header.Set("Content-Type", "application/json")
	w = httptest.NewRecorder()
	server.mux.ServeHTTP(w, req)

	if w.Code != http.StatusCreated {
		t.Errorf("insert expected status 201, got %d: %s", w.Code, w.Body.String())
	}

	// Search
	searchBody := `{"vector": [1.0, 0.1, 0.0, 0.0], "k": 2}`
	req = httptest.NewRequest(http.MethodPost, "/collections/search-test/search", bytes.NewBufferString(searchBody))
	req.Header.Set("Content-Type", "application/json")
	w = httptest.NewRecorder()
	server.mux.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("search expected status 200, got %d: %s", w.Code, w.Body.String())
	}

	var resp map[string]any
	json.NewDecoder(w.Body).Decode(&resp)

	results := resp["results"].([]any)
	if len(results) != 2 {
		t.Errorf("expected 2 results, got %d", len(results))
	}

	// First result should be vec-1 (closest to query)
	first := results[0].(map[string]any)
	if first["id"] != "vec-1" {
		t.Errorf("expected first result to be vec-1, got %v", first["id"])
	}
}

func TestGetVector(t *testing.T) {
	server, cleanup := setupTestServer(t)
	defer cleanup()

	// Create collection
	createBody := `{"name": "get-test", "dimension": 4}`
	req := httptest.NewRequest(http.MethodPost, "/collections", bytes.NewBufferString(createBody))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	server.mux.ServeHTTP(w, req)

	// Insert vector
	insertBody := `{"vectors": [{"id": "my-vec", "values": [1.0, 2.0, 3.0, 4.0]}]}`
	req = httptest.NewRequest(http.MethodPost, "/collections/get-test/vectors", bytes.NewBufferString(insertBody))
	req.Header.Set("Content-Type", "application/json")
	w = httptest.NewRecorder()
	server.mux.ServeHTTP(w, req)

	// Get vector
	req = httptest.NewRequest(http.MethodGet, "/collections/get-test/vectors?id=my-vec", nil)
	w = httptest.NewRecorder()
	server.mux.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("expected status 200, got %d: %s", w.Code, w.Body.String())
	}

	var vec map[string]any
	json.NewDecoder(w.Body).Decode(&vec)

	if vec["id"] != "my-vec" {
		t.Errorf("expected id 'my-vec', got '%v'", vec["id"])
	}
}

func TestDeleteVector(t *testing.T) {
	server, cleanup := setupTestServer(t)
	defer cleanup()

	// Create collection
	createBody := `{"name": "delete-test", "dimension": 4}`
	req := httptest.NewRequest(http.MethodPost, "/collections", bytes.NewBufferString(createBody))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	server.mux.ServeHTTP(w, req)

	// Insert vector
	insertBody := `{"vectors": [{"id": "to-delete", "values": [1.0, 2.0, 3.0, 4.0]}]}`
	req = httptest.NewRequest(http.MethodPost, "/collections/delete-test/vectors", bytes.NewBufferString(insertBody))
	req.Header.Set("Content-Type", "application/json")
	w = httptest.NewRecorder()
	server.mux.ServeHTTP(w, req)

	// Delete vector
	req = httptest.NewRequest(http.MethodDelete, "/collections/delete-test/vectors?id=to-delete", nil)
	w = httptest.NewRecorder()
	server.mux.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("delete expected status 200, got %d", w.Code)
	}

	// Verify it's gone
	req = httptest.NewRequest(http.MethodGet, "/collections/delete-test/vectors?id=to-delete", nil)
	w = httptest.NewRecorder()
	server.mux.ServeHTTP(w, req)

	if w.Code != http.StatusNotFound {
		t.Errorf("expected 404 after delete, got %d", w.Code)
	}
}

func TestDeleteCollection(t *testing.T) {
	server, cleanup := setupTestServer(t)
	defer cleanup()

	// Create collection
	createBody := `{"name": "to-delete", "dimension": 4}`
	req := httptest.NewRequest(http.MethodPost, "/collections", bytes.NewBufferString(createBody))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	server.mux.ServeHTTP(w, req)

	// Delete collection
	req = httptest.NewRequest(http.MethodDelete, "/collections/to-delete", nil)
	w = httptest.NewRecorder()
	server.mux.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("expected status 200, got %d", w.Code)
	}

	// Verify it's gone
	req = httptest.NewRequest(http.MethodGet, "/collections/to-delete", nil)
	w = httptest.NewRecorder()
	server.mux.ServeHTTP(w, req)

	if w.Code != http.StatusNotFound {
		t.Errorf("expected 404 after delete, got %d", w.Code)
	}
}

func TestSearchWithFilter(t *testing.T) {
	server, cleanup := setupTestServer(t)
	defer cleanup()

	// Create collection
	createBody := `{"name": "filter-test", "dimension": 4}`
	req := httptest.NewRequest(http.MethodPost, "/collections", bytes.NewBufferString(createBody))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	server.mux.ServeHTTP(w, req)

	// Insert vectors
	insertBody := `{
		"vectors": [
			{"id": "vec-1", "values": [1.0, 0.0, 0.0, 0.0], "metadata": {"category": "tech", "year": 2023}},
			{"id": "vec-2", "values": [0.9, 0.1, 0.0, 0.0], "metadata": {"category": "science", "year": 2023}},
			{"id": "vec-3", "values": [0.8, 0.2, 0.0, 0.0], "metadata": {"category": "tech", "year": 2022}}
		]
	}`
	req = httptest.NewRequest(http.MethodPost, "/collections/filter-test/vectors", bytes.NewBufferString(insertBody))
	req.Header.Set("Content-Type", "application/json")
	w = httptest.NewRecorder()
	server.mux.ServeHTTP(w, req)

	// Search with filter
	searchBody := `{
		"vector": [1.0, 0.0, 0.0, 0.0],
		"k": 10,
		"filter": {"category": "tech"}
	}`
	req = httptest.NewRequest(http.MethodPost, "/collections/filter-test/search", bytes.NewBufferString(searchBody))
	req.Header.Set("Content-Type", "application/json")
	w = httptest.NewRecorder()
	server.mux.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("expected status 200, got %d: %s", w.Code, w.Body.String())
	}

	var resp map[string]any
	json.NewDecoder(w.Body).Decode(&resp)

	results := resp["results"].([]any)
	// Should only get tech vectors
	for _, r := range results {
		result := r.(map[string]any)
		metadata := result["metadata"].(map[string]any)
		if metadata["category"] != "tech" {
			t.Errorf("expected category 'tech', got '%v'", metadata["category"])
		}
	}
}
