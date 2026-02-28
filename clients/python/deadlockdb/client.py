"""HTTP client for the deadlock-db vector database."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional
from urllib.error import HTTPError
from urllib.request import Request, urlopen


class DeadlockError(Exception):
    """Error returned by the deadlock-db server."""


class DeadlockClient:
    """Client for interacting with a deadlock-db server over HTTP.

    Example::

        client = DeadlockClient("http://localhost:8080")
        client.create_collection("embeddings", dimension=128, metric="cosine")
        client.insert("embeddings", [
            {"id": "vec-1", "values": [0.1, 0.2, ...], "metadata": {"k": "v"}},
        ])
        results = client.search("embeddings", vector=[0.1, 0.2, ...], k=10)
    """

    def __init__(self, base_url: str = "http://localhost:8080") -> None:
        self.base_url = base_url.rstrip("/")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _request(
        self,
        method: str,
        path: str,
        body: Optional[Dict[str, Any]] = None,
    ) -> Any:
        url = f"{self.base_url}{path}"
        data = json.dumps(body).encode("utf-8") if body is not None else None
        req = Request(url, data=data, method=method)
        req.add_header("Content-Type", "application/json")

        try:
            with urlopen(req) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except HTTPError as exc:
            try:
                detail = json.loads(exc.read().decode("utf-8"))
                raise DeadlockError(detail.get("error", str(exc))) from exc
            except (json.JSONDecodeError, UnicodeDecodeError):
                raise DeadlockError(str(exc)) from exc

    # ------------------------------------------------------------------
    # Health / version
    # ------------------------------------------------------------------

    def health(self) -> Dict[str, str]:
        """Return the server health status."""
        return self._request("GET", "/health")

    def version(self) -> str:
        """Return the server version string."""
        resp = self._request("GET", "/version")
        return resp.get("version", "")

    # ------------------------------------------------------------------
    # Collections
    # ------------------------------------------------------------------

    def list_collections(self) -> List[str]:
        """List all collection names."""
        resp = self._request("GET", "/collections")
        return resp.get("collections", [])

    def create_collection(
        self,
        name: str,
        dimension: int,
        *,
        metric: str = "",
        m: int = 0,
        ef_construction: int = 0,
        quantization: str = "",
    ) -> Dict[str, Any]:
        """Create a new collection.

        Args:
            name: Collection name.
            dimension: Vector dimensionality.
            metric: Distance metric (cosine, euclidean, dotproduct).
            m: HNSW max connections per layer.
            ef_construction: HNSW construction expansion factor.
            quantization: Quantization type (none, scalar, pq, binary).
        """
        body: Dict[str, Any] = {"name": name, "dimension": dimension}
        if metric:
            body["metric"] = metric
        if m > 0:
            body["m"] = m
        if ef_construction > 0:
            body["ef_construction"] = ef_construction
        if quantization:
            body["quantization"] = quantization
        return self._request("POST", "/collections", body)

    def get_collection(self, name: str) -> Dict[str, Any]:
        """Get collection information."""
        return self._request("GET", f"/collections/{name}")

    def delete_collection(self, name: str) -> Dict[str, str]:
        """Delete a collection."""
        return self._request("DELETE", f"/collections/{name}")

    # ------------------------------------------------------------------
    # Vectors
    # ------------------------------------------------------------------

    def insert(
        self,
        collection: str,
        vectors: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Insert vectors into a collection.

        Args:
            collection: Collection name.
            vectors: List of dicts with ``id``, ``values``, and optional ``metadata``.
        """
        return self._request(
            "POST",
            f"/collections/{collection}/vectors",
            {"vectors": vectors},
        )

    def get(self, collection: str, vector_id: str) -> Dict[str, Any]:
        """Retrieve a vector by ID."""
        return self._request(
            "GET",
            f"/collections/{collection}/vectors?id={vector_id}",
        )

    def delete(self, collection: str, vector_id: str) -> Dict[str, str]:
        """Delete a vector by ID."""
        return self._request(
            "DELETE",
            f"/collections/{collection}/vectors?id={vector_id}",
        )

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        collection: str,
        vector: List[float],
        k: int = 10,
        *,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors.

        Args:
            collection: Collection name.
            vector: Query vector.
            k: Number of results to return.
            filter: Optional metadata filter.

        Returns:
            List of search result dicts with ``id``, ``score``, and ``metadata``.
        """
        body: Dict[str, Any] = {"vector": vector, "k": k}
        if filter is not None:
            body["filter"] = filter
        resp = self._request(
            "POST",
            f"/collections/{collection}/search",
            body,
        )
        return resp.get("results", [])
