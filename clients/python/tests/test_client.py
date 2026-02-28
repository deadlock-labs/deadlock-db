"""End-to-end tests for the deadlockdb Python client.

These tests require a running deadlock-db server on localhost:8080.
Start one with:

    go run ./cmd/deadlock-db -addr :8080

Then run:

    python -m pytest clients/python/tests/ -v
"""

import random
import unittest

from deadlockdb import DeadlockClient


SERVER_URL = "http://localhost:8080"

# Set to True when a server is available for integration testing.
# CI environments should start the server first.
_SERVER_AVAILABLE = False
try:
    DeadlockClient(SERVER_URL).health()
    _SERVER_AVAILABLE = True
except Exception:
    pass


@unittest.skipUnless(_SERVER_AVAILABLE, "deadlock-db server not running")
class TestDeadlockClient(unittest.TestCase):
    """Integration tests against a live server."""

    def setUp(self):
        self.client = DeadlockClient(SERVER_URL)
        # Clean up any leftover test collection
        try:
            self.client.delete_collection("pytest-col")
        except Exception:
            pass

    def tearDown(self):
        try:
            self.client.delete_collection("pytest-col")
        except Exception:
            pass

    def test_health(self):
        resp = self.client.health()
        self.assertEqual(resp["status"], "healthy")

    def test_version(self):
        ver = self.client.version()
        self.assertTrue(len(ver) > 0)

    def test_collection_lifecycle(self):
        # Create
        resp = self.client.create_collection("pytest-col", dimension=4, metric="cosine")
        self.assertEqual(resp["name"], "pytest-col")

        # List
        cols = self.client.list_collections()
        self.assertIn("pytest-col", cols)

        # Get info
        info = self.client.get_collection("pytest-col")
        self.assertEqual(info["dimension"], 4)

        # Delete
        self.client.delete_collection("pytest-col")
        cols = self.client.list_collections()
        self.assertNotIn("pytest-col", cols)

    def test_vector_crud_and_search(self):
        self.client.create_collection("pytest-col", dimension=4)

        # Insert
        vectors = [
            {"id": "v1", "values": [1.0, 0.0, 0.0, 0.0], "metadata": {"cat": "a"}},
            {"id": "v2", "values": [0.0, 1.0, 0.0, 0.0], "metadata": {"cat": "b"}},
            {"id": "v3", "values": [0.0, 0.0, 1.0, 0.0], "metadata": {"cat": "a"}},
        ]
        resp = self.client.insert("pytest-col", vectors)
        self.assertEqual(resp["inserted"], 3)

        # Get
        v = self.client.get("pytest-col", "v1")
        self.assertEqual(v["id"], "v1")

        # Search
        results = self.client.search("pytest-col", vector=[1.0, 0.1, 0.0, 0.0], k=2)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["id"], "v1")

        # Filtered search
        results = self.client.search(
            "pytest-col",
            vector=[1.0, 0.1, 0.0, 0.0],
            k=10,
            filter={"cat": "a"},
        )
        for r in results:
            self.assertEqual(r["metadata"]["cat"], "a")

        # Delete vector
        self.client.delete("pytest-col", "v1")
        with self.assertRaises(Exception):
            self.client.get("pytest-col", "v1")


class TestClientOffline(unittest.TestCase):
    """Tests that don't require a running server."""

    def test_client_creation(self):
        c = DeadlockClient("http://localhost:9999")
        self.assertEqual(c.base_url, "http://localhost:9999")

    def test_trailing_slash_stripped(self):
        c = DeadlockClient("http://localhost:9999/")
        self.assertEqual(c.base_url, "http://localhost:9999")


if __name__ == "__main__":
    unittest.main()
