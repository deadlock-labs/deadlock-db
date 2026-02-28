"""Basic example of using the deadlockdb Python client.

Prerequisites:
    1. Start the deadlock-db server:
         go build -o deadlock-db ./cmd/deadlock-db && ./deadlock-db -addr :8080
    2. Install the client:
         pip install clients/python      # from repo root

Usage:
    python examples/python/basic.py
"""

from deadlockdb import DeadlockClient


def main() -> None:
    # Connect to the running server
    client = DeadlockClient("http://localhost:8080")

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------
    print("Health:", client.health())
    print("Version:", client.version())

    # ------------------------------------------------------------------
    # Create a collection
    # ------------------------------------------------------------------
    collection_name = "example-collection"
    dimension = 128

    client.create_collection(
        collection_name,
        dimension=dimension,
        metric="cosine",
    )
    print(f"\nCreated collection '{collection_name}'")

    # ------------------------------------------------------------------
    # Insert vectors
    # ------------------------------------------------------------------
    vectors = [
        {
            "id": f"vec-{i}",
            "values": [float(i) / 100.0] * dimension,
            "metadata": {"category": "tech" if i % 2 == 0 else "science", "index": i},
        }
        for i in range(10)
    ]
    client.insert(collection_name, vectors)
    print(f"Inserted {len(vectors)} vectors")

    # ------------------------------------------------------------------
    # Search for similar vectors
    # ------------------------------------------------------------------
    query = [0.05] * dimension
    results = client.search(collection_name, vector=query, k=5)
    print("\nTop 5 results:")
    for r in results:
        print(f"  {r['id']}: score={r['score']:.4f}")

    # ------------------------------------------------------------------
    # Search with metadata filter
    # ------------------------------------------------------------------
    filtered = client.search(
        collection_name,
        vector=query,
        k=5,
        filter={"category": "tech"},
    )
    print("\nFiltered results (category='tech'):")
    for r in filtered:
        print(f"  {r['id']}: score={r['score']:.4f}")

    # ------------------------------------------------------------------
    # Get a single vector by ID
    # ------------------------------------------------------------------
    vec = client.get(collection_name, "vec-0")
    print(f"\nRetrieved vec-0: metadata={vec.get('metadata')}")

    # ------------------------------------------------------------------
    # List collections
    # ------------------------------------------------------------------
    print("\nCollections:", client.list_collections())

    # ------------------------------------------------------------------
    # Clean up
    # ------------------------------------------------------------------
    client.delete(collection_name, "vec-0")
    print("Deleted vec-0")

    client.delete_collection(collection_name)
    print(f"Deleted collection '{collection_name}'")


if __name__ == "__main__":
    main()
