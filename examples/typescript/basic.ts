/**
 * Basic example of using the deadlockdb TypeScript client.
 *
 * Prerequisites:
 *   1. Start the deadlock-db server:
 *        go build -o deadlock-db ./cmd/deadlock-db && ./deadlock-db -addr :8080
 *   2. Install the client:
 *        npm install clients/typescript   # from repo root
 *
 * Usage:
 *   npx tsx examples/typescript/basic.ts
 */

import { DeadlockClient } from "deadlockdb";

async function main(): Promise<void> {
  // Connect to the running server
  const client = new DeadlockClient("http://localhost:8080");

  // ------------------------------------------------------------------
  // Health check
  // ------------------------------------------------------------------
  console.log("Health:", await client.health());
  console.log("Version:", await client.version());

  // ------------------------------------------------------------------
  // Create a collection
  // ------------------------------------------------------------------
  const collectionName = "example-collection";
  const dimension = 128;

  await client.createCollection(collectionName, dimension, { metric: "cosine" });
  console.log(`\nCreated collection '${collectionName}'`);

  // ------------------------------------------------------------------
  // Insert vectors
  // ------------------------------------------------------------------
  const vectors = Array.from({ length: 10 }, (_, i) => ({
    id: `vec-${i}`,
    values: Array(dimension).fill(i / 100),
    metadata: { category: i % 2 === 0 ? "tech" : "science", index: i },
  }));

  await client.insert(collectionName, vectors);
  console.log(`Inserted ${vectors.length} vectors`);

  // ------------------------------------------------------------------
  // Search for similar vectors
  // ------------------------------------------------------------------
  const query = Array(dimension).fill(0.05);
  const results = await client.search(collectionName, query, 5);
  console.log("\nTop 5 results:");
  for (const r of results) {
    console.log(`  ${r.id}: score=${r.score.toFixed(4)}`);
  }

  // ------------------------------------------------------------------
  // Search with metadata filter
  // ------------------------------------------------------------------
  const filtered = await client.search(collectionName, query, 5, {
    category: "tech",
  });
  console.log("\nFiltered results (category='tech'):");
  for (const r of filtered) {
    console.log(`  ${r.id}: score=${r.score.toFixed(4)}`);
  }

  // ------------------------------------------------------------------
  // Get a single vector by ID
  // ------------------------------------------------------------------
  const vec = await client.get(collectionName, "vec-0");
  console.log(`\nRetrieved vec-0: metadata=${JSON.stringify(vec.metadata)}`);

  // ------------------------------------------------------------------
  // List collections
  // ------------------------------------------------------------------
  console.log("\nCollections:", await client.listCollections());

  // ------------------------------------------------------------------
  // Clean up
  // ------------------------------------------------------------------
  await client.delete(collectionName, "vec-0");
  console.log("Deleted vec-0");

  await client.deleteCollection(collectionName);
  console.log(`Deleted collection '${collectionName}'`);
}

main().catch(console.error);
