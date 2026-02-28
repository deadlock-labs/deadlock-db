/**
 * End-to-end tests for the deadlockdb TypeScript client.
 *
 * These tests require a running deadlock-db server on localhost:8080.
 * Start one with:
 *
 *     go run ./cmd/deadlock-db -addr :8080
 *
 * Then run:
 *
 *     node --test clients/typescript/tests/client.test.js
 */

const { describe, it, before, after } = require("node:test");
const assert = require("node:assert/strict");

// We import from the built output.
let DeadlockClient;
try {
  DeadlockClient = require("../dist/index").DeadlockClient;
} catch {
  console.log("TypeScript client not built – run `npm run build` first. Skipping tests.");
  process.exit(0);
}

const SERVER_URL = "http://localhost:8080";
const COLLECTION = "tstest-col";

describe("DeadlockClient (integration)", () => {
  const client = new DeadlockClient(SERVER_URL);
  let serverAvailable = false;

  before(async () => {
    try {
      await client.health();
      serverAvailable = true;
    } catch {
      console.log("deadlock-db server not running – skipping integration tests.");
    }
    // Clean up any leftover test collection
    if (serverAvailable) {
      try { await client.deleteCollection(COLLECTION); } catch { /* ignore */ }
    }
  });

  after(async () => {
    if (serverAvailable) {
      try { await client.deleteCollection(COLLECTION); } catch { /* ignore */ }
    }
  });

  it("health", { skip: false }, async (t) => {
    if (!serverAvailable) return t.skip("server not running");
    const resp = await client.health();
    assert.equal(resp.status, "healthy");
  });

  it("version", async (t) => {
    if (!serverAvailable) return t.skip("server not running");
    const ver = await client.version();
    assert.ok(ver.length > 0);
  });

  it("collection lifecycle", async (t) => {
    if (!serverAvailable) return t.skip("server not running");

    const created = await client.createCollection(COLLECTION, 4, {
      metric: "cosine",
    });
    assert.equal(created.name, COLLECTION);

    const cols = await client.listCollections();
    assert.ok(cols.includes(COLLECTION));

    const info = await client.getCollection(COLLECTION);
    assert.equal(info.dimension, 4);

    await client.deleteCollection(COLLECTION);
    const cols2 = await client.listCollections();
    assert.ok(!cols2.includes(COLLECTION));
  });

  it("vector CRUD + search", async (t) => {
    if (!serverAvailable) return t.skip("server not running");

    await client.createCollection(COLLECTION, 4);

    const insertResp = await client.insert(COLLECTION, [
      { id: "v1", values: [1, 0, 0, 0], metadata: { cat: "a" } },
      { id: "v2", values: [0, 1, 0, 0], metadata: { cat: "b" } },
      { id: "v3", values: [0, 0, 1, 0], metadata: { cat: "a" } },
    ]);
    assert.equal(insertResp.inserted, 3);

    // Get
    const v = await client.get(COLLECTION, "v1");
    assert.equal(v.id, "v1");

    // Search
    const results = await client.search(COLLECTION, [1, 0.1, 0, 0], 2);
    assert.equal(results.length, 2);
    assert.equal(results[0].id, "v1");

    // Filtered search
    const filtered = await client.search(
      COLLECTION,
      [1, 0.1, 0, 0],
      10,
      { cat: "a" },
    );
    for (const r of filtered) {
      assert.equal(r.metadata?.cat, "a");
    }

    // Delete vector
    await client.delete(COLLECTION, "v1");
    await assert.rejects(() => client.get(COLLECTION, "v1"));
  });
});

describe("DeadlockClient (offline)", () => {
  it("strips trailing slash", () => {
    const c = new DeadlockClient("http://localhost:9999/");
    // Construction should not throw
    assert.ok(c);
  });
});
