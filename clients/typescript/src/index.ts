/**
 * deadlockdb – TypeScript/JavaScript client for the deadlock-db vector database.
 *
 * @example
 * ```ts
 * import { DeadlockClient } from "deadlockdb";
 *
 * const client = new DeadlockClient("http://localhost:8080");
 * await client.createCollection("embeddings", 128, { metric: "cosine" });
 * await client.insert("embeddings", [
 *   { id: "vec-1", values: Array(128).fill(0.1), metadata: { category: "tech" } },
 * ]);
 * const results = await client.search("embeddings", Array(128).fill(0.1), 10);
 * ```
 *
 * @packageDocumentation
 */

// ----------------------------------------------------------------
// Types
// ----------------------------------------------------------------

/** Options when creating a collection. */
export interface CreateCollectionOptions {
  metric?: string;
  m?: number;
  efConstruction?: number;
  quantization?: string;
}

/** A single vector to insert. */
export interface VectorInput {
  id: string;
  values: number[];
  metadata?: Record<string, unknown>;
}

/** A single search result. */
export interface SearchResult {
  id: string;
  score: number;
  metadata?: Record<string, unknown>;
}

// ----------------------------------------------------------------
// Client
// ----------------------------------------------------------------

export class DeadlockClient {
  private baseUrl: string;

  constructor(baseUrl: string = "http://localhost:8080") {
    this.baseUrl = baseUrl.replace(/\/+$/, "");
  }

  // ------------------------------------------------------------------
  // Internal helpers
  // ------------------------------------------------------------------

  private async request<T>(
    method: string,
    path: string,
    body?: unknown,
  ): Promise<T> {
    const url = `${this.baseUrl}${path}`;
    const options: RequestInit = {
      method,
      headers: { "Content-Type": "application/json" },
    };
    if (body !== undefined) {
      options.body = JSON.stringify(body);
    }

    const resp = await fetch(url, options);
    const data = await resp.json();

    if (!resp.ok) {
      throw new Error(data.error ?? `HTTP ${resp.status}`);
    }
    return data as T;
  }

  // ------------------------------------------------------------------
  // Health / version
  // ------------------------------------------------------------------

  /** Return the server health status. */
  async health(): Promise<{ status: string }> {
    return this.request("GET", "/health");
  }

  /** Return the server version string. */
  async version(): Promise<string> {
    const resp = await this.request<{ version: string }>("GET", "/version");
    return resp.version;
  }

  // ------------------------------------------------------------------
  // Collections
  // ------------------------------------------------------------------

  /** List all collection names. */
  async listCollections(): Promise<string[]> {
    const resp = await this.request<{ collections: string[] }>(
      "GET",
      "/collections",
    );
    return resp.collections ?? [];
  }

  /** Create a new collection. */
  async createCollection(
    name: string,
    dimension: number,
    options?: CreateCollectionOptions,
  ): Promise<{ name: string; dimension: number }> {
    const body: Record<string, unknown> = { name, dimension };
    if (options?.metric) body.metric = options.metric;
    if (options?.m) body.m = options.m;
    if (options?.efConstruction) body.ef_construction = options.efConstruction;
    if (options?.quantization) body.quantization = options.quantization;
    return this.request("POST", "/collections", body);
  }

  /** Get collection information. */
  async getCollection(name: string): Promise<Record<string, unknown>> {
    return this.request("GET", `/collections/${name}`);
  }

  /** Delete a collection. */
  async deleteCollection(name: string): Promise<{ status: string }> {
    return this.request("DELETE", `/collections/${name}`);
  }

  // ------------------------------------------------------------------
  // Vectors
  // ------------------------------------------------------------------

  /** Insert vectors into a collection. */
  async insert(
    collection: string,
    vectors: VectorInput[],
  ): Promise<{ inserted: number }> {
    return this.request("POST", `/collections/${collection}/vectors`, {
      vectors,
    });
  }

  /** Retrieve a vector by ID. */
  async get(
    collection: string,
    vectorId: string,
  ): Promise<Record<string, unknown>> {
    return this.request(
      "GET",
      `/collections/${collection}/vectors?id=${vectorId}`,
    );
  }

  /** Delete a vector by ID. */
  async delete(
    collection: string,
    vectorId: string,
  ): Promise<{ status: string }> {
    return this.request(
      "DELETE",
      `/collections/${collection}/vectors?id=${vectorId}`,
    );
  }

  // ------------------------------------------------------------------
  // Search
  // ------------------------------------------------------------------

  /** Search for similar vectors.
   *
   * @param collection - Collection name.
   * @param vector - Query vector.
   * @param k - Number of results.
   * @param filter - Optional metadata filter object.
   */
  async search(
    collection: string,
    vector: number[],
    k: number = 10,
    filter?: Record<string, unknown>,
  ): Promise<SearchResult[]> {
    const body: Record<string, unknown> = { vector, k };
    if (filter) body.filter = filter;
    const resp = await this.request<{ results: SearchResult[] }>(
      "POST",
      `/collections/${collection}/search`,
      body,
    );
    return resp.results ?? [];
  }
}
