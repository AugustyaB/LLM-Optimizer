"""Domain definitions and prompt banks for dataset synthesis.

Why: Centralizing domain metadata and prompt templates keeps synthesis logic modular,
prevents monolithic scripts, and ensures rich coverage across polyglot ecosystems.
"""

from __future__ import annotations

from typing import Any

# Domain definitions: (topic, files_p1, files_p2, files_p3, invariants)
_RAW_DOMAINS: list[tuple[str, list[str], list[str], list[str], list[str]]] = [
    ("CLI Application & Terminal Selector",
     ["pcompile/config.py", "pcompile/session_selector.py", "pyproject.toml"],
     ["pcompile/harvester.py", "pcompile/evictor.py", "task.md"],
     ["pcompile/modes.py", "training/synthesize_dataset.py"],
     ["Always maintain zero idle VRAM footprint.", "Do not scaffold future files.", "CPU harvesting < 50 ms."]),
    ("Spring Boot REST Service & JPA Auditing",
     ["src/main/java/config/AuditConfig.java", "pom.xml"],
     ["src/main/java/service/OrderService.java", "src/main/java/repo/OrderRepository.java"],
     ["src/main/java/controller/OrderController.java", "src/test/OrderTest.java"],
     ["Enforce transactional boundaries.", "Never bypass security filter chain.", "Annotate with @Version."]),
    ("Distributed Async Cache & Eviction Engine",
     ["cache/core.py", "cache/ring_buffer.py", "setup.py"],
     ["cache/evictor.py", "cache/lru_clock.py", "task.md"],
     ["cache/client.py", "benchmarks/bench_throughput.py"],
     ["Enforce thread-safe lock-free reads.", "Never perform blocking I/O on loop.", "Touch active files only."]),
    ("High-Throughput JSONL Telemetry Pipeline",
     ["pipeline/schema.py", "pipeline/reader.py", "pyproject.toml"],
     ["pipeline/processor.py", "pipeline/filter.py", "task.md"],
     ["pipeline/sink.py", "training/train_eval.py"],
     ["Stream records in binary chunks.", "Skip malformed records safely.", "All test assertions must pass."]),
    ("PostgreSQL Partial Indexing & Query Planner",
     ["db/conn.py", "db/indexer.py", "schema.sql"],
     ["db/explain_analyzer.py", "db/index_tuner.py", "task.md"],
     ["db/migrations.py", "benchmarks/bench_query.py"],
     ["Never execute table-locking DDL without CONCURRENTLY.", "Index foreign keys.", "Bound connection pools."]),
    ("React State & Optimistic UI Concurrency",
     ["ui/store.ts", "ui/actions.ts", "package.json"],
     ["ui/optimistic_queue.ts", "ui/reconciler.ts", "task.md"],
     ["ui/components.tsx", "tests/ui.test.ts"],
     ["Roll back state on 5xx failure.", "Prevent duplicate renders.", "Ensure immutability across state."]),
    ("Rust FFI & C ABI Memory Layout",
     ["native/bridge.rs", "native/c_abi.h", "Cargo.toml"],
     ["native/memory_arena.rs", "native/wrapper.py", "task.md"],
     ["native/bindings.py", "tests/test_ffi.py"],
     ["Explicit ownership handover at boundary.", "Never deallocate foreign pointers.", "Enforce repr(C)."]),
    ("JWT Authentication & Session Revocation",
     ["auth/jwt_codec.py", "auth/crypto_keys.py", "pyproject.toml"],
     ["auth/revocation_list.py", "auth/middleware.py", "task.md"],
     ["auth/oauth_flows.py", "tests/test_auth.py"],
     ["Verify crypto signatures constant-time.", "Never expose refresh tokens.", "Enforce expiration in 15m."]),
    ("WebSocket Realtime Feed & Heartbeat Protocol",
     ["ws/framing.py", "ws/handshake.py", "pyproject.toml"],
     ["ws/heartbeat.py", "ws/channel_mgr.py", "task.md"],
     ["ws/broadcast.py", "tests/test_ws.py"],
     ["Close stale connections after 3 missed pings.", "Never buffer unbounded queues.", "Handle backpressure."]),
    ("Docker Multi-Stage Build & Layer Cache Optimizer",
     ["docker/builder.py", "docker/dockerfile_ast.py", "pyproject.toml"],
     ["docker/layer_cache.py", "docker/pruner.py", "task.md"],
     ["docker/exporter.py", "tests/test_docker.py"],
     ["Order instructions by change frequency.", "Never include secret tokens.", "Strip dev toolchains."]),
    ("gRPC Streaming & Protobuf Serializer",
     ["rpc/codegen.py", "rpc/proto_schema.proto", "pyproject.toml"],
     ["rpc/stream_interceptor.py", "rpc/compression.py", "task.md"],
     ["rpc/client_stub.py", "tests/test_grpc.py"],
     ["Enforce max message size (4MB).", "Propagate cancellation tokens.", "Gracefully terminate streams."]),
    ("Redis Distributed Redlock & Lease Renewal",
     ["lock/redlock.py", "lock/redis_conn.py", "pyproject.toml"],
     ["lock/lease_watchdog.py", "lock/fencing_token.py", "task.md"],
     ["lock/ha_cluster.py", "tests/test_lock.py"],
     ["Acquire locks across quorum.", "Release locks with atomic Lua scripts.", "Enforce monotonic tokens."]),
    ("Python AST Code Rewriter & Invariant Checker",
     ["ast_tools/parser.py", "ast_tools/visitors.py", "pyproject.toml"],
     ["ast_tools/rewriter.py", "ast_tools/safety_gate.py", "task.md"],
     ["ast_tools/formatter.py", "tests/test_ast.py"],
     ["Preserve original line mappings.", "Never rewrite outside function scope.", "Verify syntax validity."]),
    ("Kafka Partition Consumer & Rebalance Guard",
     ["kafka/consumer.py", "kafka/offsets.py", "pyproject.toml"],
     ["kafka/rebalance_listener.py", "kafka/dead_letter.py", "task.md"],
     ["kafka/batch_sink.py", "tests/test_kafka.py"],
     ["Commit offsets only after processing.", "Flush in-flight records.", "Route unprocessable to DLQ."]),
    ("GraphQL Federation & Batch DataLoader",
     ["graphql/schema.py", "graphql/resolvers.py", "pyproject.toml"],
     ["graphql/dataloader.py", "graphql/query_complexity.py", "task.md"],
     ["graphql/gateway.py", "tests/test_graphql.py"],
     ["Batch lookups to eliminate N+1 queries.", "Reject queries exceeding depth 6.", "Sanitize variables."]),
    ("Raft Consensus Election & WAL Log Compaction",
     ["raft/log.py", "raft/state_machine.py", "pyproject.toml"],
     ["raft/election.py", "raft/snapshot.py", "task.md"],
     ["raft/peer_rpc.py", "tests/test_raft.py"],
     ["Persist currentTerm to disk before vote reply.", "Never commit prior term entries.", "Atomic snapshots."]),
    ("SIMD Matrix Multiplier & Tensor Layout",
     ["simd/intrinsics.c", "simd/matrix_view.py", "pyproject.toml"],
     ["simd/avx_kernel.c", "simd/tile_cache.py", "task.md"],
     ["simd/benchmark.py", "tests/test_simd.py"],
     ["Ensure 64-byte alignment on vector buffers.", "Tile matrix for L1 cache.", "Guard AVX instructions."]),
    ("WebAssembly WASI Sandboxed Runtime",
     ["wasm/runtime.py", "wasm/host_bindings.py", "pyproject.toml"],
     ["wasm/sandbox_limits.py", "wasm/memory_guard.py", "task.md"],
     ["wasm/module_loader.py", "tests/test_wasm.py"],
     ["Enforce strict 256MB memory cap.", "Restrict FDs to workspace.", "Kill guest on fuel exhaustion."]),
    ("TLS 1.3 Handshake & Certificate Verification",
     ["tls/handshake.py", "tls/cipher_suites.py", "pyproject.toml"],
     ["tls/cert_chain.py", "tls/session_ticket.py", "task.md"],
     ["tls/crypto_provider.py", "tests/test_tls.py"],
     ["Verify OCSP stapling status.", "Disallow TLS versions prior to 1.3.", "Erase ephemeral DH keys."]),
    ("Write-Ahead Logging & Crash Recovery Engine",
     ["wal/record.py", "wal/disk_writer.py", "pyproject.toml"],
     ["wal/recovery.py", "wal/checkpoint.py", "task.md"],
     ["wal/fdatasync_worker.py", "tests/test_wal.py"],
     ["Call fdatasync before commit return.", "Verify CRC32 checksums on replay.", "Keep checkpoints in order."]),
    ("B-Tree Self-Balancing Disk Index",
     ["btree/node.py", "btree/pager.py", "pyproject.toml"],
     ["btree/split_merge.py", "btree/cursor.py", "task.md"],
     ["btree/index_api.py", "tests/test_btree.py"],
     ["Maintain fill factor 50-100% on nodes.", "Pin buffer pool pages during descent.", "Balance siblings."]),
    ("S3 Multipart Upload & ETag Checksum Validator",
     ["s3/multipart.py", "s3/client.py", "pyproject.toml"],
     ["s3/chunk_hasher.py", "s3/retry_uploader.py", "task.md"],
     ["s3/abort_handler.py", "tests/test_s3.py"],
     ["Calculate MD5 and multipart ETag format.", "Retry failed uploads with backoff.", "Abort on errors."]),
    ("Vector Database HNSW Graph Search",
     ["hnsw/graph.py", "hnsw/distance.py", "pyproject.toml"],
     ["hnsw/search_layer.py", "hnsw/index_builder.py", "task.md"],
     ["hnsw/serialization.py", "tests/test_hnsw.py"],
     ["Precompute distance using SIMD dot products.", "Bound incoming degree.", "Lock-free read traversal."]),
    ("Kubernetes Controller & Reconciler",
     ["controller/crd.go", "controller/client.go", "go.mod"],
     ["controller/reconciler.go", "controller/queue.go", "task.md"],
     ["controller/status.go", "controller_test.go"],
     ["Reconcile state idempotently against cluster.", "Use rate-limiting queues.", "Update status atomically."]),
    ("Embedded SQLite Analytics & Log Harvester",
     ["storage/db.py", "storage/migrations.py", "pyproject.toml"],
     ["harvester/log_indexer.py", "harvester/query.py", "task.md"],
     ["api/routes.py", "training/dataset_builder.py"],
     ["Execute atomic transactions with rollback.", "Never leave unclosed connections.", "Zero-idle daemon."]),
]

DOMAINS: list[dict[str, Any]] = [
    {"topic": t, "files_p1": f1, "files_p2": f2, "files_p3": f3, "invariants": inv}
    for t, f1, f2, f3, inv in _RAW_DOMAINS
]


def build_refine_prompts_bank() -> list[str]:
    """Return a curated bank of >= 200 unique software engineering requests using 21 templates (gcd(25,21)=1)."""
    templates = [
        "Implement buffered stream reading with max 64KB chunks to prevent RAM overflow during large JSONL scans.",
        "Add thread-safe double-checked locking on the session cache to avoid data races in concurrent workers.",
        "Support case-insensitive matching for markdown section titles across CRLF and LF line endings.",
        "Enforce strict mutual exclusivity between evicted files and active phase files.",
        "Add a fallback mechanism when prompt_toolkit cannot access a valid TTY terminal.",
        "Validate that all extracted file paths are normalized to relative POSIX style without path traversal.",
        "Implement a fast binary newline counter that avoids full UTF-8 string decoding on multi-megabyte files.",
        "Gracefully catch and log FileNotFoundError when scanning empty or missing brain directories.",
        "Add execution time assertion ensuring fast-path lookup runs strictly under 15 ms.",
        "Extract imperative constraints from user turns using deduplicated case-insensitive matching.",
        "Implement zero-idle model offload with keep_alive=0 to ensure 0 MB VRAM retention.",
        "Add CRC32 checksum verification for Write-Ahead Log entries during crash recovery.",
        "Enforce monotonically increasing sequence IDs on distributed fencing tokens.",
        "Add atomic file replacement guards using temporary file rename to prevent partial writes.",
        "Implement exponential backoff with full jitter for outbound HTTP retry loops.",
        "Add connection pool health checks with heartbeat ping to detect zombie TCP sockets.",
        "Enforce max depth validation on GraphQL AST queries to block recursive nesting attacks.",
        "Implement constant-time HMAC comparison to prevent timing side-channel attacks.",
        "Add memory arena allocator for FFI structs to avoid heap fragmentation in C bindings.",
        "Implement SIMD vectorized dot product kernel with runtime AVX-512 CPU feature checks.",
        "Configure zero-copy ring buffer slices to eliminate unnecessary buffer allocations during telemetry reads.",
    ]
    # gcd(25, 21) == 1, so (i % 25, i % 21) is unique for i in range(525) > 200
    return [
        f"In `{DOMAINS[i % len(DOMAINS)]['files_p2'][i % len(DOMAINS[i % len(DOMAINS)]['files_p2'])]}` "
        f"for {DOMAINS[i % len(DOMAINS)]['topic']}, {templates[i % len(templates)].lower()}"
        for i in range(200)
    ]


def build_architect_prompts_bank() -> list[str]:
    """Return a curated bank of >= 150 unique architectural mandates using 11 templates (gcd(25,11)=1)."""
    arch_templates = [
        "Design a 5-phase modular implementation plan for a low-latency LLM prompt compiler.",
        "Architect an autonomous multi-phase test runner that enforces verification gates before commits.",
        "Overhaul the context eviction pipeline to handle 100k-line project trees with zero lag.",
        "Construct a phased blueprint for fine-tuning a 7B parameter coding model on laptop hardware.",
        "Design an ephemeral CLI architecture that offloads all heavy inference to background Ollama.",
        "Design a high-availability distributed raft consensus cluster with zero single point of failure.",
        "Architect a lock-free ring-buffer telemetry ingestion pipeline with sub-millisecond p99 latency.",
        "Design a sandboxed WASI runtime environment with strict memory and CPU fuel isolation.",
        "Architect an incremental B-Tree indexing engine supporting concurrent ACID writes.",
        "Design an eBPF-powered network monitoring daemon that streams packet traces to user-space.",
        "Architect a zero-copy shared memory IPC ring for inter-process telemetry transfer.",
    ]
    # gcd(25, 11) == 1, so (i % 25, i % 11) is unique for i in range(275) > 150
    return [
        f"Architectural Directive for {DOMAINS[i % len(DOMAINS)]['topic']}: "
        f"{arch_templates[i % len(arch_templates)]} Ensure strict phased working sets."
        for i in range(150)
    ]


def build_handoff_prompts_bank() -> list[str]:
    """Return a curated bank of >= 100 unique session handoff requests using 7 templates (gcd(25,7)=1)."""
    handoff_templates = [
        "Current chat has accumulated high context drift after completing initial phases; compile clean bootstrap state.",
        "Context window reaches token limit; generate compact handoff directive with locked files and invariants.",
        "Session turn threshold exceeded; prepare handoff payload preserving verified milestones and active code signatures.",
        "Bootstrap a fresh subagent context with completed phase history evicted and pinned invariants intact.",
        "Re-anchor operational focus onto active phase components while pruning older turn history.",
        "Initialize clean subagent session handoff to prevent degradation across subsequent implementation phases.",
        "Compile structured handoff state to restart workspace execution with zero context drift.",
    ]
    # gcd(25, 7) == 1, so (i % 25, i % 7) is unique for i in range(175) > 100
    return [
        f"Handoff Directive for {DOMAINS[i % len(DOMAINS)]['topic']}: "
        f"{handoff_templates[i % len(handoff_templates)]} Reset drift and bootstrap fresh execution."
        for i in range(100)
    ]


REFINE_PROMPTS_BANK = build_refine_prompts_bank()
ARCHITECT_PROMPTS_BANK = build_architect_prompts_bank()
HANDOFF_PROMPTS_BANK = build_handoff_prompts_bank()
