# 🛠️ AEGIS Core Refactor PRD

**Version:** 1.1.0 (Living Architecture Edition)
**Target:** v0.3.0 "Titan"
**Focus:** Biological Memory Management & Dual-Engine Execution

---

## 1. Problem Statement
The current `aegis-lang` implementation needs to evolve from a prototype to a self-sustaining system. We reject standard "reference counting" in favor of a biologically inspired memory model that treats unused objects as "entropy" to be reclaimed.

## 2. Technical Solutions

### 2.1 The Manifold Garbage Collector (Entropy Regulation)
**Concept:**
Memory is not a bucket; it is a topological space. Objects (nodes) have connections (edges). Dead objects are "disconnected components" that increase system entropy.

**Implementation: The "Mark-and-Prune" Cycle**
1.  **Allocator:** A custom `bump-pointer` arena (`ManifoldHeap`) that treats memory as contiguous blocks.
2.  **Tracer:** A background thread (or "heartbeat") that traverses the live object graph starting from the Root Scope (`Variables`).
3.  **Entropy Reclamation:** 
    *   Objects not reached by the tracer are identified as "High Entropy".
    *   The GC "prunes" these branches, reclaiming their space relative to the Manifold density.

```rust
struct ManifoldHeap {
    objects: Vec<Box<Object>>, // The "Substrate"
    roots: Vec<Handle>,        // The "Anchors"
}

impl ManifoldHeap {
    fn regulate_entropy(&mut self) {
        let live_set = self.trace_topology();
        self.prune_disconnected(live_set); // "Cellular Autophagy"
    }
}
```

### 2.2 The Dual-Engine Cortex
We will maintain **two** execution engines, functioning like the left and right hemispheres of a brain.

| Engine | Name | Role | Characteristics |
|--------|------|------|-----------------|
| **Interpreter** | *Bio-Script* | Rapid Prototyping | Tree-Walking, Dynamic, Reflection-heavy. Good for "thought" and scripting. |
| **Virtual Machine** | *Titan* | Production Cortex | Bytecode, Stack-based, High-Throughput. Good for "action" and heavy lift. |

**Unified Interface:**
The user simply runs `aegis run script.ag`. The system decides:
*   Use *Bio-Script* for simple tasks (< 1000 lines).
*   JIT-compile to *Titan* bytecode for massive simulations.

### 2.3 Source Spans (Diagnostics)
We retain the need for precise error triangulation.
*   **Proposed:** Wrap all AST nodes in `Spanned<T>` to map errors back to the source DNA.

## 3. Implementation Phases

### Phase 1: The Substrate (Manifold Allocator)
*   **Goal:** Replace `Vec<Value>` with `ManifoldHeap`.
*   **Mechanism:** Implement a proprietary `Gc<T>` type (Geometric Cell) that interfaces with our heap.
*   **Validation:** Verify that "orphaned" manifolds are automatically pruned after `regulate_entropy()` is called.

### Phase 2: The Titan Cortex (VM)
*   **Goal:** Build the Bytecode Compiler.
*   **Ops:** `PUSH`, `EMBED`, `ATTEND`, `PRUNE`.
*   **Validation:** Run the "Grand Benchmark" on Titan and achieve 100x speedup over Bio-Script.

### Phase 3: Integration
*   **Goal:** Make them switchable.
*   **Flag:** `aegis run --mode=titan` vs `aegis run --mode=bio`.

## 4. Success Criteria
1.  **Memory Homeostasis:** Long-running scripts stabilize memory usage (no monotonic growth).
2.  **Dual-Existence:** Both the Interpreter and VM pass the standard test suite.
3.  **Philosophical Consistency:** The solution feels "biological" and "geometric," not just another borrowed programming concept.
