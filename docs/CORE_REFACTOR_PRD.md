# 🛠️ AEGIS Core Refactor PRD

**Version:** 1.0.0
**Target:** v0.3.0 "Titan"
**Focus:** Stability, Diagnostics, and Memory Efficiency

---

## 1. Problem Statement
The current `aegis-lang` implementation is a valid "Research Prototype" but fails as a "Production Engine" due to:
1.  **O(N) Copy Semantics:** Lists and Objects are deep-copied on assignment.
2.  **Memory Leaks:** Append-only global storage with no reclamation.
3.  **Opaque Errors:** Paradoxical error messages without line numbers.
4.  **Recursion Limits:** Reliance on the host (Rust) stack.

## 2. Technical Solutions

### 2.1 The Reference Model (Fixing Memory & Perf)
**Current:**
```rust
enum Value {
    List(Vec<Value>), // <--- Deep copy! 1M items copied on let b = a;
}
```

**Proposed:** Switch to Python-like Reference Counting.
```rust
use std::rc::Rc;
use std::cell::RefCell;

enum Value {
    List(Rc<RefCell<Vec<Value>>>), // <--- O(1) copy, shared state
    Object(Rc<RefCell<ObjectInstance>>),
}
```
*   **Impact:** `let b = a` becomes instant. Updates to `b` reflect in `a`.
*   **Memory:** `Rc` automatically deallocates when the last reference drops. This acts as a basic Garbage Collector (GC) for acyclic graphs.

### 2.2 Source Spans (Fixing Diagnostics)
**Current:** `Token` has line/col, but `Expr` (AST) loses it.

**Proposed:** Wrap all AST nodes in a `Spanned<T>` struct.
```rust
struct Span { start: usize, end: usize, source_id: usize }

struct Spanned<T> {
    node: T,
    span: Span,
}

// Expr becomes Spanned<ExprKind>
type Expr = Spanned<ExprKind>;
```
*   **Result:** Errors change from `"Variable 'x' not found"` to:
    ```
    Error: Variable 'x' not found
      --> main.aegis:14:5
       |
    14 | let y = x * 2
       |         ^ undefined variable
    ```

### 2.3 The "Manifold" Bytecode VM (Fixing Recursion)
**Current:** Tree-Walking Interpreter (Recursive function calls).

**Proposed:** A Stack-Based Virtual Machine (`aegis-vm`).
1.  **Compiler:** `AST -> Bytecode` (Flatten the tree to linear instructions).
2.  **VM:** A `loop` matching opcodes.
    ```rust
    loop {
        match instructions[ip] {
            Op::Push(v) => stack.push(v),
            Op::Add => {
                let b = stack.pop();
                let a = stack.pop();
                stack.push(a + b);
            }
        }
    }
    ```
*   **Result:** Unlimited recursion depth (limited only by heap). Serializability (save compiled program to disk).

## 3. Implementation Phases

### Phase 1: The "Smart" Pointer (Week 1)
*   **Goal:** Switch `Value` to use `Rc<RefCell>`.
*   **Files:** `interpreter.rs`
*   **Risk:** Low. Mostly search-and-replace standard library functions.
*   **Validation:** Run `benchmark_list_copy.aegis` and see 1000x speedup.

### Phase 2: The "Enlightened" Error (Week 2)
*   **Goal:** Add Source Spans.
*   **Files:** `ast.rs`, `parser.rs`, `lexer.rs`.
*   **Risk:** Medium. Requires passing `Span` through every recursive parser call.
*   **Validation:** Verify precise error pointing in invalid scripts.

### Phase 3: The "Titan" VM (Week 3-6) -- *Long Term*
*   **Goal:** Replace `interpreter.rs` with `compiler.rs` and `vm.rs`.
*   **Risk:** High. Complete rewrite of execution logic.
*   **Transition:** Keep Tree-Walker for `cargo run` / REPL, use VM for production/deployment.

## 4. Success Criteria
1.  **Benchmark:** Copying a list of 10,000 items takes < 1μs (currently ~5ms).
2.  **Diagnostics:** Error messages correctly identify the line number of a bug.
3.  **Stability:** Infinite recursion (`fn f() { f() }`) causes a controlled "Stack Overflow" error, not a process crash.
