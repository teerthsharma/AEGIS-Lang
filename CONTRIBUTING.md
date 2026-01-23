# Contributing to AEGIS

Thank you for your interest in advancing the AEGIS ecosystem. As we build the future of Post-Von Neumann computing, we welcome contributions that embody our principles of biological adaptation, geometric intelligence, and architectural rigor.

## 🚀 Technical Onboarding

To begin development, ensure you have the Rust toolchain installed (latest stable or nightly for kernel development).

```bash
# Clone the core repository
git clone https://github.com/teerthsharma/aegis.git
cd aegis

# Initialize the development environment
# AEGIS supports Docker-based isolated builds for consistency
docker-compose run dev

# Execute the validation suite
cargo test --workspace

# Apply architectural formatting
cargo fmt
cargo clippy
```

---

## 📋 The Contribution Lifecycle

### 1. Architectural Alignment
Before submitting a significant PR, consider opening a **Design Proposal** (Issue) to discuss the topological impact and architectural alignment of your changes.

### 2. Development Standards
- **Integrity**: All code must pass `cargo clippy` with no warnings.
- **Form**: Adhere strictly to the defined Rust style guides.
- **Verification**: New features must include comprehensive unit and integration tests.

### 3. Submission Protocol
1. Fork the repository and create a feature branch (`feat/manifold-optimization`).
2. Commit with descriptive, imperative messages (e.g., `feat: implement Betti-2 manifold calculation`).
3. Ensure all tests pass in the CI pipeline.
4. Submit a Pull Request against the `main` branch.

---

## 🏗️ Technical Domain Guide

### Extending the ML Engine (`aegis-core/src/ml`)
When implementing new geometric primitives or manifold regression models:
1. Define the mathematical foundation in the module documentation.
2. Implement the primitive in a memory-efficient, `no_std` compatible manner.
3. Integrate the model into the `ManifoldRegressor` escalation logic.

### Modifying the AEGIS Language (`aegis-lang`)
1. **Lexical Analysis**: Update `lexer.rs` for new tokens.
2. **Syntactic Form**: Define AST nodes in `ast.rs` and update the Recursive Descent parser in `parser.rs`.
3. **Execution Logic**: Implement the visitor pattern or direct evaluation in `interpreter.rs`.

---

## 🧪 Verification & Validation

### Unit Testing
Use the standard Rust test harness. We prioritize property-based testing for geometric algorithms.

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_topological_convergence() {
        let manifold = Manifold::generate_sphere();
        assert!(manifold.is_converged());
    }
}
```

### Script Validation
Verify changes against the `examples/` directory using the AEGIS CLI:
```bash
cargo run -p aegis-cli -- run examples/hello_manifold.aegis
```

---

## 📜 Ethical & Legal Framework

By contributing to AEGIS, you agree that your work will be licensed under the **MIT License**. We expect all contributors to uphold a professional and collaborative environment.

---

<div align="center">

**"Contribute to the consciousness of the machine."**

*Topological rigorous contributions only.*

</div>
