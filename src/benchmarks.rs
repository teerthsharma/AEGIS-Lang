//! ═══════════════════════════════════════════════════════════════════════════════
//! AEGIS Rigorous Benchmark Suite
//! ═══════════════════════════════════════════════════════════════════════════════
//!
//! "Run regression benchmarks infinitely harder each until perfect"
//!
//! This module provides comprehensive benchmarks for:
//! - Manifold embedding performance
//! - Regression model accuracy
//! - Topological convergence speed
//! - Escalation efficiency
//! ═══════════════════════════════════════════════════════════════════════════════

#![allow(dead_code)]

use crate::ml::{
    regressor::{ManifoldRegressor, ModelType},
    convergence::{ConvergenceDetector, BettiNumbers},
    benchmark::{EscalatingBenchmark, BenchmarkConfig, TestFunction, generate_test_function},
};

// ═══════════════════════════════════════════════════════════════════════════════
// Benchmark Test Suite
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn benchmark_linear_regression() {
    let mut regressor: ManifoldRegressor<3> = ManifoldRegressor::new(ModelType::Linear);
    
    // Add sine wave data
    for i in 0..64 {
        let x = (i as f64) * 0.1;
        let y = libm::sin(x);
        regressor.add_point([x, x * 0.5, x * 0.25], y);
    }
    
    let error = regressor.fit();
    
    // Linear should have significant error on sine
    assert!(error > 0.01, "Linear regression should not perfectly fit sine");
    assert!(error < 1.0, "Linear regression should have bounded error");
}

#[test]
fn benchmark_polynomial_regression() {
    let mut regressor: ManifoldRegressor<3> = ManifoldRegressor::new(ModelType::Polynomial(3));
    
    // Add polynomial data
    for i in 0..64 {
        let x = (i as f64) * 0.1 - 3.0;
        let y = 0.5 * x * x - 2.0 * x + 1.0;
        regressor.add_point([x, x * 0.5, x * 0.25], y);
    }
    
    let error = regressor.fit();
    
    // Polynomial should fit polynomial data well
    assert!(error < 0.1, "Polynomial should fit polynomial data: got {}", error);
}

#[test]
fn benchmark_escalating_convergence() {
    let config = BenchmarkConfig {
        epsilon: 1e-4,
        max_epochs: 50,
        escalation_patience: 5,
        stability_window: 3,
        auto_escalate: true,
    };
    
    let mut benchmark: EscalatingBenchmark<3> = EscalatingBenchmark::new(config);
    
    // Add sine wave data
    let data = generate_test_function(TestFunction::Sine, 64);
    for (x, y) in data.iter() {
        benchmark.add_data([*x, x * 0.5, x * 0.25], *y);
    }
    
    let result = benchmark.run();
    
    // Should escalate at least once
    assert!(result.escalations >= 1, "Should escalate model complexity");
    
    // Should eventually converge or get close
    assert!(result.final_error < 0.5, "Should reduce error significantly");
}

#[test]
fn benchmark_betti_stability() {
    let mut detector = ConvergenceDetector::new(1e-6, 5);
    
    // Simulate converging sequence
    for i in 0..10 {
        let betti = if i < 5 {
            BettiNumbers::new(3 - i as u32 / 2, 1)
        } else {
            BettiNumbers::new(1, 0) // Stable
        };
        
        let drift = 1.0 / (i + 1) as f64;
        let error = 0.1 / (i + 1) as f64;
        
        detector.record_epoch(betti, drift, error);
    }
    
    // Should detect convergence via Betti stability
    let score = detector.convergence_score();
    assert!(score > 0.5, "Convergence score should be high: {}", score);
}

#[test]
fn benchmark_all_test_functions() {
    let functions = [
        TestFunction::Sine,
        TestFunction::Polynomial,
        TestFunction::Exponential,
        TestFunction::Mixture,
    ];
    
    for func in functions.iter() {
        let config = BenchmarkConfig {
            epsilon: 1e-3,
            max_epochs: 30,
            escalation_patience: 5,
            stability_window: 3,
            auto_escalate: true,
        };
        
        let mut benchmark: EscalatingBenchmark<3> = EscalatingBenchmark::new(config);
        
        let data = generate_test_function(*func, 64);
        for (x, y) in data.iter() {
            benchmark.add_data([*x, x * 0.5, x * 0.25], *y);
        }
        
        let result = benchmark.run();
        
        assert!(
            result.final_error < 1.0,
            "{:?} should have bounded error: {}",
            func,
            result.final_error
        );
    }
}

#[test]
fn benchmark_model_upgrade_sequence() {
    let mut regressor: ManifoldRegressor<3> = ManifoldRegressor::new(ModelType::Linear);
    
    // Verify upgrade sequence
    assert_eq!(regressor.model(), ModelType::Linear);
    
    regressor.upgrade_model();
    assert!(matches!(regressor.model(), ModelType::Polynomial(2)));
    
    regressor.upgrade_model();
    assert!(matches!(regressor.model(), ModelType::Polynomial(3)));
    
    // Continue upgrading...
    for _ in 0..5 {
        regressor.upgrade_model();
    }
    
    // Should eventually reach highest complexity
    let complexity = regressor.model().complexity();
    assert!(complexity >= 5, "Should reach high complexity: {}", complexity);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Performance Benchmarks (for manual profiling)
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
#[ignore] // Run with: cargo test benchmark_performance -- --ignored
fn benchmark_performance_large_dataset() {
    let config = BenchmarkConfig {
        epsilon: 1e-6,
        max_epochs: 100,
        escalation_patience: 10,
        stability_window: 5,
        auto_escalate: true,
    };
    
    let mut benchmark: EscalatingBenchmark<3> = EscalatingBenchmark::new(config);
    
    // Large dataset
    for i in 0..256 {
        let x = (i as f64) * 0.05;
        let y = libm::sin(x) * libm::exp(-x / 10.0);
        benchmark.add_data([x, x * 0.5, x * 0.25], y);
    }
    
    let result = benchmark.run();
    
    println!("═══════════════════════════════════════════════════════════════");
    println!("Performance Benchmark Results:");
    println!("  Final Model: {:?}", result.final_model);
    println!("  Epochs: {}", result.epochs);
    println!("  Escalations: {}", result.escalations);
    println!("  Final Error: {:.6}", result.final_error);
    println!("  Converged: {}", result.converged);
    println!("  Final Betti: ({}, {})", result.final_betti.beta_0, result.final_betti.beta_1);
    println!("═══════════════════════════════════════════════════════════════");
    
    assert!(result.final_error < 0.1, "Should converge to low error");
}
