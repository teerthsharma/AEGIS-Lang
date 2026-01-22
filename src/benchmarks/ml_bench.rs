//! ═══════════════════════════════════════════════════════════════════════════════
//! ML Benchmark: Escalating Regression & Topological Convergence
//! ═══════════════════════════════════════════════════════════════════════════════
//!
//! Benchmarks for the AEGIS ML engine:
//!   - Model escalation path
//!   - Convergence epochs per model type
//!   - Topological convergence detection
//!   - Final R² scores
//!
//! ═══════════════════════════════════════════════════════════════════════════════

use crate::ml::{EscalatingBenchmark, BenchmarkConfig, BenchmarkResult, TestFunction, 
                ManifoldRegressor, ModelType, ConvergenceDetector, BettiNumbers};
use libm::{sin, exp, sqrt};

// ═══════════════════════════════════════════════════════════════════════════════
// Benchmark Configuration
// ═══════════════════════════════════════════════════════════════════════════════

/// Configuration for ML benchmarks
#[derive(Debug, Clone)]
pub struct MlBenchConfig {
    /// Maximum epochs per test
    pub max_epochs: u32,
    /// Convergence threshold
    pub convergence_threshold: f64,
    /// Number of data points
    pub num_points: usize,
}

impl Default for MlBenchConfig {
    fn default() -> Self {
        Self {
            max_epochs: 500,
            convergence_threshold: 0.01,
            num_points: 50,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Benchmark Results
// ═══════════════════════════════════════════════════════════════════════════════

/// Results from ML benchmarks
#[derive(Debug, Clone)]
pub struct MlBenchResults {
    /// Results by test function
    pub sine_result: FunctionResult,
    pub polynomial_result: FunctionResult,
    pub exponential_result: FunctionResult,
    pub mixture_result: FunctionResult,
    /// Model escalation tracking
    pub escalation_path: EscalationPath,
    /// Topological convergence accuracy
    pub topo_convergence_accuracy: f64,
}

/// Result for a single test function
#[derive(Debug, Clone, Default)]
pub struct FunctionResult {
    /// Converged successfully
    pub converged: bool,
    /// Epochs to convergence
    pub epochs: u32,
    /// Final model used
    pub final_model: ModelName,
    /// Final error (MSE)
    pub final_error: f64,
    /// Final convergence score
    pub convergence_score: f64,
}

/// Model name for results
#[derive(Debug, Clone, Default)]
pub struct ModelName(pub [u8; 16]);

impl ModelName {
    fn from_str(s: &str) -> Self {
        let mut arr = [0u8; 16];
        for (i, c) in s.bytes().take(16).enumerate() {
            arr[i] = c;
        }
        ModelName(arr)
    }
    
    pub fn as_str(&self) -> &str {
        let len = self.0.iter().position(|&c| c == 0).unwrap_or(16);
        core::str::from_utf8(&self.0[..len]).unwrap_or("unknown")
    }
}

/// Escalation path statistics
#[derive(Debug, Clone, Default)]
pub struct EscalationPath {
    /// Times Linear was final model
    pub linear_final: u32,
    /// Times Polynomial was final model
    pub polynomial_final: u32,
    /// Times RBF was final model
    pub rbf_final: u32,
    /// Times GP was final model
    pub gp_final: u32,
    /// Average escalations per function
    pub avg_escalations: f64,
}

impl MlBenchResults {
    fn new() -> Self {
        Self {
            sine_result: FunctionResult::default(),
            polynomial_result: FunctionResult::default(),
            exponential_result: FunctionResult::default(),
            mixture_result: FunctionResult::default(),
            escalation_path: EscalationPath::default(),
            topo_convergence_accuracy: 0.0,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// ML Benchmark Implementation
// ═══════════════════════════════════════════════════════════════════════════════

/// Run comprehensive ML benchmarks
pub struct MlBenchmark {
    config: MlBenchConfig,
}

impl MlBenchmark {
    pub fn new(config: MlBenchConfig) -> Self {
        Self { config }
    }

    /// Run all ML benchmarks
    pub fn run_all(&self) -> MlBenchResults {
        let mut results = MlBenchResults::new();
        
        // Run each test function
        results.sine_result = self.run_test_function(TestFunction::Sine);
        results.polynomial_result = self.run_test_function(TestFunction::Polynomial);
        results.exponential_result = self.run_test_function(TestFunction::Exponential);
        results.mixture_result = self.run_test_function(TestFunction::Mixture);
        
        // Compute escalation statistics
        results.escalation_path = self.compute_escalation_stats(&[
            &results.sine_result,
            &results.polynomial_result,
            &results.exponential_result,
            &results.mixture_result,
        ]);
        
        // Test topological convergence detection
        results.topo_convergence_accuracy = self.test_topo_convergence();
        
        results
    }

    /// Run benchmark on a specific test function
    fn run_test_function(&self, func: TestFunction) -> FunctionResult {
        let config = BenchmarkConfig {
            max_epochs: self.config.max_epochs,
            convergence_epsilon: self.config.convergence_threshold,
            escalate_threshold: 0.1,
            stability_window: 10,
        };
        
        let mut bench = EscalatingBenchmark::<3>::new(config);
        
        // Generate test data
        let data = self.generate_function_data(func);
        for (x, y) in data.iter() {
            bench.add_data([*x, 0.0, 0.0], *y);
        }
        
        // Run benchmark
        let result = bench.run();
        
        FunctionResult {
            converged: result.converged,
            epochs: result.epochs,
            final_model: self.model_to_name(result.final_model),
            final_error: result.final_error,
            convergence_score: result.convergence_score,
        }
    }

    /// Generate data for test function
    fn generate_function_data(&self, func: TestFunction) -> [(f64, f64); 64] {
        let mut data = [(0.0, 0.0); 64];
        
        for i in 0..self.config.num_points.min(64) {
            let x = (i as f64 / self.config.num_points as f64) * 6.28 - 3.14;
            
            data[i] = match func {
                TestFunction::Sine => (x, sin(x)),
                TestFunction::Polynomial => (x, x * x - 2.0 * x + 1.0),
                TestFunction::Exponential => (x, exp(-x * x)),
                TestFunction::Mixture => (x, sin(x) + 0.5 * sin(3.0 * x)),
                TestFunction::Step => (x, if x > 0.0 { 1.0 } else { -1.0 }),
            };
        }
        
        data
    }

    /// Convert model type to name
    fn model_to_name(&self, model: ModelType) -> ModelName {
        match model {
            ModelType::Linear => ModelName::from_str("Linear"),
            ModelType::Polynomial(d) => {
                match d {
                    2 => ModelName::from_str("Polynomial-2"),
                    3 => ModelName::from_str("Polynomial-3"),
                    _ => ModelName::from_str("Polynomial"),
                }
            }
            ModelType::Rbf { .. } => ModelName::from_str("RBF"),
            ModelType::GaussianProcess { .. } => ModelName::from_str("GP"),
            ModelType::GeodesicRegression => ModelName::from_str("Geodesic"),
        }
    }

    /// Compute escalation statistics
    fn compute_escalation_stats(&self, results: &[&FunctionResult]) -> EscalationPath {
        let mut stats = EscalationPath::default();
        let mut total_escalations = 0u32;
        
        for result in results {
            let model_str = result.final_model.as_str();
            
            if model_str.starts_with("Linear") {
                stats.linear_final += 1;
            } else if model_str.starts_with("Polynomial") {
                stats.polynomial_final += 1;
                total_escalations += 1;
            } else if model_str.starts_with("RBF") {
                stats.rbf_final += 1;
                total_escalations += 2;
            } else if model_str.starts_with("GP") {
                stats.gp_final += 1;
                total_escalations += 3;
            }
        }
        
        stats.avg_escalations = total_escalations as f64 / results.len() as f64;
        stats
    }

    /// Test topological convergence detection
    fn test_topo_convergence(&self) -> f64 {
        let mut detector = ConvergenceDetector::new(self.config.convergence_threshold, 5);
        let mut correct = 0u32;
        let mut total = 0u32;
        
        // Scenario 1: Clear convergence (decreasing error, stable topology)
        for i in 0..20 {
            let error = 1.0 / (i as f64 + 1.0);
            let betti = BettiNumbers::new(1, 0);
            detector.record_epoch(betti, 0.01, error);
        }
        
        if detector.is_converged() {
            correct += 1;
        }
        total += 1;
        
        // Reset and test non-convergence
        detector.reset();
        
        // Scenario 2: Oscillating (should not converge)
        for i in 0..20 {
            let error = 0.5 + 0.4 * sin(i as f64);
            let betti = BettiNumbers::new((i % 3 + 1) as u32, (i % 2) as u32);
            detector.record_epoch(betti, 0.5, error);
        }
        
        if !detector.is_converged() {
            correct += 1;
        }
        total += 1;
        
        // Scenario 3: Converges late
        detector.reset();
        for i in 0..10 {
            let error = 0.5;
            let betti = BettiNumbers::new((i % 3 + 1) as u32, 0);
            detector.record_epoch(betti, 0.3, error);
        }
        for i in 0..15 {
            let error = 0.001;
            let betti = BettiNumbers::new(1, 0);
            detector.record_epoch(betti, 0.001, error);
        }
        
        if detector.is_converged() {
            correct += 1;
        }
        total += 1;
        
        correct as f64 / total as f64
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Unit Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sine_regression() {
        let bench = MlBenchmark::new(MlBenchConfig::default());
        let results = bench.run_all();
        
        println!("\n═══ ML Benchmark: Sine ═══");
        println!("Converged: {}", results.sine_result.converged);
        println!("Epochs: {}", results.sine_result.epochs);
        println!("Final model: {}", results.sine_result.final_model.as_str());
        println!("Final error: {:.6}", results.sine_result.final_error);
        println!("Convergence score: {:.3}", results.sine_result.convergence_score);
    }

    #[test]
    fn test_escalation_path() {
        let bench = MlBenchmark::new(MlBenchConfig::default());
        let results = bench.run_all();
        
        println!("\n═══ Escalation Statistics ═══");
        println!("Linear final: {}", results.escalation_path.linear_final);
        println!("Polynomial final: {}", results.escalation_path.polynomial_final);
        println!("RBF final: {}", results.escalation_path.rbf_final);
        println!("GP final: {}", results.escalation_path.gp_final);
        println!("Avg escalations: {:.2}", results.escalation_path.avg_escalations);
    }

    #[test]
    fn test_topo_convergence_detection() {
        let bench = MlBenchmark::new(MlBenchConfig::default());
        let results = bench.run_all();
        
        assert!(results.topo_convergence_accuracy >= 0.6,
            "Topological convergence accuracy should be >= 60%");
        println!("Topological convergence accuracy: {:.1}%", 
            results.topo_convergence_accuracy * 100.0);
    }

    #[test]
    fn test_full_ml_benchmark() {
        let bench = MlBenchmark::new(MlBenchConfig {
            max_epochs: 200,
            convergence_threshold: 0.05,
            num_points: 30,
        });
        let results = bench.run_all();
        
        println!("\n═══ Full ML Benchmark Results ═══");
        println!("Sine: {} in {} epochs ({})", 
            if results.sine_result.converged { "✓" } else { "✗" },
            results.sine_result.epochs,
            results.sine_result.final_model.as_str());
        println!("Polynomial: {} in {} epochs ({})", 
            if results.polynomial_result.converged { "✓" } else { "✗" },
            results.polynomial_result.epochs,
            results.polynomial_result.final_model.as_str());
        println!("Exponential: {} in {} epochs ({})", 
            if results.exponential_result.converged { "✓" } else { "✗" },
            results.exponential_result.epochs,
            results.exponential_result.final_model.as_str());
        println!("Mixture: {} in {} epochs ({})", 
            if results.mixture_result.converged { "✓" } else { "✗" },
            results.mixture_result.epochs,
            results.mixture_result.final_model.as_str());
    }
}
