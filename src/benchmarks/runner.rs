//! ═══════════════════════════════════════════════════════════════════════════════
//! Benchmark Runner: Unified Test Harness
//! ═══════════════════════════════════════════════════════════════════════════════
//!
//! Runs all AEGIS benchmarks and collects unified results.
//! This is the main entry point for the benchmark suite.
//!
//! ═══════════════════════════════════════════════════════════════════════════════

use crate::benchmarks::governor_bench::{GovernorBenchmark, GovernorBenchConfig, 
                                         GovernorBenchResults, LyapunovResults, StressTestResults};
use crate::benchmarks::topology_bench::{TopologyBenchmark, TopologyBenchConfig, TopologyBenchResults};
use crate::benchmarks::aether_bench::{AetherBenchmark, AetherBenchConfig, AetherBenchResults};
use crate::benchmarks::manifold_bench::{ManifoldBenchmark, ManifoldBenchConfig, ManifoldBenchResults};
use crate::benchmarks::ml_bench::{MlBenchmark, MlBenchConfig, MlBenchResults};

// ═══════════════════════════════════════════════════════════════════════════════
// Unified Results
// ═══════════════════════════════════════════════════════════════════════════════

/// Complete benchmark suite results
#[derive(Debug)]
pub struct BenchmarkSuiteResults {
    /// Governor benchmark results (one per target rate)
    pub governor: GovernorSuiteResults,
    /// Topology benchmark results
    pub topology: TopologyBenchResults,
    /// AETHER benchmark results
    pub aether: AetherBenchResults,
    /// Manifold benchmark results
    pub manifold: ManifoldBenchResults,
    /// ML engine benchmark results
    pub ml: MlBenchResults,
}

/// Governor results with all sub-tests
#[derive(Debug)]
pub struct GovernorSuiteResults {
    /// Convergence at 10Hz
    pub convergence_10hz: GovernorBenchResults,
    /// Convergence at 100Hz
    pub convergence_100hz: GovernorBenchResults,
    /// Convergence at 1000Hz
    pub convergence_1000hz: GovernorBenchResults,
    /// Convergence at 10000Hz
    pub convergence_10000hz: GovernorBenchResults,
    /// Lyapunov analysis
    pub lyapunov: LyapunovResults,
    /// Stress test
    pub stress: StressTestResults,
}

// ═══════════════════════════════════════════════════════════════════════════════
// Benchmark Suite Runner
// ═══════════════════════════════════════════════════════════════════════════════

/// Run the complete benchmark suite
pub struct BenchmarkSuiteRunner;

impl BenchmarkSuiteRunner {
    /// Run all benchmarks with default configuration
    pub fn run_all() -> BenchmarkSuiteResults {
        Self::run_with_config(
            GovernorBenchConfig::default(),
            TopologyBenchConfig::default(),
            AetherBenchConfig::default(),
            ManifoldBenchConfig::default(),
            MlBenchConfig::default(),
        )
    }

    /// Run with custom configurations
    pub fn run_with_config(
        gov_config: GovernorBenchConfig,
        topo_config: TopologyBenchConfig,
        aether_config: AetherBenchConfig,
        manifold_config: ManifoldBenchConfig,
        ml_config: MlBenchConfig,
    ) -> BenchmarkSuiteResults {
        // Governor benchmarks
        let gov_bench = GovernorBenchmark::new(gov_config);
        let gov_results = gov_bench.run_all();
        let lyapunov = gov_bench.run_lyapunov_analysis();
        let stress = gov_bench.run_stress_test();
        
        let governor = GovernorSuiteResults {
            convergence_10hz: gov_results[0].clone(),
            convergence_100hz: gov_results[1].clone(),
            convergence_1000hz: gov_results[2].clone(),
            convergence_10000hz: gov_results[3].clone(),
            lyapunov,
            stress,
        };
        
        // Topology benchmarks
        let topo_bench = TopologyBenchmark::new(topo_config);
        let topology = topo_bench.run_all();
        
        // AETHER benchmarks
        let aether_bench = AetherBenchmark::new(aether_config);
        let aether = aether_bench.run_all();
        
        // Manifold benchmarks
        let manifold_bench = ManifoldBenchmark::new(manifold_config);
        let manifold = manifold_bench.run_all();
        
        // ML benchmarks
        let ml_bench = MlBenchmark::new(ml_config);
        let ml = ml_bench.run_all();
        
        BenchmarkSuiteResults {
            governor,
            topology,
            aether,
            manifold,
            ml,
        }
    }

    /// Print summary of results
    pub fn print_summary(results: &BenchmarkSuiteResults) {
        #[cfg(test)]
        {
            println!("\n╔═══════════════════════════════════════════════════════════════╗");
            println!("║           AEGIS Benchmark Suite Results                       ║");
            println!("╠═══════════════════════════════════════════════════════════════╣");
            
            // Governor
            println!("║ GOVERNOR (PID-on-Manifold)                                    ║");
            println!("║   Lyapunov stable: {}                                         ║", 
                if results.governor.lyapunov.is_stable { "YES" } else { "NO " });
            println!("║   Stress bounded:  {}                                         ║",
                if results.governor.stress.all_bounded { "YES" } else { "NO " });
            
            // Topology  
            println!("╠═══════════════════════════════════════════════════════════════╣");
            println!("║ TOPOLOGY (Gatekeeper)                                         ║");
            println!("║   NOP Sled TPR:    {:5.1}%                                     ║", 
                results.topology.nop_sled_tpr * 100.0);
            println!("║   ROP Chain TPR:   {:5.1}%                                     ║",
                results.topology.rop_chain_tpr * 100.0);
            println!("║   Legitimate FPR:  {:5.1}%                                     ║",
                results.topology.legitimate_fpr * 100.0);
            
            // AETHER
            println!("╠═══════════════════════════════════════════════════════════════╣");
            println!("║ AETHER (Hierarchical Sparse Attention)                        ║");
            println!("║   Pruning @0.5:    {:5.1}%                                     ║",
                results.aether.pruning_ratios[2] * 100.0);
            println!("║   Upper bound:     {:5.1}%                                     ║",
                results.aether.upper_bound_tightness * 100.0);
            println!("║   Hierarchical:    {:5.1}%                                     ║",
                results.aether.hierarchical_efficiency * 100.0);
            
            // Manifold
            println!("╠═══════════════════════════════════════════════════════════════╣");
            println!("║ MANIFOLD (Embedding & Sparse Attention)                       ║");
            println!("║   Sine embed:      {:5.1}%                                     ║",
                results.manifold.embedding_quality.sine_quality * 100.0);
            println!("║   Circle β₀,β₁:   ({}, {})                                       ║",
                results.manifold.betti_numbers.circle_betti.0,
                results.manifold.betti_numbers.circle_betti.1);
            
            // ML
            println!("╠═══════════════════════════════════════════════════════════════╣");
            println!("║ ML ENGINE (Escalating Regression)                             ║");
            println!("║   Topo convergence: {:5.1}%                                    ║",
                results.ml.topo_convergence_accuracy * 100.0);
            println!("║   Avg escalations:  {:5.2}                                     ║",
                results.ml.escalation_path.avg_escalations);
            
            println!("╚═══════════════════════════════════════════════════════════════╝");
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Unit Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_full_benchmark_suite() {
        let results = BenchmarkSuiteRunner::run_all();
        BenchmarkSuiteRunner::print_summary(&results);
        
        // Validate critical metrics
        assert!(results.governor.lyapunov.is_stable, 
            "Governor should be Lyapunov stable");
        assert!(results.governor.stress.all_bounded,
            "Governor should stay bounded under stress");
        assert!(results.topology.nop_sled_tpr >= 0.80,
            "NOP sled detection should be >= 80%");
        assert!(results.aether.drift_detection_accuracy >= 0.6,
            "Drift detection should be >= 60%");
    }

    #[test]
    fn test_quick_benchmark() {
        // Quick benchmark with reduced iterations
        let results = BenchmarkSuiteRunner::run_with_config(
            GovernorBenchConfig {
                iterations: 1000,
                ..Default::default()
            },
            TopologyBenchConfig {
                samples_per_category: 20,
                ..Default::default()
            },
            AetherBenchConfig {
                num_blocks: 8,
                ..Default::default()
            },
            ManifoldBenchConfig {
                embedding_points: 50,
                ..Default::default()
            },
            MlBenchConfig {
                max_epochs: 100,
                num_points: 20,
                ..Default::default()
            },
        );
        
        println!("\n═══ Quick Benchmark Complete ═══");
        println!("Governor stable: {}", results.governor.lyapunov.is_stable);
        println!("Topology samples: {}", results.topology.total_samples);
    }
}
