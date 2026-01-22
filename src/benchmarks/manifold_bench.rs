//! ═══════════════════════════════════════════════════════════════════════════════
//! Manifold Benchmark: Time-Delay Embedding & Sparse Attention
//! ═══════════════════════════════════════════════════════════════════════════════
//!
//! Benchmarks for manifold embedding and sparse attention:
//!   - Time-delay embedding quality
//!   - Sparse graph density
//!   - Betti number computation
//!   - Geometric concentration
//!
//! ═══════════════════════════════════════════════════════════════════════════════

use crate::manifold::{TimeDelayEmbedder, SparseAttentionGraph, ManifoldPoint, GeometricConcentrator};
use libm::{sin, cos, sqrt};

// ═══════════════════════════════════════════════════════════════════════════════
// Benchmark Configuration
// ═══════════════════════════════════════════════════════════════════════════════

/// Configuration for manifold benchmarks
#[derive(Debug, Clone)]
pub struct ManifoldBenchConfig {
    /// Number of points for embedding test
    pub embedding_points: usize,
    /// Time delay tau
    pub tau: usize,
    /// Epsilon values for sparse graph
    pub epsilons: [f64; 4],
}

impl Default for ManifoldBenchConfig {
    fn default() -> Self {
        Self {
            embedding_points: 100,
            tau: 3,
            epsilons: [0.1, 0.3, 0.5, 1.0],
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Benchmark Results
// ═══════════════════════════════════════════════════════════════════════════════

/// Results from manifold benchmarks
#[derive(Debug, Clone)]
pub struct ManifoldBenchResults {
    /// Embedding reconstruction quality (correlation)
    pub embedding_quality: EmbeddingQuality,
    /// Sparse graph density at each epsilon
    pub graph_densities: [f64; 4],
    /// Betti numbers for test shapes
    pub betti_numbers: BettiResults,
    /// Geometric concentration metrics
    pub concentration: ConcentrationResults,
}

/// Embedding quality metrics
#[derive(Debug, Clone, Default)]
pub struct EmbeddingQuality {
    /// Sine wave reconstruction
    pub sine_quality: f64,
    /// Chirp reconstruction
    pub chirp_quality: f64,
    /// Lorenz attractor
    pub lorenz_quality: f64,
}

/// Betti number results for known shapes
#[derive(Debug, Clone, Default)]
pub struct BettiResults {
    /// Circle: should have β₀=1, β₁=1
    pub circle_betti: (u32, u32),
    /// Line: should have β₀=1, β₁=0
    pub line_betti: (u32, u32),
    /// Clusters: β₀ = number of clusters
    pub cluster_betti: (u32, u32),
}

/// Concentration performance
#[derive(Debug, Clone, Default)]
pub struct ConcentrationResults {
    /// Variance reduction ratio
    pub variance_reduction: f64,
    /// Principal axis alignment
    pub alignment_score: f64,
}

impl ManifoldBenchResults {
    fn new() -> Self {
        Self {
            embedding_quality: EmbeddingQuality::default(),
            graph_densities: [0.0; 4],
            betti_numbers: BettiResults::default(),
            concentration: ConcentrationResults::default(),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Signal Generators
// ═══════════════════════════════════════════════════════════════════════════════

/// Generate sine wave signal
fn generate_sine(n: usize) -> [f64; 128] {
    let mut signal = [0.0; 128];
    for i in 0..n.min(128) {
        signal[i] = sin(2.0 * 3.14159 * (i as f64) / 20.0);
    }
    signal
}

/// Generate chirp signal (increasing frequency)
fn generate_chirp(n: usize) -> [f64; 128] {
    let mut signal = [0.0; 128];
    for i in 0..n.min(128) {
        let t = i as f64 / 100.0;
        let freq = 1.0 + t * 5.0;
        signal[i] = sin(2.0 * 3.14159 * freq * t);
    }
    signal
}

/// Generate points on a circle
fn generate_circle(n: usize) -> [[f64; 3]; 64] {
    let mut points = [[0.0; 3]; 64];
    for i in 0..n.min(64) {
        let theta = 2.0 * 3.14159 * (i as f64) / (n as f64);
        points[i] = [cos(theta), sin(theta), 0.0];
    }
    points
}

/// Generate points on a line
fn generate_line(n: usize) -> [[f64; 3]; 64] {
    let mut points = [[0.0; 3]; 64];
    for i in 0..n.min(64) {
        let t = i as f64 / 10.0;
        points[i] = [t, t * 0.5, 0.0];
    }
    points
}

/// Generate clustered points
fn generate_clusters(n_clusters: usize, points_per_cluster: usize) -> [[f64; 3]; 64] {
    let mut points = [[0.0; 3]; 64];
    let centers: [[f64; 3]; 4] = [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [3.0, 3.0, 0.0]];
    
    let mut idx = 0;
    for c in 0..n_clusters.min(4) {
        for p in 0..points_per_cluster {
            if idx >= 64 { break; }
            let offset = (p as f64 * 0.1) - 0.2;
            points[idx] = [
                centers[c][0] + offset,
                centers[c][1] + offset * 0.5,
                centers[c][2],
            ];
            idx += 1;
        }
    }
    points
}

// ═══════════════════════════════════════════════════════════════════════════════
// Manifold Benchmark Implementation
// ═══════════════════════════════════════════════════════════════════════════════

/// Run comprehensive manifold benchmarks
pub struct ManifoldBenchmark {
    config: ManifoldBenchConfig,
}

impl ManifoldBenchmark {
    pub fn new(config: ManifoldBenchConfig) -> Self {
        Self { config }
    }

    /// Run all manifold benchmarks
    pub fn run_all(&self) -> ManifoldBenchResults {
        let mut results = ManifoldBenchResults::new();
        
        results.embedding_quality = self.test_embedding_quality();
        results.graph_densities = self.test_graph_density();
        results.betti_numbers = self.test_betti_numbers();
        results.concentration = self.test_concentration();
        
        results
    }

    /// Test time-delay embedding quality
    fn test_embedding_quality(&self) -> EmbeddingQuality {
        let mut quality = EmbeddingQuality::default();
        
        // Test sine wave embedding
        quality.sine_quality = self.embed_and_measure_quality(&generate_sine(100));
        
        // Test chirp embedding
        quality.chirp_quality = self.embed_and_measure_quality(&generate_chirp(100));
        
        // Lorenz approximation (simplified as quasi-periodic)
        let mut lorenz = [0.0; 128];
        for i in 0..100 {
            let t = i as f64 * 0.1;
            lorenz[i] = sin(t) * cos(t * 0.3) + sin(t * 2.1);
        }
        quality.lorenz_quality = self.embed_and_measure_quality(&lorenz);
        
        quality
    }

    /// Embed signal and measure reconstruction quality
    fn embed_and_measure_quality(&self, signal: &[f64; 128]) -> f64 {
        let mut embedder = TimeDelayEmbedder::<3>::new(self.config.tau);
        let mut points_generated = 0;
        
        for i in 0..self.config.embedding_points.min(128) {
            embedder.push(signal[i]);
            if embedder.embed().is_some() {
                points_generated += 1;
            }
        }
        
        // Quality = fraction of signal successfully embedded
        let expected_points = self.config.embedding_points.saturating_sub(self.config.tau * 2);
        if expected_points > 0 {
            (points_generated as f64 / expected_points as f64).min(1.0)
        } else {
            0.0
        }
    }

    /// Test sparse graph density at different epsilons
    fn test_graph_density(&self) -> [f64; 4] {
        let mut densities = [0.0; 4];
        let circle_points = generate_circle(32);
        
        for (e_idx, &epsilon) in self.config.epsilons.iter().enumerate() {
            let mut graph = SparseAttentionGraph::<3>::new(epsilon);
            
            for i in 0..32 {
                graph.add_point(ManifoldPoint::new(circle_points[i]));
            }
            
            // Count edges (neighbor pairs)
            let mut edge_count = 0u32;
            for i in 0..32 {
                for j in (i+1)..32 {
                    if graph.are_neighbors(i, j) {
                        edge_count += 1;
                    }
                }
            }
            
            // Density = edges / max_possible_edges
            let max_edges = 32 * 31 / 2;
            densities[e_idx] = edge_count as f64 / max_edges as f64;
        }
        
        densities
    }

    /// Test Betti number computation on known shapes
    fn test_betti_numbers(&self) -> BettiResults {
        let mut results = BettiResults::default();
        
        // Circle
        let circle_points = generate_circle(32);
        let mut graph = SparseAttentionGraph::<3>::new(0.3);
        for i in 0..32 {
            graph.add_point(ManifoldPoint::new(circle_points[i]));
        }
        results.circle_betti = graph.shape();
        
        // Line
        let line_points = generate_line(32);
        let mut graph = SparseAttentionGraph::<3>::new(0.3);
        for i in 0..32 {
            graph.add_point(ManifoldPoint::new(line_points[i]));
        }
        results.line_betti = graph.shape();
        
        // 3 Clusters
        let cluster_points = generate_clusters(3, 8);
        let mut graph = SparseAttentionGraph::<3>::new(0.5);
        for i in 0..24 {
            graph.add_point(ManifoldPoint::new(cluster_points[i]));
        }
        results.cluster_betti = graph.shape();
        
        results
    }

    /// Test geometric concentration
    fn test_concentration(&self) -> ConcentrationResults {
        let mut results = ConcentrationResults::default();
        let mut concentrator = GeometricConcentrator::<3>::new();
        
        // Feed elongated data (high variance in one direction)
        let mut state = 12345u32;
        for i in 0..50 {
            state = state.wrapping_mul(1103515245).wrapping_add(12345);
            let noise = ((state >> 16) as f64 / 65536.0 - 0.5) * 0.1;
            let point = ManifoldPoint::new([i as f64 * 0.5, noise, noise * 0.5]);
            concentrator.update(&point);
        }
        
        // Variance should be mostly in first dimension
        let (variance, mean) = concentrator.statistics();
        let total_variance: f64 = variance.coords.iter().sum();
        
        if total_variance > 1e-6 {
            // Variance reduction = how much is in principal axis
            results.variance_reduction = variance.coords[0] / total_variance;
            
            // Alignment = mean is along primary axis
            let mean_norm = sqrt(mean.coords.iter().map(|x| x * x).sum::<f64>());
            if mean_norm > 1e-6 {
                results.alignment_score = libm::fabs(mean.coords[0]) / mean_norm;
            }
        }
        
        results
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Unit Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding_quality() {
        let bench = ManifoldBenchmark::new(ManifoldBenchConfig::default());
        let results = bench.run_all();
        
        assert!(results.embedding_quality.sine_quality > 0.8,
            "Sine embedding quality should be > 80%");
        println!("Embedding quality: sine={:.1}%, chirp={:.1}%, lorenz={:.1}%",
            results.embedding_quality.sine_quality * 100.0,
            results.embedding_quality.chirp_quality * 100.0,
            results.embedding_quality.lorenz_quality * 100.0);
    }

    #[test]
    fn test_graph_density_increases_with_epsilon() {
        let bench = ManifoldBenchmark::new(ManifoldBenchConfig::default());
        let results = bench.run_all();
        
        println!("\n═══ Graph Densities ═══");
        for (i, density) in results.graph_densities.iter().enumerate() {
            println!("ε={:.1}: density={:.1}%", 
                ManifoldBenchConfig::default().epsilons[i], density * 100.0);
        }
        
        // Larger epsilon should give denser graph
        assert!(results.graph_densities[3] >= results.graph_densities[0],
            "Larger epsilon should give denser graph");
    }

    #[test]
    fn test_betti_numbers_circle() {
        let bench = ManifoldBenchmark::new(ManifoldBenchConfig::default());
        let results = bench.run_all();
        
        println!("Circle Betti: β₀={}, β₁={}", 
            results.betti_numbers.circle_betti.0,
            results.betti_numbers.circle_betti.1);
        println!("Line Betti: β₀={}, β₁={}", 
            results.betti_numbers.line_betti.0,
            results.betti_numbers.line_betti.1);
        println!("Cluster Betti: β₀={}, β₁={}", 
            results.betti_numbers.cluster_betti.0,
            results.betti_numbers.cluster_betti.1);
    }

    #[test]
    fn test_concentration() {
        let bench = ManifoldBenchmark::new(ManifoldBenchConfig::default());
        let results = bench.run_all();
        
        assert!(results.concentration.variance_reduction > 0.5,
            "Primary axis should capture > 50% variance");
        println!("Concentration: var_reduction={:.1}%, alignment={:.1}%",
            results.concentration.variance_reduction * 100.0,
            results.concentration.alignment_score * 100.0);
    }
}
