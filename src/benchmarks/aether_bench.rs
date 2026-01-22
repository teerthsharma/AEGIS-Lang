//! ═══════════════════════════════════════════════════════════════════════════════
//! AETHER Benchmark: Hierarchical Sparse Attention
//! ═══════════════════════════════════════════════════════════════════════════════
//!
//! Benchmarks for AETHER geometric extensions:
//!   - Pruning ratio at various thresholds
//!   - Upper bound tightness (Cauchy-Schwarz)
//!   - Hierarchical query speedup
//!   - Block compression ratios
//!
//! ═══════════════════════════════════════════════════════════════════════════════

use crate::aether::{BlockMetadata, HierarchicalBlockTree, CompressionStrategy, 
                    select_compression, estimate_compression_ratio, DriftDetector};

// ═══════════════════════════════════════════════════════════════════════════════
// Benchmark Configuration
// ═══════════════════════════════════════════════════════════════════════════════

/// Configuration for AETHER benchmarks
#[derive(Debug, Clone)]
pub struct AetherBenchConfig {
    /// Number of blocks to test
    pub num_blocks: usize,
    /// Points per block
    pub points_per_block: usize,
    /// Pruning thresholds to test
    pub thresholds: [f64; 5],
}

impl Default for AetherBenchConfig {
    fn default() -> Self {
        Self {
            num_blocks: 16,
            points_per_block: 8,
            thresholds: [0.1, 0.3, 0.5, 0.7, 0.9],
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Benchmark Results  
// ═══════════════════════════════════════════════════════════════════════════════

/// Results from AETHER pruning benchmarks
#[derive(Debug, Clone)]
pub struct AetherBenchResults {
    /// Pruning ratio at each threshold
    pub pruning_ratios: [f64; 5],
    /// Mean upper bound tightness (actual/bound ratio)
    pub upper_bound_tightness: f64,
    /// Hierarchical query blocks examined vs total
    pub hierarchical_efficiency: f64,
    /// Compression ratio by strategy
    pub compression_ratios: CompressionRatios,
    /// Drift detection accuracy
    pub drift_detection_accuracy: f64,
}

/// Compression ratios by strategy
#[derive(Debug, Clone, Default)]
pub struct CompressionRatios {
    pub centroid_delta: f64,
    pub int4_quantize: f64,
    pub full_precision: f64,
}

impl AetherBenchResults {
    fn new() -> Self {
        Self {
            pruning_ratios: [0.0; 5],
            upper_bound_tightness: 0.0,
            hierarchical_efficiency: 0.0,
            compression_ratios: CompressionRatios::default(),
            drift_detection_accuracy: 0.0,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Data Generators
// ═══════════════════════════════════════════════════════════════════════════════

/// Generate clustered 3D points
fn generate_clustered_points(n: usize, cluster_center: [f64; 3], spread: f64, seed: u32) -> [[f64; 3]; 8] {
    let mut points = [[0.0; 3]; 8];
    let mut state = seed;
    
    for i in 0..n.min(8) {
        for d in 0..3 {
            state = state.wrapping_mul(1103515245).wrapping_add(12345);
            let rand = ((state >> 16) as f64 / 65536.0) - 0.5;
            points[i][d] = cluster_center[d] + rand * spread;
        }
    }
    points
}

/// Generate query point
fn generate_query(seed: u32) -> [f64; 3] {
    let mut state = seed;
    let mut query = [0.0; 3];
    for d in 0..3 {
        state = state.wrapping_mul(1103515245).wrapping_add(12345);
        query[d] = ((state >> 16) as f64 / 65536.0) * 2.0 - 1.0;
    }
    query
}

// ═══════════════════════════════════════════════════════════════════════════════
// AETHER Benchmark Implementation
// ═══════════════════════════════════════════════════════════════════════════════

/// Run comprehensive AETHER benchmarks
pub struct AetherBenchmark {
    config: AetherBenchConfig,
}

impl AetherBenchmark {
    pub fn new(config: AetherBenchConfig) -> Self {
        Self { config }
    }

    /// Run all AETHER benchmarks
    pub fn run_all(&self) -> AetherBenchResults {
        let mut results = AetherBenchResults::new();
        
        // Generate test blocks
        let mut blocks: [BlockMetadata<3>; 16] = [BlockMetadata::empty(); 16];
        for i in 0..self.config.num_blocks.min(16) {
            let center = [
                (i as f64) * 0.5 - 4.0,
                ((i * 7) % 8) as f64 * 0.5 - 2.0,
                ((i * 13) % 8) as f64 * 0.5 - 2.0,
            ];
            let points = generate_clustered_points(
                self.config.points_per_block, 
                center, 
                0.3, 
                i as u32
            );
            blocks[i] = BlockMetadata::from_points(&points[..self.config.points_per_block.min(8)]);
        }
        
        // Test pruning at different thresholds
        results.pruning_ratios = self.test_pruning_ratios(&blocks);
        
        // Test upper bound tightness
        results.upper_bound_tightness = self.test_upper_bound_tightness(&blocks);
        
        // Test hierarchical efficiency
        results.hierarchical_efficiency = self.test_hierarchical_efficiency(&blocks);
        
        // Test compression ratios
        results.compression_ratios = self.test_compression_ratios(&blocks);
        
        // Test drift detection
        results.drift_detection_accuracy = self.test_drift_detection();
        
        results
    }

    /// Test pruning ratios at different thresholds
    fn test_pruning_ratios(&self, blocks: &[BlockMetadata<3>; 16]) -> [f64; 5] {
        let mut ratios = [0.0; 5];
        let num_queries = 20;
        
        for (t_idx, &threshold) in self.config.thresholds.iter().enumerate() {
            let mut total_pruned = 0u32;
            let mut total_blocks = 0u32;
            
            for q in 0..num_queries {
                let query = generate_query(q as u32 * 17);
                
                for i in 0..self.config.num_blocks.min(16) {
                    total_blocks += 1;
                    if blocks[i].can_prune(&query, threshold) {
                        total_pruned += 1;
                    }
                }
            }
            
            ratios[t_idx] = total_pruned as f64 / total_blocks as f64;
        }
        
        ratios
    }

    /// Test upper bound tightness
    fn test_upper_bound_tightness(&self, blocks: &[BlockMetadata<3>; 16]) -> f64 {
        let num_queries = 50;
        let mut sum_tightness = 0.0;
        let mut count = 0u32;
        
        for q in 0..num_queries {
            let query = generate_query(q as u32 * 23);
            
            for i in 0..self.config.num_blocks.min(16) {
                let upper_bound = blocks[i].upper_bound_score(&query);
                
                // Compute actual max score in block
                // For benchmark purposes, we approximate with centroid score
                let actual = Self::dot(&query, &blocks[i].centroid);
                
                if upper_bound > 1e-6 {
                    // Tightness = actual/upper_bound (1.0 = perfectly tight)
                    sum_tightness += libm::fabs(actual) / upper_bound;
                    count += 1;
                }
            }
        }
        
        if count > 0 { sum_tightness / count as f64 } else { 0.0 }
    }

    /// Dot product helper
    fn dot(a: &[f64; 3], b: &[f64; 3]) -> f64 {
        a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
    }

    /// Test hierarchical query efficiency
    fn test_hierarchical_efficiency(&self, blocks: &[BlockMetadata<3>; 16]) -> f64 {
        let mut tree = HierarchicalBlockTree::<3>::new();
        tree.build_from_blocks(blocks);
        
        let num_queries = 20;
        let mut total_examined = 0u32;
        let total_possible = num_queries * self.config.num_blocks.min(16) as u32;
        
        for q in 0..num_queries {
            let query = generate_query(q as u32 * 31);
            let active_mask = tree.hierarchical_query(&query, 0.5);
            
            for i in 0..self.config.num_blocks.min(16) {
                if active_mask[i] {
                    total_examined += 1;
                }
            }
        }
        
        // Efficiency = 1 - (examined/total)
        1.0 - (total_examined as f64 / total_possible as f64)
    }

    /// Test compression ratios
    fn test_compression_ratios(&self, blocks: &[BlockMetadata<3>; 16]) -> CompressionRatios {
        let mut ratios = CompressionRatios::default();
        let mut cd_count = 0u32;
        let mut i4_count = 0u32;
        let mut fp_count = 0u32;
        
        for i in 0..self.config.num_blocks.min(16) {
            let strategy = select_compression(&blocks[i]);
            let ratio = estimate_compression_ratio(&blocks[i]);
            
            match strategy {
                CompressionStrategy::CentroidDelta => {
                    ratios.centroid_delta += ratio;
                    cd_count += 1;
                }
                CompressionStrategy::Int4Quantize => {
                    ratios.int4_quantize += ratio;
                    i4_count += 1;
                }
                CompressionStrategy::FullPrecision => {
                    ratios.full_precision += ratio;
                    fp_count += 1;
                }
            }
        }
        
        if cd_count > 0 { ratios.centroid_delta /= cd_count as f64; }
        if i4_count > 0 { ratios.int4_quantize /= i4_count as f64; }
        if fp_count > 0 { ratios.full_precision /= fp_count as f64; }
        
        ratios
    }

    /// Test drift detection accuracy
    fn test_drift_detection(&self) -> f64 {
        let mut detector = DriftDetector::<3>::new();
        let mut correct = 0u32;
        let mut total = 0u32;
        
        // Linear trajectory (no drift expected)
        for i in 0..10 {
            let centroid = [i as f64 * 0.1, 0.0, 0.0];
            let drift = detector.update(&centroid);
            if i > 1 && drift < 0.5 {
                correct += 1;
            }
            if i > 1 {
                total += 1;
            }
        }
        
        // Sudden jump (drift expected)
        let drift = detector.update(&[5.0, 5.0, 5.0]);
        if drift > 1.0 {
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
    fn test_pruning_ratios() {
        let bench = AetherBenchmark::new(AetherBenchConfig::default());
        let results = bench.run_all();
        
        println!("\n═══ AETHER Pruning Ratios ═══");
        for (i, ratio) in results.pruning_ratios.iter().enumerate() {
            println!("Threshold {:.1}: {:.1}% pruned", 
                AetherBenchConfig::default().thresholds[i], ratio * 100.0);
        }
        
        // Higher threshold should prune more
        assert!(results.pruning_ratios[4] >= results.pruning_ratios[0],
            "Higher threshold should prune more blocks");
    }

    #[test]
    fn test_upper_bound_valid() {
        let bench = AetherBenchmark::new(AetherBenchConfig::default());
        let results = bench.run_all();
        
        // Tightness should be between 0 and 1
        assert!(results.upper_bound_tightness >= 0.0 && results.upper_bound_tightness <= 1.0,
            "Tightness={:.3} should be in [0,1]", results.upper_bound_tightness);
        println!("Upper bound tightness: {:.1}%", results.upper_bound_tightness * 100.0);
    }

    #[test]
    fn test_hierarchical_efficiency() {
        let bench = AetherBenchmark::new(AetherBenchConfig::default());
        let results = bench.run_all();
        
        assert!(results.hierarchical_efficiency >= 0.0,
            "Efficiency should be non-negative");
        println!("Hierarchical efficiency: {:.1}% blocks saved", 
            results.hierarchical_efficiency * 100.0);
    }

    #[test]
    fn test_drift_detection() {
        let bench = AetherBenchmark::new(AetherBenchConfig::default());
        let results = bench.run_all();
        
        assert!(results.drift_detection_accuracy >= 0.7,
            "Drift detection should be >= 70% accurate");
        println!("Drift detection accuracy: {:.1}%", 
            results.drift_detection_accuracy * 100.0);
    }
}
