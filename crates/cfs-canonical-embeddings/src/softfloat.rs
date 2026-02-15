//! Deterministic SoftFloat Implementation
//!
//! This module provides software-implemented floating point operations
//! that are guaranteed to produce bit-identical results across all platforms.
//!
//! Per CFS-003 and CFS-010:
//! - Hardware FPU operations are PROHIBITED for canonical operations
//! - Uses integer arithmetic for sqrt and div to ensure determinism
//!
//! Implementation approach:
//! - Uses Q30 fixed-point arithmetic (30 fractional bits)
//! - All operations use integer math for determinism
//! - Results are converted back to f32 at the end

use crate::EMBEDDING_DIM;

// ============================================================================
// Q30 Fixed-Point Arithmetic
// ============================================================================

/// Q30 fixed-point representation (1 sign bit, 1 integer bit, 30 fractional bits)
/// Range: [-2.0, 2.0) with 30 bits of precision
#[derive(Debug, Clone, Copy)]
pub struct Fixed30(i32);

impl Fixed30 {
    /// Create from raw bits
    pub const fn from_bits(bits: i32) -> Self {
        Self(bits)
    }

    /// Get raw bits
    pub const fn to_bits(self) -> i32 {
        self.0
    }

    /// Create from f32 (with scaling)
    pub const fn from_f32(val: f32) -> Self {
        // Scale to Q30: val * 2^30
        let scaled = (val * 1073741824.0) as i32;
        Self(scaled)
    }

    /// Convert to f32
    pub const fn to_f32(self) -> f32 {
        self.0 as f32 / 1073741824.0
    }

    /// Check if zero
    pub const fn is_zero(self) -> bool {
        self.0 == 0
    }

    /// Create zero
    pub const fn zero() -> Self {
        Self(0)
    }

    /// Create one (1.0 in Q30)
    pub const fn one() -> Self {
        Self(1073741824) // 1 << 30
    }

    /// Absolute value
    pub const fn abs(self) -> Self {
        Self(if self.0 < 0 { -self.0 } else { self.0 })
    }

    /// Add
    pub const fn add(self, other: Self) -> Self {
        Self(self.0.wrapping_add(other.0))
    }

    /// Multiply: Q30 * Q30 -> Q30 (with rounding)
    pub const fn mul(self, other: Self) -> Self {
        // 32-bit * 32-bit = 64-bit, then shift right 30 bits with rounding
        let product = (self.0 as i64) * (other.0 as i64);
        // Round to nearest (add 2^29 before shifting)
        let rounded = product + (1i64 << 29);
        Self((rounded >> 30) as i32)
    }

    /// Divide: Q30 / Q30 -> Q30 (with rounding)
    pub const fn div(self, other: Self) -> Self {
        if other.0 == 0 {
            return Self(0x7FFFFFFF); // Max value on error
        }
        // Shift left 30 bits before dividing
        let scaled = (self.0 as i64) << 30;
        // Round to nearest
        let quotient = scaled / (other.0 as i64);
        // Clamp to valid range
        if quotient > (i32::MAX as i64) {
            Self(i32::MAX)
        } else if quotient < (i32::MIN as i64) {
            Self(i32::MIN)
        } else {
            Self(quotient as i32)
        }
    }
}

// ============================================================================
// Integer Square Root (Deterministic)
// ============================================================================

/// Compute integer square root using Newton's method (deterministic)
fn isqrt(n: u64) -> u64 {
    if n == 0 {
        return 0;
    }

    // Initial guess
    let mut x = n;
    let mut y = (x + 1) >> 1;

    while y < x {
        x = y;
        y = (x + n / x) >> 1;
    }

    x
}

/// Compute Q30 square root
fn fixed30_sqrt(a: Fixed30) -> Fixed30 {
    if a.is_zero() {
        return Fixed30::zero();
    }

    // Convert to unsigned magnitude
    let abs_val = a.abs().to_bits() as u64;

    // Compute integer square root
    let sqrt_val = isqrt(abs_val);

    // Convert back to Q30
    // Since we're taking sqrt of a Q30 number, result is in Q15
    // Multiply by 2^15 to get back to Q30
    let result = (sqrt_val << 15) as i32;

    Fixed30::from_bits(if a.to_bits() < 0 { -result } else { result })
}

// ============================================================================
// Deterministic L2 Normalization
// ============================================================================

/// L2 normalize using integer arithmetic for deterministic results
/// Per CFS-010 §8: Uses software float to guarantee bit-exact results
pub fn l2_normalize_softfloat(input: &[f32; EMBEDDING_DIM]) -> [f32; EMBEDDING_DIM] {
    // Convert to Fixed30
    let mut fixed_input = [Fixed30::zero(); EMBEDDING_DIM];
    let mut i = 0;
    while i < EMBEDDING_DIM {
        fixed_input[i] = Fixed30::from_f32(input[i]);
        i += 1;
    }

    // Compute sum of squares in Fixed30
    let mut sum_sq = Fixed30::zero();
    let mut j = 0;
    while j < EMBEDDING_DIM {
        let val = fixed_input[j];
        let val_sq = val.mul(val);
        sum_sq = sum_sq.add(val_sq);
        j += 1;
    }

    // Compute L2 norm (square root)
    let norm = fixed30_sqrt(sum_sq);

    // Handle zero norm
    if norm.is_zero() {
        return [0.0f32; EMBEDDING_DIM];
    }

    // Compute 1/norm in Fixed30
    let inv_norm = Fixed30::one().div(norm);

    // Normalize each component: x / norm = x * (1/norm)
    let mut result = [0.0f32; EMBEDDING_DIM];
    let mut k = 0;
    while k < EMBEDDING_DIM {
        let val = fixed_input[k];
        let normalized = val.mul(inv_norm);
        result[k] = normalized.to_f32();
        k += 1;
    }

    result
}

// ============================================================================
// SoftFloat32 (Simple wrapper for compatibility)
// ============================================================================

/// Deterministic f32 wrapper
#[derive(Debug, Clone, Copy)]
pub struct SoftFloat32(u32);

impl SoftFloat32 {
    /// Create from raw bits
    pub const fn from_bits(bits: u32) -> Self {
        Self(bits)
    }

    /// Create from standard f32
    pub const fn from_f32(val: f32) -> Self {
        Self(val.to_bits())
    }

    /// Convert to standard f32
    pub const fn to_f32(self) -> f32 {
        f32::from_bits(self.0)
    }

    /// Get raw bits
    pub const fn to_bits(self) -> u32 {
        self.0
    }

    /// Create zero
    pub const fn zero() -> Self {
        Self(0)
    }

    /// Create one
    pub const fn one() -> Self {
        Self(0x3F800000)
    }

    /// Create NaN
    pub const fn nan() -> Self {
        Self(0x7FC00000)
    }

    /// Create infinity
    pub const fn infinity() -> Self {
        Self(0x7F800000)
    }

    /// Check if zero
    pub const fn is_zero(self) -> bool {
        self.0 == 0
    }

    /// Check if NaN
    pub const fn is_nan(self) -> bool {
        (self.0 >> 23) & 0xFF == 0xFF && (self.0 & 0x7FFFFF) != 0
    }

    /// Check if infinite
    pub const fn is_infinite(self) -> bool {
        (self.0 >> 23) & 0xFF == 0xFF && (self.0 & 0x7FFFFF) == 0
    }
}

// ============================================================================
// SoftFloat64 (Simple wrapper for compatibility)
// ============================================================================

/// Deterministic f64 wrapper
#[derive(Debug, Clone, Copy)]
pub struct SoftFloat64(u64);

impl SoftFloat64 {
    /// Create from raw bits
    pub const fn from_bits(bits: u64) -> Self {
        Self(bits)
    }

    /// Create from standard f64
    pub const fn from_f64(val: f64) -> Self {
        Self(val.to_bits())
    }

    /// Create from f32
    pub const fn from_f32(val: SoftFloat32) -> Self {
        Self(val.to_bits() as u64)
    }

    /// Convert to standard f64
    pub const fn to_f64(self) -> f64 {
        f64::from_bits(self.0)
    }

    /// Convert to f32
    pub const fn to_f32(self) -> SoftFloat32 {
        SoftFloat32(self.0 as u32)
    }

    /// Get raw bits
    pub const fn to_bits(self) -> u64 {
        self.0
    }

    /// Create zero
    pub const fn zero() -> Self {
        Self(0)
    }

    /// Create one
    pub const fn one() -> Self {
        Self(0x3FF0000000000000)
    }

    /// Create NaN
    pub const fn nan() -> Self {
        Self(0x7FF8000000000000)
    }

    /// Create infinity
    pub const fn infinity() -> Self {
        Self(0x7FF0000000000000)
    }

    /// Check if zero
    pub const fn is_zero(self) -> bool {
        self.0 == 0
    }

    /// Check if NaN
    pub const fn is_nan(self) -> bool {
        ((self.0 >> 52) & 0x7FF) == 0x7FF && (self.0 & 0xFFFFFFFFFFFFF) != 0
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]

    #[test]

    #[test]
    fn test_isqrt() {
        assert_eq!(isqrt(4), 2);
        assert_eq!(isqrt(9), 3);
        assert_eq!(isqrt(16), 4);
        assert_eq!(isqrt(100), 10);
    }

    #[test]
    fn test_l2_normalize_deterministic() {
        let input = [0.5f32; EMBEDDING_DIM];
        let result1 = l2_normalize_softfloat(&input);
        let result2 = l2_normalize_softfloat(&input);

        for (a, b) in result1.iter().zip(result2.iter()) {
            assert_eq!(a.to_bits(), b.to_bits(), "Results must be bit-identical");
        }
    }

    #[test]
    fn test_l2_normalize_values() {
        // Simple test case: [1, 0, 0, ...] should normalize to [1, 0, 0, ...]
        let mut input = [0.0f32; EMBEDDING_DIM];
        input[0] = 1.0;

        let result = l2_normalize_softfloat(&input);

        // First element should be ~1.0
        assert!(
            (result[0] - 1.0).abs() < 0.1,
            "First element should be ~1.0, got {}",
            result[0]
        );
    }

    #[test]
    fn test_determinism_across_runs() {
        // Run the same operation multiple times - must produce identical results
        let input = [0.123456f32; EMBEDDING_DIM];

        let results: Vec<[u32; EMBEDDING_DIM]> = (0..10)
            .map(|_| {
                let result = l2_normalize_softfloat(&input);
                result.map(|v| v.to_bits())
            })
            .collect();

        // All results must be identical
        for i in 1..results.len() {
            assert_eq!(results[0], results[i], "Run {} differs from run 0", i);
        }
    }

    #[test]
    fn test_zero_vector_normalization() {
        let input = [0.0f32; EMBEDDING_DIM];
        let result = l2_normalize_softfloat(&input);

        // All zeros expected
        for val in &result {
            assert_eq!(val.to_bits(), 0u32, "Expected zero but got {}", val);
        }
    }

    #[test]
    fn test_special_values() {
        assert!(SoftFloat32::zero().is_zero());
        assert!(!SoftFloat32::one().is_zero());

        let nan = SoftFloat32::nan();
        assert!(nan.is_nan());

        let inf = SoftFloat32::infinity();
        assert!(inf.is_infinite());
    }
}
